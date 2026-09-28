from __future__ import annotations

from typing import Any, Callable
import hashlib
import os
import tempfile
import time

import torch
from torch.nn import functional as F

from rl.policy_model import TacticalPolicyValueNetwork


PPO_CHECKPOINT_SCHEMA_VERSION = 1


class PpoTrainer:
    def __init__(self, feature_spec: dict[str, Any], hyperparameters: dict[str, Any], seed: int, device: torch.device | str):
        if feature_spec.get("schemaVersion") != 2:
            raise ValueError("PPO requires Feature Spec schemaVersion 2")
        self.feature_spec = feature_spec
        self.hyperparameters = dict(hyperparameters)
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.model = TacticalPolicyValueNetwork(feature_spec).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(hyperparameters["learningRate"]))
        self._act_forward = self.model.forward_prepared_batch
        compile_mode = os.environ.get("PPO_ACT_COMPILE_MODE")
        if compile_mode:
            if self.device.type != "cuda":
                raise ValueError("PPO_ACT_COMPILE_MODE is only supported for CUDA action inference")
            if compile_mode != "reduce-overhead":
                raise ValueError(f"Unsupported PPO_ACT_COMPILE_MODE: {compile_mode}")
            self._act_forward = torch.compile(
                self.model.forward_prepared_batch,
                mode=compile_mode,
                dynamic=True,
            )
        self.update_count = 0
        self.episode_count = 0
        self._accumulation: dict[str, Any] | None = None
        self._act_streams: list[torch.cuda.Stream] = []
        self._act_state_branch_cache: dict[str, tuple[Any, torch.Tensor]] = {}
        self._act_cuda_graph_enabled = (
            os.environ.get("PPO_ACT_CUDA_GRAPH_HOT") == "1"
        )
        if self._act_cuda_graph_enabled and self.device.type != "cuda":
            raise ValueError("PPO_ACT_CUDA_GRAPH_HOT requires CUDA")
        if self._act_cuda_graph_enabled and compile_mode:
            raise ValueError(
                "PPO_ACT_CUDA_GRAPH_HOT cannot be combined with PPO_ACT_COMPILE_MODE"
            )
        self._act_cuda_graph_min_hits = int(
            os.environ.get("PPO_ACT_CUDA_GRAPH_MIN_HITS", "3")
        )
        self._act_cuda_graph_max_entries = int(
            os.environ.get("PPO_ACT_CUDA_GRAPH_MAX_ENTRIES", "12")
        )
        if self._act_cuda_graph_min_hits < 2:
            raise ValueError("PPO_ACT_CUDA_GRAPH_MIN_HITS must be >= 2")
        if self._act_cuda_graph_max_entries <= 0:
            raise ValueError("PPO_ACT_CUDA_GRAPH_MAX_ENTRIES must be positive")
        self._act_cuda_graph_seen: dict[tuple[Any, ...], int] = {}
        self._act_cuda_graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._act_cuda_graph_captures = 0
        self._act_cuda_graph_replays = 0
        self._act_cuda_graph_fallbacks = 0

    def load_initial_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != 2:
            raise ValueError("PPO initial checkpoint must use schemaVersion 2")
        if checkpoint.get("featureSpec") != self.feature_spec:
            raise ValueError("PPO initial checkpoint Feature Spec mismatch")
        self.model.load_state_dict(checkpoint.get("modelStateDict"), strict=True)
        self._act_state_branch_cache.clear()

    def act(self, observation: dict[str, Any], actions: list[list[float]]) -> dict[str, float | int]:
        if not actions:
            raise ValueError("PPO cannot act without legal actions")
        self.model.eval()
        with torch.no_grad():
            logits, value, _, _ = self.model(observation, actions)
            distribution = torch.distributions.Categorical(logits=logits)
            selected = distribution.sample()
            log_probability = distribution.log_prob(selected)
        return {
            "actionIndex": int(selected.item()),
            "logProbability": float(log_probability.item()),
            "value": float(value.item()),
        }

    def update(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("PPO update requires at least one sample")
        self._act_state_branch_cache.clear()
        self.model.train(True)
        actions = [sample["actions"] for sample in samples]
        selected = torch.tensor([int(sample["selectedActionIndex"]) for sample in samples], dtype=torch.long, device=self.device)
        old_log_probabilities = torch.tensor([float(sample["oldLogProbability"]) for sample in samples], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([float(sample["advantage"]) for sample in samples], dtype=torch.float32, device=self.device)
        returns = torch.tensor([float(sample["return"]) for sample in samples], dtype=torch.float32, device=self.device)
        if not all(torch.isfinite(value).all() for value in (old_log_probabilities, advantages, returns)):
            raise FloatingPointError("PPO input contains NaN or Inf")
        logits, values, _, _, action_mask = self.model.forward_batch([sample["observation"] for sample in samples], actions)
        if torch.any(selected < 0) or torch.any(selected >= action_mask.shape[1]) or not torch.all(action_mask.gather(1, selected.unsqueeze(1))):
            raise ValueError("PPO selected action is outside the legal action mask")
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(selected)
        entropy = distribution.entropy().mean()
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        clip_epsilon = float(self.hyperparameters["clipEpsilon"])
        policy_loss = -torch.minimum(ratio * advantages, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + float(self.hyperparameters["valueCoefficient"]) * value_loss - float(self.hyperparameters["entropyCoefficient"]) * entropy
        if not all(torch.isfinite(value).all() for value in (logits[action_mask], values, log_probabilities, entropy, policy_loss, value_loss, loss)):
            raise FloatingPointError("PPO calculation contains NaN or Inf")
        policy_before = self.parameter_hash(exclude_prefix="value_head.")
        value_before = self.parameter_hash(include_prefix="value_head.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("PPO gradient contains NaN or Inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.hyperparameters["maxGradientNorm"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("PPO gradient norm is NaN or Inf")
        self.optimizer.step()
        self.update_count += 1
        policy_after = self.parameter_hash(exclude_prefix="value_head.")
        value_after = self.parameter_hash(include_prefix="value_head.")
        return {
            "sampleCount": len(samples), "loss": float(loss.detach().item()),
            "policyLoss": float(policy_loss.detach().item()), "valueLoss": float(value_loss.detach().item()),
            "entropy": float(entropy.detach().item()), "gradientNorm": float(gradient_norm.detach().item()),
            "policyParametersChanged": policy_before != policy_after,
            "valueParametersChanged": value_before != value_after,
            "updateCount": self.update_count,
        }

    def act_prepared(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
        profile_stage: Callable[[str, float], None] | None = None,
        fast_guard_mode: bool = False,
        manual_categorical_mode: bool = False,
        state_branch_fingerprints: dict[str, bytes] | None = None,
    ) -> dict[str, float | int]:
        if prepared_actions.shape[0] != 1:
            raise ValueError("Packed PPO act requires exactly one sample")
        if not fast_guard_mode and not bool(action_mask[0].any()):
            raise ValueError("Packed PPO act requires legal actions")

        def timed(stage: str, operation: Callable[[], Any]) -> Any:
            if profile_stage is None:
                return operation()
            # CUDA kernels are asynchronous; synchronize only in opt-in diagnostics.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            try:
                return operation()
            finally:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                profile_stage(stage, time.perf_counter() - start)

        self.model.eval()
        with torch.no_grad():
            if state_branch_fingerprints is not None:
                forward_operation = lambda: self.model.forward_prepared_batch_cached_state(
                    prepared_observations,
                    prepared_actions,
                    action_mask,
                    state_branch_fingerprints,
                    self._act_state_branch_cache,
                )
            else:
                forward_operation = (
                    (lambda: self._act_forward(prepared_observations, prepared_actions, action_mask))
                    if profile_stage is None
                    else (lambda: self.model.forward_prepared_batch(
                        prepared_observations, prepared_actions, action_mask,
                        profile_stage=profile_stage,
                    ))
                )
            logits, values, _, _, returned_mask = timed(
                "act_model_forward",
                forward_operation,
            )

            finite_flag = None
            if fast_guard_mode:
                finite_flag = torch.logical_and(
                    torch.isfinite(logits[returned_mask]).all(),
                    torch.isfinite(values).all(),
                )
            else:
                def validate_outputs() -> None:
                    if not torch.isfinite(logits[returned_mask]).all() or not torch.isfinite(values).all():
                        raise FloatingPointError("Packed PPO action calculation contains NaN or Inf")

                timed("act_finite_checks", validate_outputs)

            if manual_categorical_mode:
                normalized_logits = timed(
                    "act_distribution_init",
                    lambda: logits[0] - logits[0].logsumexp(dim=-1, keepdim=True),
                )
                probabilities = normalized_logits.softmax(dim=-1)

                def sample_manual() -> torch.Tensor:
                    samples_2d = torch.multinomial(
                        probabilities.reshape(-1, probabilities.shape[-1]),
                        1,
                        replacement=True,
                    ).T
                    return samples_2d.reshape(())

                selected = timed("act_sampling", sample_manual)

                def log_prob_manual() -> torch.Tensor:
                    return normalized_logits.gather(
                        -1,
                        selected.long().unsqueeze(-1),
                    ).squeeze(-1)

                log_probability = timed("act_log_probability", log_prob_manual)
            else:
                distribution = timed(
                    "act_distribution_init",
                    lambda: torch.distributions.Categorical(logits=logits[0]),
                )
                selected = timed("act_sampling", distribution.sample)
                log_probability = timed(
                    "act_log_probability", lambda: distribution.log_prob(selected)
                )

        if fast_guard_mode:
            host_values = timed(
                "act_host_scalars",
                lambda: torch.stack((
                    selected.to(dtype=torch.float32),
                    log_probability,
                    values[0],
                    finite_flag.to(dtype=torch.float32),
                )).tolist(),
            )
            if not bool(host_values[3]):
                raise FloatingPointError("Packed PPO action calculation contains NaN or Inf")
            return {
                "actionIndex": int(host_values[0]),
                "logProbability": float(host_values[1]),
                "value": float(host_values[2]),
            }

        return timed("act_host_scalars", lambda: {
            "actionIndex": int(selected.item()),
            "logProbability": float(log_probability.item()),
            "value": float(values[0].item()),
        })

    @staticmethod
    def _act_graph_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
            str(tensor.dtype),
        )

    def _act_graph_signature(
        self,
        prepared: dict[str, Any],
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Any, ...]:
        masked = tuple(
            (
                key,
                self._act_graph_tensor_signature(table),
                self._act_graph_tensor_signature(mask),
            )
            for key, (table, mask) in sorted(prepared["masked"].items())
        )
        strategic = tuple(
            (
                key,
                self._act_graph_tensor_signature(table),
                self._act_graph_tensor_signature(mask),
            )
            for key, (table, mask) in sorted(prepared["strategic"].items())
        )
        nonempty = tuple(
            (key, bool(value))
            for key, value in sorted((prepared.get("_nonempty") or {}).items())
        )
        valid_prefix = tuple(
            (key, int(value))
            for key, value in sorted(
                (prepared.get("_validPrefixCount") or {}).items()
            )
        )
        return (
            ("global", self._act_graph_tensor_signature(prepared["global"])),
            (
                "strategicGlobal",
                self._act_graph_tensor_signature(prepared["strategicGlobal"]),
            ),
            ("masked", masked),
            (
                "map",
                self._act_graph_tensor_signature(prepared["map"][0]),
                self._act_graph_tensor_signature(prepared["map"][1]),
            ),
            ("strategic", strategic),
            ("actions", self._act_graph_tensor_signature(actions)),
            ("actionMask", self._act_graph_tensor_signature(action_mask)),
            ("nonempty", nonempty),
            ("validPrefix", valid_prefix),
        )

    @staticmethod
    def _act_graph_static_tensor(tensor: torch.Tensor) -> torch.Tensor:
        static = torch.empty_strided(
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        static.copy_(tensor)
        return static

    def _act_graph_static_prepared(
        self,
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        static: dict[str, Any] = {
            "global": self._act_graph_static_tensor(prepared["global"]),
            "strategicGlobal": self._act_graph_static_tensor(
                prepared["strategicGlobal"]
            ),
            "masked": {
                key: (
                    self._act_graph_static_tensor(table),
                    self._act_graph_static_tensor(mask),
                )
                for key, (table, mask) in prepared["masked"].items()
            },
            "map": (
                self._act_graph_static_tensor(prepared["map"][0]),
                self._act_graph_static_tensor(prepared["map"][1]),
            ),
            "strategic": {
                key: (
                    self._act_graph_static_tensor(table),
                    self._act_graph_static_tensor(mask),
                )
                for key, (table, mask) in prepared["strategic"].items()
            },
        }
        if "_nonempty" in prepared:
            static["_nonempty"] = dict(prepared["_nonempty"])
        if "_validPrefixCount" in prepared:
            static["_validPrefixCount"] = dict(prepared["_validPrefixCount"])
        return static

    @staticmethod
    def _act_graph_copy_prepared(
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        target["global"].copy_(source["global"])
        target["strategicGlobal"].copy_(source["strategicGlobal"])
        for key, (table, mask) in target["masked"].items():
            source_table, source_mask = source["masked"][key]
            table.copy_(source_table)
            mask.copy_(source_mask)
        target["map"][0].copy_(source["map"][0])
        target["map"][1].copy_(source["map"][1])
        for key, (table, mask) in target["strategic"].items():
            source_table, source_mask = source["strategic"][key]
            table.copy_(source_table)
            mask.copy_(source_mask)

    def _capture_act_cuda_graph(
        self,
        prepared: dict[str, Any],
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, Any]:
        static_prepared = self._act_graph_static_prepared(prepared)
        static_actions = self._act_graph_static_tensor(actions)
        static_action_mask = self._act_graph_static_tensor(action_mask)

        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                self.model.forward_prepared_batch(
                    static_prepared,
                    static_actions,
                    static_action_mask,
                )
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self.model.forward_prepared_batch(
                static_prepared,
                static_actions,
                static_action_mask,
            )
        torch.cuda.synchronize(self.device)

        return {
            "prepared": static_prepared,
            "actions": static_actions,
            "actionMask": static_action_mask,
            "graph": graph,
            "outputs": outputs,
        }

    def _act_forward_hot_cuda_graph(
        self,
        prepared: dict[str, Any],
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._act_cuda_graph_enabled:
            return self._act_forward(prepared, actions, action_mask)

        signature = self._act_graph_signature(
            prepared,
            actions,
            action_mask,
        )
        seen = self._act_cuda_graph_seen.get(signature, 0) + 1
        self._act_cuda_graph_seen[signature] = seen

        entry = self._act_cuda_graph_cache.get(signature)
        if entry is None:
            if (
                seen < self._act_cuda_graph_min_hits
                or len(self._act_cuda_graph_cache)
                >= self._act_cuda_graph_max_entries
            ):
                self._act_cuda_graph_fallbacks += 1
                return self._act_forward(prepared, actions, action_mask)
            entry = self._capture_act_cuda_graph(
                prepared,
                actions,
                action_mask,
            )
            self._act_cuda_graph_cache[signature] = entry
            self._act_cuda_graph_captures += 1

        self._act_graph_copy_prepared(entry["prepared"], prepared)
        entry["actions"].copy_(actions)
        entry["actionMask"].copy_(action_mask)
        entry["graph"].replay()
        self._act_cuda_graph_replays += 1
        return entry["outputs"]

    def act_cuda_graph_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._act_cuda_graph_enabled,
            "minHits": self._act_cuda_graph_min_hits,
            "maxEntries": self._act_cuda_graph_max_entries,
            "seenSignatures": len(self._act_cuda_graph_seen),
            "capturedGraphs": self._act_cuda_graph_captures,
            "graphReplays": self._act_cuda_graph_replays,
            "fallbackForwards": self._act_cuda_graph_fallbacks,
        }

    def act_prepared_batch(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
        manual_categorical_mode: bool = False,
    ) -> dict[str, list[float] | list[int]]:
        batch_size = int(prepared_actions.shape[0])
        if batch_size <= 0 or action_mask.shape[0] != batch_size:
            raise ValueError("Packed PPO batch act requires a non-empty aligned batch")
        if not bool(action_mask.any(dim=1).all()):
            raise ValueError("Packed PPO batch act requires legal actions for every sample")

        self.model.eval()
        with torch.no_grad():
            logits, values, _, _, returned_mask = (
                self._act_forward_hot_cuda_graph(
                    prepared_observations,
                    prepared_actions,
                    action_mask,
                )
            )
            if not torch.isfinite(logits[returned_mask]).all() or not torch.isfinite(values).all():
                raise FloatingPointError("Packed PPO batch action calculation contains NaN or Inf")
            if manual_categorical_mode:
                normalized_logits = logits - logits.logsumexp(
                    dim=-1,
                    keepdim=True,
                )
                probabilities = normalized_logits.softmax(dim=-1)
                selected = torch.multinomial(
                    probabilities,
                    1,
                    replacement=True,
                ).squeeze(-1)
                log_probabilities = normalized_logits.gather(
                    -1,
                    selected.unsqueeze(-1),
                ).squeeze(-1)
            else:
                distribution = torch.distributions.Categorical(logits=logits)
                selected = distribution.sample()
                log_probabilities = distribution.log_prob(selected)

        return {
            "actionIndices": [int(value) for value in selected.tolist()],
            "logProbabilities": [float(value) for value in log_probabilities.tolist()],
            "values": [float(value) for value in values.tolist()],
        }

    def act_prepared_stream_batch(
        self,
        samples: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]],
    ) -> dict[str, list[float] | list[int]]:
        if self.device.type != "cuda":
            raise ValueError("Packed PPO stream-batch act requires CUDA")
        if not samples:
            raise ValueError("Packed PPO stream-batch act requires samples")

        for prepared_observations, prepared_actions, action_mask in samples:
            if prepared_actions.shape[0] != 1 or action_mask.shape[0] != 1:
                raise ValueError("Each PPO stream-batch sample must have batch size one")

        while len(self._act_streams) < len(samples):
            self._act_streams.append(torch.cuda.Stream(device=self.device))

        self.model.eval()
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for index, (prepared_observations, prepared_actions, action_mask) in enumerate(samples):
                stream = self._act_streams[index]
                with torch.cuda.stream(stream):
                    logits, values, _, _, returned_mask = self._act_forward(
                        prepared_observations,
                        prepared_actions,
                        action_mask,
                    )
                    outputs.append((logits, values, returned_mask))

            # All batch-1 forwards are now queued. Wait once before preserving
            # the original serial sampling/RNG order below.
            for stream in self._act_streams[:len(samples)]:
                stream.synchronize()

            action_indices: list[int] = []
            log_probabilities: list[float] = []
            values_out: list[float] = []

            for sample_index, ((_, _, action_mask), (logits, values, returned_mask)) in enumerate(zip(samples, outputs)):
                if not bool(action_mask[0].any()):
                    raise ValueError(
                        f"Packed PPO stream-batch sample {sample_index} has no legal actions"
                    )
                if not torch.isfinite(logits[returned_mask]).all() or not torch.isfinite(values).all():
                    raise FloatingPointError(
                        f"Packed PPO stream-batch sample {sample_index} contains NaN or Inf"
                    )

                normalized_logits = logits[0] - logits[0].logsumexp(
                    dim=-1,
                    keepdim=True,
                )
                probabilities = normalized_logits.softmax(dim=-1)
                selected = torch.multinomial(
                    probabilities.reshape(-1, probabilities.shape[-1]),
                    1,
                    replacement=True,
                ).T.reshape(())
                log_probability = normalized_logits.gather(
                    -1,
                    selected.long().unsqueeze(-1),
                ).squeeze(-1)

                # Keep host extraction in serial sample order as an additional
                # equivalence guard against changing RNG/launch ordering.
                action_indices.append(int(selected.item()))
                log_probabilities.append(float(log_probability.item()))
                values_out.append(float(values[0].item()))

        return {
            "actionIndices": action_indices,
            "logProbabilities": log_probabilities,
            "values": values_out,
        }

    def begin_accumulated_update(self, total_samples: int) -> dict[str, int]:
        if self._accumulation is not None:
            raise RuntimeError("A PPO accumulated update is already active")
        self._act_state_branch_cache.clear()
        self._act_cuda_graph_cache.clear()
        if not isinstance(total_samples, int) or isinstance(total_samples, bool) or total_samples <= 0:
            raise ValueError("PPO total sample count must be a positive integer")
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulation = {
            "totalSamples": total_samples,
            "processedSamples": 0,
            "policyLossSum": 0.0,
            "valueLossSum": 0.0,
            "entropySum": 0.0,
            "policyBefore": self.parameter_hash(exclude_prefix="value_head."),
            "valueBefore": self.parameter_hash(include_prefix="value_head."),
        }
        return {"totalSamples": total_samples}

    def accumulate_prepared_chunk(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
        selected: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, int]:
        state = self._accumulation
        if state is None:
            raise RuntimeError("No PPO accumulated update is active")
        chunk_size = int(selected.shape[0])
        if chunk_size <= 0 or state["processedSamples"] + chunk_size > state["totalSamples"]:
            raise ValueError("PPO accumulated chunk exceeds declared sample count")
        if not all(torch.isfinite(value).all() for value in (old_log_probabilities, advantages, returns)):
            raise FloatingPointError("PPO packed input contains NaN or Inf")
        logits, values, _, _, returned_mask = self.model.forward_prepared_batch(
            prepared_observations, prepared_actions, action_mask
        )
        if torch.any(selected < 0) or torch.any(selected >= returned_mask.shape[1]) or not torch.all(returned_mask.gather(1, selected.unsqueeze(1))):
            raise ValueError("PPO selected action is outside the legal action mask")
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(selected)
        entropies = distribution.entropy()
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        clip_epsilon = float(self.hyperparameters["clipEpsilon"])
        clipped_objective = torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages,
        )
        squared_errors = F.mse_loss(values, returns, reduction="none")
        if not all(torch.isfinite(value).all() for value in (logits[returned_mask], values, log_probabilities, entropies, clipped_objective, squared_errors)):
            raise FloatingPointError("PPO packed calculation contains NaN or Inf")
        total_samples = state["totalSamples"]
        chunk_loss = (
            -clipped_objective.sum()
            + float(self.hyperparameters["valueCoefficient"]) * squared_errors.sum()
            - float(self.hyperparameters["entropyCoefficient"]) * entropies.sum()
        ) / total_samples
        if not torch.isfinite(chunk_loss):
            raise FloatingPointError("PPO packed loss contains NaN or Inf")
        chunk_loss.backward()
        state["processedSamples"] += chunk_size
        state["policyLossSum"] += float((-clipped_objective.sum()).detach().item())
        state["valueLossSum"] += float(squared_errors.sum().detach().item())
        state["entropySum"] += float(entropies.sum().detach().item())
        return {"acceptedSamples": chunk_size, "accumulatedSamples": state["processedSamples"]}

    def finish_accumulated_update(self) -> dict[str, Any]:
        state = self._accumulation
        if state is None:
            raise RuntimeError("No PPO accumulated update is active")
        if state["processedSamples"] != state["totalSamples"]:
            raise ValueError(
                f"PPO accumulated sample count mismatch: {state['processedSamples']} != {state['totalSamples']}"
            )
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("PPO gradient contains NaN or Inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float(self.hyperparameters["maxGradientNorm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("PPO gradient norm is NaN or Inf")
        equivalence_diagnostics = os.environ.get("PPO_EQUIVALENCE_DIAGNOSTICS") == "1"
        gradient_hash = self.gradient_hash() if equivalence_diagnostics else None
        self.optimizer.step()
        self.update_count += 1
        total = state["totalSamples"]
        policy_loss = state["policyLossSum"] / total
        value_loss = state["valueLossSum"] / total
        entropy = state["entropySum"] / total
        loss = policy_loss + float(self.hyperparameters["valueCoefficient"]) * value_loss - float(self.hyperparameters["entropyCoefficient"]) * entropy
        result = {
            "sampleCount": total,
            "loss": loss,
            "policyLoss": policy_loss,
            "valueLoss": value_loss,
            "entropy": entropy,
            "gradientNorm": float(gradient_norm.detach().item()),
            "policyParametersChanged": state["policyBefore"] != self.parameter_hash(exclude_prefix="value_head."),
            "valueParametersChanged": state["valueBefore"] != self.parameter_hash(include_prefix="value_head."),
            "updateCount": self.update_count,
        }
        if equivalence_diagnostics:
            result["diagnostics"] = {
                "gradientHash": gradient_hash,
                "parameterHash": self.parameter_hash(),
                "optimizerHash": self.optimizer_hash(),
                "rngHash": self.rng_hash(),
            }
        self._accumulation = None
        return result

    def gradient_hash(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            digest.update(name.encode())
            if parameter.grad is None:
                digest.update(b"<none>")
            else:
                digest.update(parameter.grad.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def rng_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(torch.get_rng_state().cpu().contiguous().numpy().tobytes())
        if torch.cuda.is_available():
            for index, state in enumerate(torch.cuda.get_rng_state_all()):
                digest.update(f"cuda:{index}".encode())
                digest.update(state.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def diagnostics(self) -> dict[str, str]:
        return {
            "parameterHash": self.parameter_hash(),
            "optimizerHash": self.optimizer_hash(),
            "rngHash": self.rng_hash(),
            "gradientHash": self.gradient_hash(),
        }

    def parameter_hash(self, include_prefix: str | None = None, exclude_prefix: str | None = None) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            if include_prefix is not None and not name.startswith(include_prefix):
                continue
            if exclude_prefix is not None and name.startswith(exclude_prefix):
                continue
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def optimizer_hash(self) -> str:
        """Read-only, deterministic hash used by resume regression tests."""
        digest = hashlib.sha256()
        state = self.optimizer.state_dict()
        for group_index, group in enumerate(state["param_groups"]):
            digest.update(f"group:{group_index}".encode())
            for key in sorted(key for key in group if key != "params"):
                digest.update(key.encode())
                digest.update(repr(group[key]).encode())
            digest.update(repr(tuple(group["params"])).encode())
        for parameter_id in sorted(state["state"]):
            digest.update(f"parameter:{parameter_id}".encode())
            for key, value in sorted(state["state"][parameter_id].items()):
                digest.update(key.encode())
                if torch.is_tensor(value):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
        return digest.hexdigest()

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".ppo-checkpoint-", suffix=".tmp", dir=directory)
        os.close(descriptor)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        self._atomic_save({
            "schemaVersion": PPO_CHECKPOINT_SCHEMA_VERSION,
            "checkpointKind": "ppo_self_play",
            "featureSpec": self.feature_spec,
            "modelStateDict": self.model.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "updateCount": self.update_count,
            "episodeCount": self.episode_count,
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "torchRngState": torch.get_rng_state(),
            "cudaRngStateAll": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metadata": metadata or {},
        }, path)

    def resume(self, path: str) -> dict[str, int]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != PPO_CHECKPOINT_SCHEMA_VERSION or checkpoint.get("checkpointKind") != "ppo_self_play":
            raise ValueError("Unsupported PPO checkpoint")
        if checkpoint.get("featureSpec") != self.feature_spec:
            raise ValueError("PPO resume Feature Spec mismatch")
        if checkpoint.get("hyperparameters") != self.hyperparameters or checkpoint.get("seed") != self.seed:
            raise ValueError("PPO resume configuration mismatch")
        for field in ("modelStateDict", "optimizerStateDict", "torchRngState"):
            if field not in checkpoint:
                raise ValueError(f"PPO checkpoint is missing {field}")
        self.model.load_state_dict(checkpoint["modelStateDict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizerStateDict"])
        self._act_state_branch_cache.clear()
        self.update_count = int(checkpoint.get("updateCount", -1))
        self.episode_count = int(checkpoint.get("episodeCount", -1))
        if self.update_count < 0 or self.episode_count < 0:
            raise ValueError("PPO checkpoint counters are invalid")
        torch.set_rng_state(checkpoint["torchRngState"].cpu())
        cuda_states = checkpoint.get("cudaRngStateAll")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        return {"updateCount": self.update_count, "episodeCount": self.episode_count}
