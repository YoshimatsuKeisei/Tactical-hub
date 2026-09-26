"""Read-only PPO batch-prepare profiler for real encoded observations/actions.

Profiles the current Python-list -> padded tensor preparation path at a fixed batch
size. It does not update the model or checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from rl.policy_model import (
    STRATEGIC_TABLES,
    TacticalPolicyValueNetwork,
    padded_masks,
    padded_rows,
)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_cpu(operation):
    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000


def timed_device(device: torch.device, operation):
    sync(device)
    started = time.perf_counter()
    result = operation()
    sync(device)
    return result, (time.perf_counter() - started) * 1000


def add(timings: dict[str, float], name: str, elapsed_ms: float) -> None:
    timings[name] += elapsed_ms


def prepare_profiled(
    model: TacticalPolicyValueNetwork,
    samples: list[dict],
    timings: dict[str, float],
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    device = model.device
    observations, elapsed = timed_cpu(lambda: [sample["observation"] for sample in samples])
    add(timings, "python.collect_observations", elapsed)

    global_values, elapsed = timed_cpu(lambda: [observation["global"] for observation in observations])
    add(timings, "python.collect_global", elapsed)
    global_tensor, elapsed = timed_device(
        device, lambda: torch.tensor(global_values, dtype=torch.float32, device=device)
    )
    add(timings, "tensor.global", elapsed)

    strategic_global_values, elapsed = timed_cpu(
        lambda: [observation["strategicState"]["global"] for observation in observations]
    )
    add(timings, "python.collect_strategic_global", elapsed)
    strategic_global_tensor, elapsed = timed_device(
        device,
        lambda: torch.tensor(strategic_global_values, dtype=torch.float32, device=device),
    )
    add(timings, "tensor.strategic_global", elapsed)

    prepared = {
        "global": global_tensor,
        "strategicGlobal": strategic_global_tensor,
        "masked": {},
        "strategic": {},
    }

    for key, mask_key, width in (
        ("teams", "teamMask", model.feature_spec["teamWidth"]),
        ("units", "unitMask", model.feature_spec["unitWidth"]),
        ("bases", "baseMask", model.feature_spec["baseWidth"]),
        ("constructions", "constructionMask", model.feature_spec["constructionWidth"]),
    ):
        values, elapsed = timed_cpu(lambda k=key: [observation[k] for observation in observations])
        add(timings, f"python.collect_{key}", elapsed)
        table_and_presence, elapsed = timed_device(
            device, lambda v=values, w=width: padded_rows(v, w, device)
        )
        add(timings, f"tensor.padded_rows_{key}", elapsed)
        table, presence = table_and_presence

        masks, elapsed = timed_cpu(
            lambda mk=mask_key: [observation[mk] for observation in observations]
        )
        add(timings, f"python.collect_{mask_key}", elapsed)
        explicit, elapsed = timed_device(
            device, lambda m=masks, rows=table.shape[1]: padded_masks(m, rows, device)
        )
        add(timings, f"tensor.padded_masks_{key}", elapsed)
        combined, elapsed = timed_device(device, lambda: presence & explicit)
        add(timings, f"tensor.mask_and_{key}", elapsed)
        prepared["masked"][key] = (table, combined)

    map_rows, elapsed = timed_cpu(
        lambda: [[tile for row in observation["map"] for tile in row] for observation in observations]
    )
    add(timings, "python.flatten_map", elapsed)
    prepared_map, elapsed = timed_device(
        device,
        lambda: padded_rows(map_rows, model.feature_spec["mapTileWidth"], device),
    )
    add(timings, "tensor.padded_rows_map", elapsed)
    prepared["map"] = prepared_map

    for name in STRATEGIC_TABLES:
        values, elapsed = timed_cpu(
            lambda n=name: [observation["strategicState"][n] for observation in observations]
        )
        add(timings, f"python.collect_strategic.{name}", elapsed)
        strategic_table, elapsed = timed_device(
            device,
            lambda v=values, n=name: padded_rows(
                v, model.feature_spec["strategicTableRowWidths"][n], device
            ),
        )
        add(timings, f"tensor.padded_rows_strategic.{name}", elapsed)
        prepared["strategic"][name] = strategic_table

    actions, elapsed = timed_cpu(lambda: [sample["actions"] for sample in samples])
    add(timings, "python.collect_actions", elapsed)
    action_pair, elapsed = timed_device(
        device,
        lambda: padded_rows(actions, model.feature_spec["actionFeatureWidth"], device),
    )
    add(timings, "tensor.padded_rows_actions", elapsed)
    action_rows, action_mask = action_pair
    return prepared, action_rows, action_mask


def tensors_close(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.equal(left, right) if left.dtype == torch.bool else torch.allclose(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensors_close(left[k], right[k]) for k in left)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(tensors_close(a, b) for a, b in zip(left, right))
    raise TypeError(type(left))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-pairs", type=int, default=5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refuse a misleading GPU prepare profile")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    device = torch.device(args.device)
    timings: dict[str, float] = defaultdict(float)
    processed = 0
    profiled = 0
    saw_end = False

    with args.fixture.open(encoding="utf-8") as fixture:
        header = json.loads(next(fixture))
        if header.get("type") != "header" or header.get("schemaVersion") != 2:
            raise RuntimeError("Not a schema-v2 PPO probe fixture")
        decisions = int(header["decisions"])
        environment_count = int(header.get("environmentCount", len(header.get("seeds", []))))
        if args.batch_size > environment_count:
            raise RuntimeError("Fixture has fewer environments than requested batch size")
        if not 0 <= args.warmup_pairs < decisions:
            raise ValueError("--warmup-pairs must be smaller than fixture decisions")

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("featureSpec") != header["featureSpec"]:
            raise RuntimeError("Checkpoint and fixture Feature Spec do not match")
        model = TacticalPolicyValueNetwork(header["featureSpec"]).to(device)
        model.load_state_dict(checkpoint["modelStateDict"], strict=True)
        model.eval()

        with torch.inference_mode():
            for record in fixture:
                pair = json.loads(record)
                if pair.get("type") == "end":
                    saw_end = True
                    break
                if pair.get("type") != "pair" or pair.get("step") != processed:
                    raise RuntimeError("Probe fixture sequence is invalid")
                samples = pair["samples"][:args.batch_size]

                if processed == 0:
                    baseline_observations = [sample["observation"] for sample in samples]
                    baseline_actions = [sample["actions"] for sample in samples]
                    baseline_prepared = model.prepare_observation_batch(baseline_observations)
                    baseline_action_rows, baseline_action_mask = padded_rows(
                        baseline_actions, model.feature_spec["actionFeatureWidth"], device
                    )
                    check_timings: dict[str, float] = defaultdict(float)
                    check_prepared, check_action_rows, check_action_mask = prepare_profiled(
                        model, samples, check_timings
                    )
                    if not tensors_close(baseline_prepared, check_prepared):
                        raise RuntimeError("Profiled observation preparation differs from production path")
                    if not torch.allclose(baseline_action_rows, check_action_rows):
                        raise RuntimeError("Profiled action rows differ from production path")
                    if not torch.equal(baseline_action_mask, check_action_mask):
                        raise RuntimeError("Profiled action mask differs from production path")

                if processed >= args.warmup_pairs:
                    prepare_profiled(model, samples, timings)
                    profiled += 1
                processed += 1

    if processed != decisions or not saw_end or profiled != decisions - args.warmup_pairs:
        raise RuntimeError("Fixture did not complete as expected")

    total_ms = sum(timings.values())
    python_ms = sum(value for name, value in timings.items() if name.startswith("python."))
    tensor_ms = sum(value for name, value in timings.items() if name.startswith("tensor."))
    action_ms = timings["python.collect_actions"] + timings["tensor.padded_rows_actions"]
    observation_ms = total_ms - action_ms
    rows = [
        {
            "stage": name,
            "totalMs": round(value, 3),
            "avgMsPerBatch": round(value / profiled, 4),
            "sharePct": round((value / total_ms) * 100, 2) if total_ms else 0.0,
        }
        for name, value in sorted(timings.items(), key=lambda item: item[1], reverse=True)
    ]

    print(json.dumps({
        "probe": "ppo_prepare_breakdown",
        "status": "passed",
        "device": str(device),
        "batchSize": args.batch_size,
        "profiledPairs": profiled,
        "totalMs": round(total_ms, 3),
        "avgMsPerBatch": round(total_ms / profiled, 4),
        "avgMsPerEnvironment": round(total_ms / (profiled * args.batch_size), 4),
        "pythonMs": round(python_ms, 3),
        "tensorMs": round(tensor_ms, 3),
        "observationPrepareMs": round(observation_ms, 3),
        "actionPrepareMs": round(action_ms, 3),
        "actionSharePct": round((action_ms / total_ms) * 100, 2) if total_ms else 0.0,
        "rows": rows,
        "productionEquivalentVerified": True,
        "modelUpdate": False,
        "checkpointUpdateCount": checkpoint.get("updateCount"),
        "checkpointEpisodeCount": checkpoint.get("episodeCount"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
