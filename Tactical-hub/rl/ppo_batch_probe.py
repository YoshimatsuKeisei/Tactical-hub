"""Standalone read-only GPU PPO probe: same two real-environment inputs, serial vs batch.
Only times model input preparation and Forward, not Node/Python IPC, PPO update, or full matches.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rl.policy_model import TacticalPolicyValueNetwork, padded_rows


def measured(device: torch.device, operation):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - started) * 1000


def prepare(model: TacticalPolicyValueNetwork, samples: list[dict]):
    observations = [sample["observation"] for sample in samples]
    actions = [sample["actions"] for sample in samples]
    if any(not action_rows for action_rows in actions):
        raise ValueError("Probe requires at least one legal action per environment")
    prepared = model.prepare_observation_batch(observations)
    padded_actions, mask = padded_rows(
        actions, model.feature_spec["actionFeatureWidth"], model.device
    )
    return prepared, padded_actions, mask


def forward(model: TacticalPolicyValueNetwork, sample_batch):
    observation, actions, mask = sample_batch
    logits, values, _, _, returned_mask = model.forward_prepared_batch(
        observation, actions, mask
    )
    if not torch.equal(mask, returned_mask):
        raise RuntimeError("Forward changed legal action mask")
    return logits, values, returned_mask


def compare(serial, batched, samples, atol: float):
    batch_logits, batch_values, batch_mask = batched
    for index, ((logits, values, mask), sample) in enumerate(zip(serial, samples)):
        count = sample["legalCount"]
        if count != len(sample["actions"]) or count < 1:
            raise RuntimeError("Fixture legal action count mismatch")
        if not bool(batch_mask[index, :count].all()) or bool(batch_mask[index, count:].any()):
            raise RuntimeError("Batch mask included invalid padded actions")
        single_logits = logits[0, :count]
        group_logits = batch_logits[index, :count]
        if not bool(torch.isfinite(group_logits).all()) or not bool(torch.isfinite(batch_values[index])):
            raise RuntimeError("Batch inference produced non-finite output")
        if not torch.allclose(single_logits, group_logits, atol=atol, rtol=1e-4):
            raise RuntimeError(f"Logits differ for environment {index}")
        if not torch.allclose(values[0], batch_values[index], atol=atol, rtol=1e-4):
            raise RuntimeError(f"State value differs for environment {index}")
        choice = int(torch.argmax(group_logits).item())
        if not 0 <= choice < count:
            raise RuntimeError("Batch model selected illegal action")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup-pairs", type=int, default=5)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refuse a misleading GPU speed comparison")
    if not 0 <= args.warmup_pairs < 100:
        raise ValueError("--warmup-pairs must be in 0..99")
    device = torch.device(args.device)
    with args.fixture.open(encoding="utf-8") as fixture:
        header = json.loads(next(fixture))
        if header.get("type") != "header" or header.get("schemaVersion") != 2:
            raise RuntimeError("Not a schema-v2 PPO probe fixture")
        if header.get("decisions") != 100 or len(header.get("seeds", [])) != 2:
            raise RuntimeError("Expected exactly two environments, each with 100 decisions")
        # Inspect once on CPU to avoid loading optimizer/RNG or changing source checkpoint.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (checkpoint.get("checkpointKind") != "ppo_self_play"
                or checkpoint.get("featureSpec") != header["featureSpec"]):
            raise RuntimeError("Checkpoint and fixture Feature Spec do not match")
        model = TacticalPolicyValueNetwork(header["featureSpec"]).to(device)
        model.load_state_dict(checkpoint["modelStateDict"], strict=True)
        model.eval()
        timings = {"serial_prepare": 0., "serial_forward": 0.,
                   "batch_prepare": 0., "batch_forward": 0.}
        max_legal = 0
        processed = 0
        saw_end = False
        with torch.inference_mode():
            for record in fixture:
                pair = json.loads(record)
                if pair.get("type") == "end":
                    if pair.get("decisionCounts") != [100, 100]:
                        raise RuntimeError("Fixture ended early")
                    saw_end = True
                    break
                if pair.get("type") != "pair" or pair.get("step") != processed:
                    raise RuntimeError("Probe fixture sequence is invalid")
                samples = pair["samples"]
                if len(samples) != 2:
                    raise RuntimeError("Probe needs exactly two independent inputs")
                max_legal = max(max_legal, *(s["legalCount"] for s in samples))
                def run_serial():
                    serial_outputs = []
                    prep_total = forward_total = 0.
                    for sample in samples:
                        tensors, prep_ms = measured(device, lambda s=sample: prepare(model, [s]))
                        output, forward_ms = measured(device, lambda t=tensors: forward(model, t))
                        serial_outputs.append(output)
                        prep_total += prep_ms
                        forward_total += forward_ms
                    return serial_outputs, prep_total, forward_total

                def run_batch():
                    batched_input, prep_ms = measured(device, lambda: prepare(model, samples))
                    output, forward_ms = measured(device, lambda: forward(model, batched_input))
                    return output, prep_ms, forward_ms

                # Alternate call order to limit warm-cache/order bias.
                if processed % 2:
                    batched_output, batch_prep_ms, batch_forward_ms = run_batch()
                    serial, serial_prep_ms, serial_forward_ms = run_serial()
                else:
                    serial, serial_prep_ms, serial_forward_ms = run_serial()
                    batched_output, batch_prep_ms, batch_forward_ms = run_batch()
                compare(serial, batched_output, samples, atol=3e-4)
                if processed >= args.warmup_pairs:
                    for key, elapsed in (
                        ("serial_prepare", serial_prep_ms), ("serial_forward", serial_forward_ms),
                        ("batch_prepare", batch_prep_ms), ("batch_forward", batch_forward_ms),
                    ):
                        timings[key] += elapsed
                processed += 1
        if processed != 100 or not saw_end:
            raise RuntimeError(f"Expected 100 input pairs and end marker, got {processed}")
    benchmark_pairs = processed - args.warmup_pairs
    if benchmark_pairs <= 0:
        raise RuntimeError("Warmup must be smaller than total probe pairs")
    serial_ms = timings["serial_prepare"] + timings["serial_forward"]
    batch_ms = timings["batch_prepare"] + timings["batch_forward"]
    print(json.dumps({
        "probe": "two_environment_forward_only", "status": "passed",
        "seeds": header["seeds"], "decisionsPerEnvironment": processed,
        "warmupPairsExcluded": args.warmup_pairs, "timedPairs": benchmark_pairs,
        "modelUpdate": False, "checkpointUpdateCount": checkpoint["updateCount"],
        "checkpointEpisodeCount": checkpoint["episodeCount"],
        "device": str(device), "maxLegalActions": max_legal,
        "logitsAndValuesEquivalent": True, "legalActionsVerified": True,
        "serialTotalMs": round(serial_ms, 3), "batchTotalMs": round(batch_ms, 3),
        "serialForwardMs": round(timings["serial_forward"], 3),
        "batchForwardMs": round(timings["batch_forward"], 3),
        "serialPrepareMs": round(timings["serial_prepare"], 3),
        "batchPrepareMs": round(timings["batch_prepare"], 3),
        "modelOnlyThroughputRatio": round(serial_ms / batch_ms, 3) if batch_ms > 0 else None,
        "limits": "Model preparation+Forward only; excludes Node/Python IPC, sampling, replay, training and full-match throughput",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
