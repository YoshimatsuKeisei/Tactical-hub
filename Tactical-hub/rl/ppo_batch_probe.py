"""Standalone read-only GPU PPO probe: serial inference vs multi-environment batches.

Benchmarks the same real-environment inputs at batch sizes such as 2/4/8/16.
Measures model input preparation and Forward only; it does not update the model or
checkpoint and does not include Node/Python IPC or full-match scheduling.
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


def parse_batch_sizes(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 2 for value in values) or values != sorted(set(values)):
        raise ValueError("--batch-sizes must be unique ascending integers >= 2")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-sizes", default="2,4,8,16")
    parser.add_argument("--warmup-pairs", type=int, default=5)
    args = parser.parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refuse a misleading GPU speed comparison")

    device = torch.device(args.device)
    with args.fixture.open(encoding="utf-8") as fixture:
        header = json.loads(next(fixture))
        if header.get("type") != "header" or header.get("schemaVersion") != 2:
            raise RuntimeError("Not a schema-v2 PPO probe fixture")
        decisions = int(header.get("decisions", 0))
        seeds = header.get("seeds", [])
        environment_count = int(header.get("environmentCount", len(seeds)))
        if decisions <= 0 or len(seeds) != environment_count:
            raise RuntimeError("Invalid multi-environment fixture header")
        if max(batch_sizes) > environment_count:
            raise RuntimeError("Fixture has fewer environments than requested max batch size")
        if not 0 <= args.warmup_pairs < decisions:
            raise ValueError("--warmup-pairs must be smaller than fixture decisions")

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (checkpoint.get("checkpointKind") != "ppo_self_play"
                or checkpoint.get("featureSpec") != header["featureSpec"]):
            raise RuntimeError("Checkpoint and fixture Feature Spec do not match")
        model = TacticalPolicyValueNetwork(header["featureSpec"]).to(device)
        model.load_state_dict(checkpoint["modelStateDict"], strict=True)
        model.eval()

        timings = {
            size: {
                "serial_prepare": 0.0, "serial_forward": 0.0,
                "batch_prepare": 0.0, "batch_forward": 0.0,
            }
            for size in batch_sizes
        }
        max_legal = 0
        processed = 0
        saw_end = False

        with torch.inference_mode():
            for record in fixture:
                pair = json.loads(record)
                if pair.get("type") == "end":
                    if pair.get("decisionCounts") != [decisions] * environment_count:
                        raise RuntimeError("Fixture ended early")
                    saw_end = True
                    break
                if pair.get("type") != "pair" or pair.get("step") != processed:
                    raise RuntimeError("Probe fixture sequence is invalid")
                samples = pair["samples"]
                if len(samples) != environment_count:
                    raise RuntimeError("Fixture environment count changed within the stream")
                max_legal = max(max_legal, *(sample["legalCount"] for sample in samples))

                serial_outputs = []
                serial_prepare_ms = []
                serial_forward_ms = []
                for sample in samples[:max(batch_sizes)]:
                    tensors, prep_ms = measured(device, lambda s=sample: prepare(model, [s]))
                    output, forward_ms = measured(device, lambda t=tensors: forward(model, t))
                    serial_outputs.append(output)
                    serial_prepare_ms.append(prep_ms)
                    serial_forward_ms.append(forward_ms)

                order = batch_sizes if processed % 2 == 0 else list(reversed(batch_sizes))
                for size in order:
                    subset = samples[:size]
                    batched_input, batch_prep_ms = measured(device, lambda s=subset: prepare(model, s))
                    batched_output, batch_forward_ms = measured(
                        device, lambda t=batched_input: forward(model, t)
                    )
                    compare(serial_outputs[:size], batched_output, subset, atol=3e-4)
                    if processed >= args.warmup_pairs:
                        timings[size]["serial_prepare"] += sum(serial_prepare_ms[:size])
                        timings[size]["serial_forward"] += sum(serial_forward_ms[:size])
                        timings[size]["batch_prepare"] += batch_prep_ms
                        timings[size]["batch_forward"] += batch_forward_ms
                processed += 1

        if processed != decisions or not saw_end:
            raise RuntimeError(f"Expected {decisions} input pairs and end marker, got {processed}")

    timed_pairs = processed - args.warmup_pairs
    rows = []
    for size in batch_sizes:
        item = timings[size]
        serial_total = item["serial_prepare"] + item["serial_forward"]
        batch_total = item["batch_prepare"] + item["batch_forward"]
        rows.append({
            "batchSize": size,
            "timedPairs": timed_pairs,
            "timedSamples": timed_pairs * size,
            "serialTotalMs": round(serial_total, 3),
            "batchTotalMs": round(batch_total, 3),
            "serialPrepareMs": round(item["serial_prepare"], 3),
            "batchPrepareMs": round(item["batch_prepare"], 3),
            "serialForwardMs": round(item["serial_forward"], 3),
            "batchForwardMs": round(item["batch_forward"], 3),
            "preparePlusForwardThroughputRatio": round(serial_total / batch_total, 3)
                if batch_total > 0 else None,
            "forwardOnlyThroughputRatio": round(item["serial_forward"] / item["batch_forward"], 3)
                if item["batch_forward"] > 0 else None,
            "batchMsPerEnvironment": round(batch_total / (timed_pairs * size), 4)
                if timed_pairs > 0 else None,
        })

    print(json.dumps({
        "probe": "multi_environment_batch_scaling",
        "status": "passed",
        "seeds": seeds,
        "fixtureEnvironmentCount": environment_count,
        "decisionsPerEnvironment": processed,
        "warmupPairsExcluded": args.warmup_pairs,
        "device": str(device),
        "maxLegalActions": max_legal,
        "modelUpdate": False,
        "checkpointUpdateCount": checkpoint["updateCount"],
        "checkpointEpisodeCount": checkpoint["episodeCount"],
        "logitsAndValuesEquivalent": True,
        "legalActionsVerified": True,
        "rows": rows,
        "limits": (
            "Model preparation+Forward only; excludes Node/Python IPC, sampling, "
            "full-match scheduling, replay and PPO training"
        ),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
