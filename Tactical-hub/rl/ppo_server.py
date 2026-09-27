from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

from rl.bc_packed import decode_packed_views, prepare_packed_tensors
from rl.device import report_torch_device, resolve_torch_device
from rl.ppo_trainer import PpoTrainer


def send(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    trainer = None
    stream = sys.stdin.buffer
    profile = os.environ.get("PPO_PROFILE") == "1"
    timings = {}
    legal_action_counts = []
    retained_chunks = {}
    consumed_retention_ids = set()
    retention_totals = {
        "storedChunks": 0,
        "storedSamples": 0,
        "rawBytes": 0,
        "compressedBytes": 0,
        "currentRetainedBytes": 0,
        "peakRetainedBytes": 0,
    }

    def record(stage, elapsed):
        if not profile:
            return
        item = timings.setdefault(stage, {"count": 0, "totalMs": 0.0})
        item["count"] += 1
        item["totalMs"] += elapsed * 1000.0

    def sync_device():
        if profile and trainer is not None and trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

    def read_binary(byte_length):
        if byte_length < 0:
            raise ValueError("Packed PPO byteLength must be non-negative")
        payload = bytearray()
        while len(payload) < byte_length:
            chunk = stream.read(byte_length - len(payload))
            if not chunk:
                raise EOFError("Packed PPO payload ended early")
            payload.extend(chunk)
        return payload

    def retention_stats():
        return {
            "currentChunks": len(retained_chunks),
            **retention_totals,
        }

    while True:
        line = stream.readline()
        if not line:
            return
        try:
            message = json.loads(line)
            kind = message.get("type")
            if kind == "init":
                device = resolve_torch_device(message.get("device", "auto"))
                report_torch_device(message.get("device", "auto"), device)
                trainer = PpoTrainer(message["featureSpec"], message["hyperparameters"], int(message["seed"]), device)
                if message.get("resume"):
                    state = trainer.resume(message["resume"])
                else:
                    trainer.load_initial_model(message["initialCheckpoint"])
                    state = {"updateCount": 0, "episodeCount": 0}
                send({"type": "ready", "selectedDevice": device.type, **state})
            elif trainer is None:
                raise RuntimeError("PPO server is not initialized")
            elif kind == "retainPackedChunk":
                retention_id = str(message.get("retentionId", ""))
                if not retention_id:
                    raise ValueError("PPO retentionId must not be empty")
                compressed = read_binary(int(message["byteLength"]))
                if retention_id in retained_chunks or retention_id in consumed_retention_ids:
                    raise ValueError(f"Duplicate PPO retentionId: {retention_id}")
                if message.get("encoding") != "packed-deflate-raw-v1" or message.get("codec") != "deflate-raw-1":
                    raise ValueError("Unsupported PPO retention encoding")
                try:
                    raw = bytearray(zlib.decompress(bytes(compressed), wbits=-zlib.MAX_WBITS))
                except zlib.error as error:
                    raise ValueError(f"PPO retained payload decompression failed: {error}") from error
                raw_byte_length = int(message["rawByteLength"])
                if len(raw) != raw_byte_length:
                    raise ValueError(f"PPO retained raw length mismatch: {len(raw)} != {raw_byte_length}")
                raw_sha256 = hashlib.sha256(raw).hexdigest()
                if raw_sha256 != message.get("rawSha256"):
                    raise ValueError("PPO retained payload SHA256 mismatch")
                views = decode_packed_views(message, raw)
                batch_size = int(message["batchSize"])
                if batch_size <= 0 or int(views["targets"].shape[0]) != batch_size:
                    raise ValueError("PPO retained batch size mismatch")
                retained_chunks[retention_id] = {
                    "header": {
                        "tensors": message["tensors"],
                        "batchSize": batch_size,
                    },
                    "compressed": bytes(compressed),
                    "rawByteLength": raw_byte_length,
                    "rawSha256": raw_sha256,
                    "batchSize": batch_size,
                }
                retention_totals["storedChunks"] += 1
                retention_totals["storedSamples"] += batch_size
                retention_totals["rawBytes"] += raw_byte_length
                retention_totals["compressedBytes"] += len(compressed)
                retention_totals["currentRetainedBytes"] += len(compressed)
                retention_totals["peakRetainedBytes"] = max(
                    retention_totals["peakRetainedBytes"],
                    retention_totals["currentRetainedBytes"],
                )
                send({
                    "type": "retainedChunkStored",
                    "requestId": message["requestId"],
                    "retentionId": retention_id,
                    "batchSize": batch_size,
                    "rawBytes": raw_byte_length,
                    "compressedBytes": len(compressed),
                    "rawSha256": raw_sha256,
                })
            elif kind == "retainedUpdateChunk":
                retention_id = str(message.get("retentionId", ""))
                scalar_payload = read_binary(int(message["byteLength"]))
                if retention_id in consumed_retention_ids:
                    raise ValueError(f"PPO retentionId already consumed: {retention_id}")
                record = retained_chunks.get(retention_id)
                if record is None:
                    raise ValueError(f"Unknown PPO retentionId: {retention_id}")
                if message.get("encoding") != "ppo-retained-scalars-v1":
                    raise ValueError("Unsupported PPO retained scalar encoding")
                batch_size = int(message["batchSize"])
                if batch_size != record["batchSize"]:
                    raise ValueError("PPO retained scalar batch size mismatch")
                expected_scalar_bytes = batch_size * 3 * 4
                if len(scalar_payload) != expected_scalar_bytes:
                    raise ValueError("PPO retained scalar byte length mismatch")
                raw = bytearray(zlib.decompress(record["compressed"], wbits=-zlib.MAX_WBITS))
                if len(raw) != record["rawByteLength"] or hashlib.sha256(raw).hexdigest() != record["rawSha256"]:
                    raise ValueError("PPO retained payload integrity mismatch")
                views = decode_packed_views(record["header"], raw)
                prepared, actions, action_mask, targets = prepare_packed_tensors(views, trainer.device)
                scalar_values = np.frombuffer(scalar_payload, dtype=np.dtype("<f4"))
                old = torch.from_numpy(scalar_values[:batch_size]).to(device=trainer.device, dtype=torch.float32)
                advantages = torch.from_numpy(scalar_values[batch_size:batch_size * 2]).to(device=trainer.device, dtype=torch.float32)
                returns = torch.from_numpy(scalar_values[batch_size * 2:]).to(device=trainer.device, dtype=torch.float32)
                result = trainer.accumulate_prepared_chunk(
                    prepared, actions, action_mask, targets, old, advantages, returns,
                )
                retained_chunks.pop(retention_id)
                consumed_retention_ids.add(retention_id)
                retention_totals["currentRetainedBytes"] -= len(record["compressed"])
                send({"type": "updateChunkAccepted", "requestId": message["requestId"], **result})
            elif kind == "discardRetained":
                retention_ids = [str(value) for value in message.get("retentionIds", [])]
                discarded_count = 0
                for retention_id in retention_ids:
                    record = retained_chunks.pop(retention_id, None)
                    if record is not None:
                        retention_totals["currentRetainedBytes"] -= len(record["compressed"])
                        discarded_count += 1
                send({"type": "retentionDiscarded", "requestId": message["requestId"], "discardedCount": discarded_count})
            elif kind == "retentionStats":
                send({"type": "retentionStats", "requestId": message["requestId"], **retention_stats()})
            elif kind == "diagnostics":
                send({"type": "diagnostics", "requestId": message["requestId"], **trainer.diagnostics()})
            elif kind in ("packedAct", "packedActBatch", "packedUpdateChunk"):
                if profile:
                    prepare_start = time.perf_counter()
                payload = read_binary(int(message["byteLength"]))
                views = decode_packed_views(message, payload)
                prepared, actions, action_mask, targets = prepare_packed_tensors(views, trainer.device)
                if profile:
                    sync_device()
                    record("packed_read_decode_prepare", time.perf_counter() - prepare_start)
                if kind == "packedAct":
                    if profile:
                        # Packed action requests have a single sample, with no action padding.
                        legal_action_counts.append(int(action_mask.shape[1]))
                        sync_device()
                        inference_start = time.perf_counter()
                    action = trainer.act_prepared(
                        prepared, actions, action_mask,
                        profile_stage=record if profile else None,
                    )
                    if profile:
                        sync_device()
                        record("act_inference", time.perf_counter() - inference_start)
                    send({"type": "action", "requestId": message["requestId"], **action})
                elif kind == "packedActBatch":
                    actions_result = trainer.act_prepared_batch(prepared, actions, action_mask)
                    send({"type": "actions", "requestId": message["requestId"], **actions_result})
                else:
                    if profile:
                        sync_device()
                        accumulate_start = time.perf_counter()
                    floating = lambda name: torch.from_numpy(views[name]).to(device=trainer.device, dtype=torch.float32)
                    result = trainer.accumulate_prepared_chunk(
                        prepared, actions, action_mask, targets,
                        floating("oldLogProbabilities"), floating("advantages"), floating("returns"),
                    )
                    if profile:
                        sync_device()
                        record("update_accumulate_chunk", time.perf_counter() - accumulate_start)
                    send({"type": "updateChunkAccepted", "requestId": message["requestId"], **result})
            elif kind == "act":
                send({"type": "action", "requestId": message["requestId"], **trainer.act(message["observation"], message["actions"])})
            elif kind == "beginUpdate":
                result = trainer.begin_accumulated_update(int(message["totalSamples"]))
                send({"type": "updateBegun", "requestId": message["requestId"], **result})
            elif kind == "finishUpdate":
                if profile:
                    sync_device()
                    update_start = time.perf_counter()
                update_result = trainer.finish_accumulated_update()
                if profile:
                    sync_device()
                    record("update_finish", time.perf_counter() - update_start)
                trainer.episode_count += int(message.get("completedEpisodes", 0))
                send({"type": "updateResult", "requestId": message["requestId"], **update_result, "episodeCount": trainer.episode_count})
            elif kind == "update":
                update_result = trainer.update(message["samples"])
                trainer.episode_count += int(message.get("completedEpisodes", 0))
                send({"type": "updateResult", "requestId": message["requestId"], **update_result, "episodeCount": trainer.episode_count})
            elif kind == "save":
                trainer.save(message["path"], message.get("metadata"))
                send({"type": "saved", "requestId": message["requestId"], "path": message["path"], "updateCount": trainer.update_count, "episodeCount": trainer.episode_count})
            elif kind == "close":
                if profile:
                    summary = {name: {"count": item["count"], "totalMs": round(item["totalMs"], 2), "avgMs": round(item["totalMs"] / item["count"], 3)} for name, item in timings.items()}
                    legal_summary = {
                        "count": len(legal_action_counts),
                        "min": min(legal_action_counts) if legal_action_counts else 0,
                        "max": max(legal_action_counts) if legal_action_counts else 0,
                        "avg": round(sum(legal_action_counts) / len(legal_action_counts), 2) if legal_action_counts else 0,
                    }
                    sys.stderr.write("[PPO profile python] " + json.dumps(
                        {"stages": summary, "legalActions": legal_summary}, separators=(",", ":")
                    ) + "\n")
                    sys.stderr.flush()
                send({"type": "closed"})
                return
            else:
                raise ValueError(f"Unknown PPO message type: {kind}")
        except Exception as error:
            send({"type": "error", "requestId": locals().get("message", {}).get("requestId"), "message": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
