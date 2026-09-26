from __future__ import annotations

import json
import os
import sys
import time

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

    def record(stage, elapsed):
        if not profile:
            return
        item = timings.setdefault(stage, {"count": 0, "totalMs": 0.0})
        item["count"] += 1
        item["totalMs"] += elapsed * 1000.0

    def sync_device():
        if profile and trainer is not None and trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)

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
            elif kind in ("packedAct", "packedActBatch", "packedUpdateChunk"):
                if profile:
                    prepare_start = time.perf_counter()
                byte_length = int(message["byteLength"])
                payload = bytearray()
                while len(payload) < byte_length:
                    chunk = stream.read(byte_length - len(payload))
                    if not chunk:
                        raise EOFError("Packed PPO payload ended early")
                    payload.extend(chunk)
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
