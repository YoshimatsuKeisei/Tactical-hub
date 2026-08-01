from __future__ import annotations

import json
import sys
import time
from typing import Any

try:
    import torch
    from rl.bc_packed import decode_packed_views, prepare_packed_tensors
    from rl.bc_trainer import BehavioralCloningTrainer
    from rl.device import report_torch_device, resolve_torch_device
except ModuleNotFoundError as error:
    print(f"RL Python dependency is missing: {error}", file=sys.stderr)
    raise SystemExit(3)


def send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    trainer: BehavioralCloningTrainer | None = None
    stream = sys.stdin.buffer
    while True:
        line = stream.readline()
        if not line:
            return
        try:
            deserialize_started = time.perf_counter()
            message = json.loads(line)
            deserialize_ms = (time.perf_counter() - deserialize_started) * 1000
            message_type = message.get("type")
            if message_type == "init":
                requested_device = message.get("device", "auto")
                selected_device = resolve_torch_device(requested_device)
                report_torch_device(requested_device, selected_device)
                torch_threads = message.get("torchThreads")
                torch_interop_threads = message.get("torchInteropThreads")
                if torch_threads is not None:
                    if not isinstance(torch_threads, int) or isinstance(torch_threads, bool) or torch_threads <= 0:
                        raise ValueError("torchThreads must be a positive integer")
                    torch.set_num_threads(torch_threads)
                if torch_interop_threads is not None:
                    if not isinstance(torch_interop_threads, int) or isinstance(torch_interop_threads, bool) or torch_interop_threads <= 0:
                        raise ValueError("torchInteropThreads must be a positive integer")
                    torch.set_num_interop_threads(torch_interop_threads)
                trainer = BehavioralCloningTrainer(
                    message["featureSpec"],
                    float(message["learningRate"]),
                    int(message["seed"]),
                    selected_device,
                )
                send({
                    "type": "ready",
                    "torchThreads": torch.get_num_threads(),
                    "torchInteropThreads": torch.get_num_interop_threads(),
                    "selectedDevice": selected_device.type,
                })
            elif message_type == "batch":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                metrics = trainer.process_batch(message["samples"], bool(message["train"]))
                send({"type": "batchResult", "requestId": message["requestId"], **metrics})
            elif message_type in ("packedBatch", "packedProfileBatch"):
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                byte_length = int(message["byteLength"])
                payload = bytearray()
                while len(payload) < byte_length:
                    chunk = stream.read(byte_length - len(payload))
                    if not chunk:
                        raise EOFError("Packed BC payload ended early")
                    payload.extend(chunk)
                if message_type == "packedBatch":
                    views = decode_packed_views(message, payload)
                    prepared, actions, action_mask, targets = prepare_packed_tensors(views, trainer.device)
                    metrics = trainer.process_packed_batch(
                        prepared, actions, action_mask, targets, bool(message["train"])
                    )
                    send({"type": "batchResult", "requestId": message["requestId"], **metrics})
                else:
                    decode_started = time.perf_counter()
                    views = decode_packed_views(message, payload)
                    binary_decode_ms = (time.perf_counter() - decode_started) * 1000
                    if trainer.device.type == "cuda":
                        torch.cuda.synchronize(trainer.device)
                    preparation_started = time.perf_counter()
                    prepared, actions, action_mask, targets = prepare_packed_tensors(views, trainer.device)
                    if trainer.device.type == "cuda":
                        torch.cuda.synchronize(trainer.device)
                    preparation_ms = (time.perf_counter() - preparation_started) * 1000
                    metrics = trainer.process_profile_packed_batch(
                        prepared, actions, action_mask, targets, preparation_ms
                    )
                    send({
                        "type": "profileBatchResult",
                        "requestId": message["requestId"],
                        "deserializeMs": deserialize_ms,
                        "binaryDecodeMs": binary_decode_ms,
                        **metrics,
                    })
            elif message_type == "profileBatch":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                metrics = trainer.process_profile_batch(message["samples"])
                send({
                    "type": "profileBatchResult",
                    "requestId": message["requestId"],
                    "deserializeMs": deserialize_ms,
                    "binaryDecodeMs": 0.0,
                    **metrics,
                })
            elif message_type == "save":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                trainer.save(message["path"], message.get("metadata", {}))
                send({"type": "saved", "requestId": message["requestId"]})
            elif message_type == "saveTrainingCheckpoint":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                trainer.save_training_checkpoint(
                    message["path"],
                    message["state"],
                )
                send({"type": "trainingCheckpointSaved", "requestId": message["requestId"]})
            elif message_type == "load":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                trainer.load(message["path"])
                send({"type": "loaded", "requestId": message["requestId"]})
            elif message_type == "resumeTrainingCheckpoint":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                state = trainer.resume_training_checkpoint(
                    message["path"], message["expected"]
                )
                send({"type": "trainingCheckpointResumed", "requestId": message["requestId"], "state": state})
            elif message_type == "parameterHash":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                send({"type": "parameterHash", "requestId": message["requestId"], "hash": trainer.parameter_hash()})
            elif message_type == "optimizerHash":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                send({"type": "optimizerHash", "requestId": message["requestId"], "hash": trainer.optimizer_hash()})
            elif message_type == "close":
                send({"type": "closed"})
                return
            else:
                raise ValueError(f"Unknown message type: {message_type}")
        except Exception as error:
            send({"type": "error", "message": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
