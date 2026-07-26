from __future__ import annotations

import json
import sys
from typing import Any

try:
    import torch
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
    for line in sys.stdin:
        try:
            message = json.loads(line)
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
            elif message_type == "save":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                trainer.save(message["path"], message.get("metadata", {}))
                send({"type": "saved", "requestId": message["requestId"]})
            elif message_type == "load":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                trainer.load(message["path"])
                send({"type": "loaded", "requestId": message["requestId"]})
            elif message_type == "parameterHash":
                if trainer is None:
                    raise RuntimeError("Trainer is not initialized")
                send({"type": "parameterHash", "requestId": message["requestId"], "hash": trainer.parameter_hash()})
            elif message_type == "close":
                send({"type": "closed"})
                return
            else:
                raise ValueError(f"Unknown message type: {message_type}")
        except Exception as error:
            send({"type": "error", "message": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
