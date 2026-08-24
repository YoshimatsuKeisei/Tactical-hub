from __future__ import annotations

import json
import sys

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
            elif kind in ("packedAct", "packedUpdateChunk"):
                byte_length = int(message["byteLength"])
                payload = bytearray()
                while len(payload) < byte_length:
                    chunk = stream.read(byte_length - len(payload))
                    if not chunk:
                        raise EOFError("Packed PPO payload ended early")
                    payload.extend(chunk)
                views = decode_packed_views(message, payload)
                prepared, actions, action_mask, targets = prepare_packed_tensors(views, trainer.device)
                if kind == "packedAct":
                    send({"type": "action", "requestId": message["requestId"], **trainer.act_prepared(prepared, actions, action_mask)})
                else:
                    floating = lambda name: torch.from_numpy(views[name]).to(device=trainer.device, dtype=torch.float32)
                    result = trainer.accumulate_prepared_chunk(
                        prepared, actions, action_mask, targets,
                        floating("oldLogProbabilities"), floating("advantages"), floating("returns"),
                    )
                    send({"type": "updateChunkAccepted", "requestId": message["requestId"], **result})
            elif kind == "act":
                send({"type": "action", "requestId": message["requestId"], **trainer.act(message["observation"], message["actions"])})
            elif kind == "beginUpdate":
                result = trainer.begin_accumulated_update(int(message["totalSamples"]))
                send({"type": "updateBegun", "requestId": message["requestId"], **result})
            elif kind == "finishUpdate":
                update_result = trainer.finish_accumulated_update()
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
                send({"type": "closed"})
                return
            else:
                raise ValueError(f"Unknown PPO message type: {kind}")
        except Exception as error:
            send({"type": "error", "requestId": locals().get("message", {}).get("requestId"), "message": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
