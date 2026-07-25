from __future__ import annotations

import json
import sys
from typing import Any

try:
    import torch
    from rl.policy_model import TacticalPolicyValueNetwork
except ModuleNotFoundError as error:
    print(
        "RL Python dependency is missing. Install requirements-rl.txt "
        f"with the Python interpreter used by the runner. Details: {error}",
        file=sys.stderr,
    )
    raise SystemExit(3)


def send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    model: TacticalPolicyValueNetwork | None = None
    for raw_line in sys.stdin:
        try:
            message = json.loads(raw_line)
            message_type = message.get("type")
            if message_type == "init":
                torch.manual_seed(int(message["seed"]))
                model = TacticalPolicyValueNetwork(message["featureSpec"])
                model.eval()
                send({"type": "ready"})
            elif message_type == "act":
                if model is None:
                    raise RuntimeError("Policy server has not been initialized")
                action_index, value = model.act(message["observation"], message["actions"])
                send({"type": "action", "requestId": message["requestId"], "actionIndex": action_index, "value": value})
            elif message_type == "close":
                send({"type": "closed"})
                return
            else:
                raise ValueError(f"Unknown message type: {message_type}")
        except Exception as error:  # protocol errors must remain JSON on stdout
            send({"type": "error", "message": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
