from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    import torch
    from rl.policy_model import TacticalPolicyValueNetwork
except ModuleNotFoundError as error:
    raise SystemExit(f"RL Python dependency is missing; install requirements-rl.txt. Details: {error}")


class BrowserBcModel:
    def __init__(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != 1:
            raise ValueError("BC browser inference requires a schemaVersion 1 best-model checkpoint")
        if "featureSpec" not in checkpoint or "modelStateDict" not in checkpoint:
            raise ValueError("BC best-model checkpoint is missing featureSpec or modelStateDict")
        self.feature_spec = checkpoint["featureSpec"]
        self.metadata = checkpoint.get("metadata", {})
        self.model = TacticalPolicyValueNetwork(self.feature_spec).to(torch.device("cpu"))
        self.model.load_state_dict(checkpoint["modelStateDict"])
        self.model.eval()

    def infer(self, request: dict[str, Any]) -> str:
        if request.get("featureSpec") != self.feature_spec:
            raise ValueError("BC inference featureSpec does not match bc-best.pt")
        action_keys = request.get("actionKeys")
        actions = request.get("actions")
        if not isinstance(action_keys, list) or not action_keys or not isinstance(actions, list) or len(actions) != len(action_keys):
            raise ValueError("BC inference requires matching non-empty actions and actionKeys")
        action_index, _value = self.model.act(request["observation"], actions)
        if action_index < 0 or action_index >= len(action_keys):
            raise ValueError(f"Model returned invalid action index {action_index}")
        return str(action_keys[action_index])


def create_handler(model: BrowserBcModel):
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: int) -> None:
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-headers", "content-type")
            self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
            self.end_headers()

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            self._headers(status)
            self.wfile.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

        def do_OPTIONS(self) -> None:
            self._headers(204)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"ready": True, "checkpoint": "bc-best.pt"})

        def do_POST(self) -> None:
            if self.path != "/infer":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length <= 0 or length > 64 * 1024 * 1024:
                    raise ValueError("Invalid BC inference request size")
                request = json.loads(self.rfile.read(length))
                self._send(200, {"decisionKey": request.get("decisionKey"), "actionKey": model.infer(request)})
            except Exception as error:
                self._send(400, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser BC inference server")
    parser.add_argument("--checkpoint", default="rl-checkpoints/bc-best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"BC checkpoint not found: {checkpoint}")
    torch.manual_seed(args.seed)
    model = BrowserBcModel(str(checkpoint))
    server = ThreadingHTTPServer((args.host, args.port), create_handler(model))
    print(f"BC inference ready http://{args.host}:{args.port} checkpoint={checkpoint}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
