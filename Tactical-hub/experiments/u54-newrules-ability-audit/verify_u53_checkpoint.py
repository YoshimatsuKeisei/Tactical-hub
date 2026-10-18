from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from rl.ppo_trainer import PpoTrainer

EXPECTED_SHA256 = "cce23ec427ee350071244690840e787455baa1651b4d60ce779a7951ae42222a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: verify_u53_checkpoint.py CHECKPOINT FEATURE_SPEC_JSON")
    checkpoint_path = Path(sys.argv[1])
    expected_spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    before = sha256_file(checkpoint_path)
    if before != EXPECTED_SHA256:
        raise RuntimeError(f"u53 SHA256 mismatch: {before}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("featureSpec") != expected_spec:
        raise RuntimeError("u53 Feature Spec mismatch")
    if checkpoint.get("schemaVersion") != 1 or checkpoint.get("checkpointKind") != "ppo_self_play":
        raise RuntimeError("u53 checkpoint schema/kind mismatch")
    if checkpoint.get("updateCount") != 53 or checkpoint.get("episodeCount") != 410:
        raise RuntimeError("u53 checkpoint counters mismatch")
    if checkpoint.get("seed") != 7:
        raise RuntimeError("u53 trainer seed mismatch")
    trainer = PpoTrainer(expected_spec, checkpoint["hyperparameters"], 7, "cpu")
    resumed = trainer.resume(str(checkpoint_path))
    after = sha256_file(checkpoint_path)
    if after != before:
        raise RuntimeError("u53 checkpoint changed during read-only verification")
    print(json.dumps({
        "loaded": True,
        "sha256Before": before,
        "sha256After": after,
        "schemaVersion": checkpoint["schemaVersion"],
        "checkpointKind": checkpoint["checkpointKind"],
        "updateCount": trainer.update_count,
        "episodeCount": trainer.episode_count,
        "resumeState": resumed,
        "modelTensorCount": len(trainer.model.state_dict()),
        "optimizerStateEntries": len(trainer.optimizer.state_dict()["state"]),
        "featureSpec": expected_spec,
        "hyperparameters": checkpoint["hyperparameters"],
    }, indent=2))


if __name__ == "__main__":
    main()
