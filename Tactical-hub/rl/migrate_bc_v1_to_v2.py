from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from rl.policy_model import TacticalPolicyValueNetwork


MIGRATION_TYPE = "v1_to_v2_zero_init_new_features"
UNIT_INPUT_WEIGHT = "unit_encoder.0.weight"
ACTION_INPUT_WEIGHT = "action_encoder.0.weight"


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1:
        raise ValueError("Unsupported migration manifest schemaVersion")
    source = manifest.get("sourceFeatureSpec")
    target = manifest.get("targetFeatureSpec")
    if not isinstance(source, dict) or source.get("schemaVersion") != 1:
        raise ValueError("Migration manifest source Feature Spec must use schemaVersion 1")
    if not isinstance(target, dict) or target.get("schemaVersion") != 2:
        raise ValueError("Migration manifest target Feature Spec must use schemaVersion 2")
    return manifest


def _copy_columns(
    source: torch.Tensor,
    target_template: torch.Tensor,
    segments: list[dict[str, int]],
) -> torch.Tensor:
    if source.ndim != 2 or target_template.ndim != 2 or source.shape[0] != target_template.shape[0]:
        raise ValueError("Expanded input weights have incompatible output dimensions")
    target = torch.zeros_like(target_template)
    copied_source: set[int] = set()
    copied_target: set[int] = set()
    for segment in segments:
        source_start = segment["sourceStart"]
        target_start = segment["targetStart"]
        width = segment["width"]
        if min(source_start, target_start, width) < 0:
            raise ValueError("Migration column segment is invalid")
        if source_start + width > source.shape[1] or target_start + width > target.shape[1]:
            raise ValueError("Migration column segment exceeds input width")
        source_columns = set(range(source_start, source_start + width))
        target_columns = set(range(target_start, target_start + width))
        if copied_source & source_columns or copied_target & target_columns:
            raise ValueError("Migration column segments overlap")
        copied_source.update(source_columns)
        copied_target.update(target_columns)
        target[:, target_start:target_start + width].copy_(source[:, source_start:source_start + width])
    if copied_source != set(range(source.shape[1])):
        raise ValueError("Migration does not copy every v1 input column")
    return target


def migrate_model_state(
    source_state: dict[str, torch.Tensor],
    target_model: TacticalPolicyValueNetwork,
    manifest: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    target_state = target_model.state_dict()
    if set(source_state) != set(target_state):
        missing = sorted(set(target_state) - set(source_state))
        extra = sorted(set(source_state) - set(target_state))
        raise ValueError(f"Model parameter keys differ: missing={missing}, extra={extra}")
    migrated: dict[str, torch.Tensor] = {}
    complete: list[str] = []
    partial: list[str] = []
    for name, target_tensor in target_state.items():
        source_tensor = source_state[name]
        if source_tensor.shape == target_tensor.shape:
            migrated[name] = source_tensor.detach().clone()
            complete.append(name)
        elif name == UNIT_INPUT_WEIGHT:
            migrated[name] = _copy_columns(source_tensor, target_tensor, manifest["unitInputColumns"])
            partial.append(name)
        elif name == ACTION_INPUT_WEIGHT:
            migrated[name] = _copy_columns(source_tensor, target_tensor, manifest["actionInputColumns"])
            partial.append(name)
        else:
            raise ValueError(
                f"Unsupported parameter shape change for {name}: {tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
    target_model.load_state_dict(migrated, strict=True)
    return migrated, {"complete": complete, "partial": partial}


def _atomic_save_new(checkpoint: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as error:
            raise FileExistsError(f"Output checkpoint already exists: {output_path}") from error
        temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)


def migrate_checkpoint(input_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output checkpoint paths must differ")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output checkpoint already exists: {output_path}")
    manifest = load_manifest(manifest_path)
    source_bytes_before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(source_checkpoint, dict) or source_checkpoint.get("schemaVersion") != 1:
        raise ValueError("Input checkpoint must be a schemaVersion 1 best-model checkpoint")
    if source_checkpoint.get("featureSpec") != manifest["sourceFeatureSpec"]:
        raise ValueError("Input checkpoint Feature Spec does not match the manifest v1 Feature Spec")
    source_state = source_checkpoint.get("modelStateDict")
    if not isinstance(source_state, dict):
        raise ValueError("Input checkpoint is missing modelStateDict")
    source_model = TacticalPolicyValueNetwork(manifest["sourceFeatureSpec"])
    source_model.load_state_dict(source_state, strict=True)
    target_model = TacticalPolicyValueNetwork(manifest["targetFeatureSpec"])
    migrated_state, report = migrate_model_state(source_model.state_dict(), target_model, manifest)
    metadata = dict(source_checkpoint.get("metadata") or {})
    metadata.update({
        "sourceCheckpoint": str(input_path),
        "sourceSchemaVersion": 1,
        "targetSchemaVersion": 2,
        "migrationType": MIGRATION_TYPE,
    })
    _atomic_save_new({
        "schemaVersion": 2,
        "featureSpec": manifest["targetFeatureSpec"],
        "modelStateDict": migrated_state,
        "metadata": metadata,
    }, output_path)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_bytes_before:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Source checkpoint changed during migration")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "migrationType": MIGRATION_TYPE,
        "completeParameterCount": len(report["complete"]),
        "partialParameters": report["partial"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a schema v1 BC best checkpoint to a schema v2 initialization checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("bc_v1_to_v2_manifest.json")))
    args = parser.parse_args()
    try:
        result = migrate_checkpoint(Path(args.input), Path(args.output), Path(args.manifest))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
