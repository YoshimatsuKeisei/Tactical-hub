from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from rl.migrate_bc_v1_to_v2 import (
        ACTION_INPUT_WEIGHT,
        UNIT_INPUT_WEIGHT,
        load_manifest,
        migrate_checkpoint,
        migrate_model_state,
    )
    from rl.policy_model import STRATEGIC_TABLES, TacticalPolicyValueNetwork
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed; install requirements-rl.txt")
class BcV1ToV2MigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = Path(__file__).with_name("bc_v1_to_v2_manifest.json")
        cls.manifest = load_manifest(cls.manifest_path)

    def observation(self, spec: dict, unit_row: list[float]) -> dict:
        strategic = {"global": [0.0] * spec["strategicGlobalWidth"]}
        strategic.update({name: [] for name in STRATEGIC_TABLES})
        return {
            "global": [0.0] * spec["globalWidth"],
            "teams": [], "teamMask": [],
            "units": [unit_row], "unitMask": [1],
            "map": [],
            "bases": [], "baseMask": [],
            "constructions": [], "constructionMask": [],
            "strategicState": strategic,
        }

    @staticmethod
    def expand_row(source: list[float], target_width: int, segments: list[dict[str, int]]) -> list[float]:
        target = [0.0] * target_width
        for segment in segments:
            source_start = segment["sourceStart"]
            target_start = segment["targetStart"]
            width = segment["width"]
            target[target_start:target_start + width] = source[source_start:source_start + width]
        return target

    def test_migrates_common_parameters_and_zero_initializes_only_new_input_columns(self):
        torch.manual_seed(41)
        source = TacticalPolicyValueNetwork(self.manifest["sourceFeatureSpec"])
        target = TacticalPolicyValueNetwork(self.manifest["targetFeatureSpec"])
        migrated, report = migrate_model_state(source.state_dict(), target, self.manifest)
        self.assertEqual(set(report["partial"]), {UNIT_INPUT_WEIGHT, ACTION_INPUT_WEIGHT})
        for name, source_tensor in source.state_dict().items():
            target_tensor = migrated[name]
            if name not in report["partial"]:
                self.assertTrue(torch.equal(source_tensor, target_tensor), name)
        for name, segments in (
            (UNIT_INPUT_WEIGHT, self.manifest["unitInputColumns"]),
            (ACTION_INPUT_WEIGHT, self.manifest["actionInputColumns"]),
        ):
            source_tensor = source.state_dict()[name]
            target_tensor = migrated[name]
            copied = set()
            for segment in segments:
                source_slice = source_tensor[:, segment["sourceStart"]:segment["sourceStart"] + segment["width"]]
                target_slice = target_tensor[:, segment["targetStart"]:segment["targetStart"] + segment["width"]]
                self.assertTrue(torch.equal(source_slice, target_slice), name)
                copied.update(range(segment["targetStart"], segment["targetStart"] + segment["width"]))
            new_columns = sorted(set(range(target_tensor.shape[1])) - copied)
            self.assertTrue(torch.equal(target_tensor[:, new_columns], torch.zeros_like(target_tensor[:, new_columns])), name)

    def test_old_rule_inputs_produce_equivalent_logits_and_value(self):
        torch.manual_seed(73)
        source = TacticalPolicyValueNetwork(self.manifest["sourceFeatureSpec"])
        target = TacticalPolicyValueNetwork(self.manifest["targetFeatureSpec"])
        migrated, _ = migrate_model_state(source.state_dict(), target, self.manifest)
        target.load_state_dict(migrated, strict=True)
        source_unit = torch.randn(self.manifest["sourceFeatureSpec"]["unitWidth"]).tolist()
        source_actions = torch.randn(3, self.manifest["sourceFeatureSpec"]["actionFeatureWidth"]).tolist()
        target_unit = self.expand_row(source_unit, self.manifest["targetFeatureSpec"]["unitWidth"], self.manifest["unitInputColumns"])
        target_actions = [self.expand_row(row, self.manifest["targetFeatureSpec"]["actionFeatureWidth"], self.manifest["actionInputColumns"]) for row in source_actions]
        source.eval()
        target.eval()
        with torch.no_grad():
            source_logits, source_value, _, _ = source(self.observation(self.manifest["sourceFeatureSpec"], source_unit), source_actions)
            target_logits, target_value, _, _ = target(self.observation(self.manifest["targetFeatureSpec"], target_unit), target_actions)
        self.assertTrue(torch.allclose(source_logits, target_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(source_value, target_value, atol=1e-6, rtol=1e-6))

    def test_checkpoint_metadata_source_integrity_and_cli_guards(self):
        source_spec = self.manifest["sourceFeatureSpec"]
        torch.manual_seed(97)
        source_model = TacticalPolicyValueNetwork(source_spec)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "v1.pt"
            output_path = root / "v2.pt"
            torch.save({
                "schemaVersion": 1,
                "featureSpec": source_spec,
                "modelStateDict": source_model.state_dict(),
                "metadata": {"fixture": True},
            }, source_path)
            source_before = source_path.read_bytes()
            migrate_checkpoint(source_path, output_path, self.manifest_path)
            self.assertEqual(source_path.read_bytes(), source_before)
            migrated = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertEqual(migrated["schemaVersion"], 2)
            self.assertEqual(migrated["featureSpec"], self.manifest["targetFeatureSpec"])
            self.assertNotIn("optimizerStateDict", migrated)
            self.assertEqual(migrated["metadata"]["sourceSchemaVersion"], 1)
            self.assertEqual(migrated["metadata"]["targetSchemaVersion"], 2)
            self.assertEqual(migrated["metadata"]["migrationType"], "v1_to_v2_zero_init_new_features")
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))
            with self.assertRaisesRegex(ValueError, "must differ"):
                migrate_checkpoint(source_path, source_path, self.manifest_path)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                migrate_checkpoint(source_path, output_path, self.manifest_path)

    def test_rejects_non_v1_and_feature_spec_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output.pt"
            wrong_schema = root / "wrong-schema.pt"
            torch.save({"schemaVersion": 2}, wrong_schema)
            with self.assertRaisesRegex(ValueError, "schemaVersion 1"):
                migrate_checkpoint(wrong_schema, output, self.manifest_path)
            mismatch = root / "mismatch.pt"
            wrong_spec = copy.deepcopy(self.manifest["sourceFeatureSpec"])
            wrong_spec["unitWidth"] += 1
            torch.save({"schemaVersion": 1, "featureSpec": wrong_spec, "modelStateDict": {}}, mismatch)
            with self.assertRaisesRegex(ValueError, "Feature Spec"):
                migrate_checkpoint(mismatch, output, self.manifest_path)


if __name__ == "__main__":
    unittest.main()
