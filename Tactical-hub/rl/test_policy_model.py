import unittest

try:
    import torch
    from rl.policy_model import TacticalPolicyValueNetwork, masked_mean_pool
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed; install requirements-rl.txt")
class PolicyModelTest(unittest.TestCase):
    def feature_spec(self):
        table_widths = {name: 3 for name in (
            "siegeStates", "kingCampaignStates", "rewardPlacementRequests",
            "strategistCooldowns", "teleportCooldowns", "productionIntents",
            "movementIntents", "attackIntents", "strategistActionIntents",
            "teleportIntents",
        )}
        return {
            "globalWidth": 4, "teamWidth": 3, "unitWidth": 5,
            "mapTileWidth": 4, "baseWidth": 4, "constructionWidth": 3,
            "strategicGlobalWidth": 2, "strategicTableRowWidths": table_widths,
            "actionFeatureWidth": 6,
        }

    def observation(self):
        strategic = {"global": [0, 1]}
        strategic.update({name: [] for name in self.feature_spec()["strategicTableRowWidths"]})
        return {
            "global": [1, 2, 3, 4],
            "teams": [[1, 0, 0], [0, 1, 0]], "teamMask": [1, 1],
            "units": [[1, 2, 3, 4, 5], [0, 0, 0, 0, 0]], "unitMask": [1, 0],
            "map": [[[0, 0, 1, 0], [1, 0, 0, 1]]],
            "bases": [[1, 0, 0, 1]], "baseMask": [1],
            "constructions": [[0, 0, 0]], "constructionMask": [0],
            "strategicState": strategic,
        }

    def test_shapes_finite_empty_tables_and_reproducibility(self):
        spec = self.feature_spec()
        actions = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        torch.manual_seed(19)
        first = TacticalPolicyValueNetwork(spec)
        logits, value, state, encoded_actions = first(self.observation(), actions)
        self.assertEqual(tuple(state.shape), (256,))
        self.assertEqual(tuple(encoded_actions.shape), (2, 128))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(value.shape), ())
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(value))
        first_index, _ = first.act(self.observation(), actions)
        self.assertIn(first_index, range(len(actions)))

        torch.manual_seed(19)
        second = TacticalPolicyValueNetwork(spec)
        second_index, _ = second.act(self.observation(), actions)
        torch.manual_seed(19)
        replay = TacticalPolicyValueNetwork(spec)
        replay_index, _ = replay.act(self.observation(), actions)
        self.assertEqual(second_index, replay_index)

    def test_masked_pooling_excludes_padding_and_handles_zero_valid_rows(self):
        encoded = torch.tensor([[2.0, 4.0], [100.0, 100.0]])
        self.assertTrue(torch.equal(masked_mean_pool(encoded, [1, 0]), torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(masked_mean_pool(encoded, [0, 0]), torch.tensor([0.0, 0.0])))

    def test_batched_forward_matches_single_sample_and_masks_padded_actions(self):
        spec = self.feature_spec()
        model = TacticalPolicyValueNetwork(spec)
        first_observation = self.observation()
        second_observation = self.observation()
        second_observation["units"] = second_observation["units"][:1]
        second_observation["unitMask"] = [1]
        first_actions = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        second_actions = [[0, 0, 1, 0, 0, 0]]

        first_logits, first_value, _, _ = model(first_observation, first_actions)
        second_logits, second_value, _, _ = model(second_observation, second_actions)
        logits, values, states, actions, action_mask = model.forward_batch(
            [first_observation, second_observation],
            [first_actions, second_actions],
        )

        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(tuple(states.shape), (2, 256))
        self.assertEqual(tuple(actions.shape), (2, 2, 128))
        self.assertTrue(torch.allclose(logits[0], first_logits, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(logits[1, :1], second_logits, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(values, torch.stack((first_value, second_value)), atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(action_mask, torch.tensor([[True, True], [True, False]])))
        self.assertTrue(torch.isneginf(logits[1, 1]))
        self.assertEqual(int(torch.argmax(logits[1]).item()), 0)


if __name__ == "__main__":
    unittest.main()
