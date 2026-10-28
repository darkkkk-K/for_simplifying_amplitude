"""Synthetic checks of legal actions, policy observations and top-k probabilities."""
import math
import unittest
import torch
from project.rl.early_fusion_policy import sample_one
from project.rl.restricted_policy import build_observation, CONTEXT_FEATURES


class InferenceTests(unittest.TestCase):
    def test_expression_observation_and_legal_pairs(self):
        tokens = [
            ['+', '0', '1', '<12>', '^', '0', '1', '<34>', '^', '0', '1', '_', '4'],
            ['+', '0', '1', '<13>', '^', '0', '1', '<24>', '^', '0', '1', '_', '4']]
        observation = build_observation(tokens, 4, 2)
        self.assertEqual(tuple(observation.symbolic.shape), (2, 32))
        self.assertEqual(tuple(observation.actions.shape), (2, 3))
        self.assertEqual(len(CONTEXT_FEATURES), 5)
        self.assertEqual(observation.best_terms, 2)

    def test_top4_uses_conditional_softmax(self):
        logits = torch.tensor([[4., 3., 2., 1., 0.]])
        mask = torch.ones_like(logits, dtype=torch.bool)
        selected, logp, _ = sample_one(logits, mask, 4, torch.Generator().manual_seed(103))
        self.assertIn(selected, (0, 1, 2, 3))
        expected = float(logits[0, selected])-math.log(sum(math.exp(float(v)) for v in logits[0, :4]))
        self.assertAlmostEqual(float(logp), expected, places=7)


if __name__ == '__main__':
    unittest.main()
