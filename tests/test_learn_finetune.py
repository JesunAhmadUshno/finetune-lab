"""Unit tests for Level 1, Learn: LoRA and GRPO teaching code.

Hand-computed expectations:
- LoRA trainable parameters at rank 2, in 16, classes 4:
  A is 16 x 2 = 32, B is 2 x 4 = 8, total 40.
- Full fine-tune parameters for the same layer: 16 x 4 = 64.
- GRPO on rewards [1, 0, 1, 0]: mean 0.5, advantages
  [0.5, -0.5, 0.5, -0.5], population std 0.5, normalized
  advantages [1, -1, 1, -1].
- GRPO on rewards [2, 1, 0, 1]: mean 1.0, advantages
  [1, 0, -1, 0], population std sqrt(0.5) = 0.7071 rounded,
  normalized [1.4142, 0, -1.4142, 0] rounded.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))

import grpo_learn
import lora_learn


class TestParameterCounts(unittest.TestCase):
    def test_lora_parameter_count(self):
        # 16*2 + 2*4 = 32 + 8 = 40
        self.assertEqual(lora_learn.lora_parameter_count(), 40)

    def test_full_parameter_count(self):
        # 16*4 = 64
        self.assertEqual(lora_learn.full_parameter_count(), 64)

    def test_lora_count_formula_other_rank(self):
        # rank 4: 16*4 + 4*4 = 64 + 16 = 80
        self.assertEqual(lora_learn.lora_parameter_count(rank=4), 80)

    def test_adapter_shapes(self):
        a, b = lora_learn.init_adapter()
        self.assertEqual(a.shape, (16, 2))
        self.assertEqual(b.shape, (2, 4))
        delta = a @ b
        self.assertEqual(delta.shape, (16, 4))


class TestLoRA(unittest.TestCase):
    def test_b_starts_at_zero_so_start_is_base(self):
        _, _, _, _, w_base = lora_learn.make_task()
        a, b = lora_learn.init_adapter()
        self.assertTrue(np.all(b == 0.0))
        eff = lora_learn.effective_weights(w_base, a, b)
        self.assertTrue(np.array_equal(eff, w_base))

    def test_base_weights_unchanged_by_training(self):
        x_train, y_train, _, _, w_base = lora_learn.make_task()
        before = w_base.copy()
        hash_before = lora_learn.weights_hash(w_base)
        a, b = lora_learn.init_adapter()
        lora_learn.train(x_train, y_train, w_base, a, b, epochs=5)
        self.assertTrue(np.array_equal(w_base, before))
        self.assertEqual(lora_learn.weights_hash(w_base), hash_before)

    def test_training_improves_test_accuracy(self):
        result = lora_learn.run_lora()
        before = result["before_training"]["test_accuracy"]
        after = result["after_training"]["test_accuracy"]
        self.assertGreater(after, before)
        self.assertTrue(result["lora"]["base_weights_unchanged"])

    def test_determinism_of_run(self):
        first = lora_learn.run_lora()
        second = lora_learn.run_lora()
        self.assertEqual(first, second)

    def test_task_shapes_and_labels(self):
        x_train, y_train, x_test, y_test, w_base = lora_learn.make_task()
        self.assertEqual(x_train.shape, (512, 16))
        self.assertEqual(x_test.shape, (128, 16))
        self.assertEqual(w_base.shape, (16, 4))
        self.assertTrue(set(np.unique(y_train)).issubset({0, 1, 2, 3}))
        self.assertTrue(set(np.unique(y_test)).issubset({0, 1, 2, 3}))


class TestGRPO(unittest.TestCase):
    def test_binary_group_by_hand(self):
        out = grpo_learn.group_advantages([1.0, 0.0, 1.0, 0.0])
        self.assertEqual(out["baseline_mean"], 0.5)
        self.assertEqual(out["advantages"], [0.5, -0.5, 0.5, -0.5])
        self.assertEqual(out["std_population"], 0.5)
        self.assertEqual(out["advantages_normalized"], [1.0, -1.0, 1.0, -1.0])

    def test_graded_group_by_hand(self):
        out = grpo_learn.group_advantages([2.0, 1.0, 0.0, 1.0])
        self.assertEqual(out["baseline_mean"], 1.0)
        self.assertEqual(out["advantages"], [1.0, 0.0, -1.0, 0.0])
        self.assertEqual(out["std_population"], 0.7071)
        self.assertEqual(out["advantages_normalized"], [1.4142, 0.0, -1.4142, 0.0])

    def test_all_equal_group_has_zero_advantages(self):
        out = grpo_learn.group_advantages([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(out["advantages"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(out["advantages_normalized"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(out["std_population"], 0.0)

    def test_advantages_sum_to_zero(self):
        out = grpo_learn.group_advantages([3.0, 1.0, 4.0, 0.0, 2.0])
        self.assertAlmostEqual(sum(out["advantages"]), 0.0, places=3)

    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):
            grpo_learn.group_advantages([])


if __name__ == "__main__":
    unittest.main()
