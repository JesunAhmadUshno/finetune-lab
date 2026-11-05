"""Unit tests for Level 2, Build: pipeline and GRPO policy training.

Hand-computed expectations:
- Clipping: clip(1.5, eps 0.2) = 1.2; clip(0.5) = 0.8; clip(1.1) = 1.1.
- Surrogate: ratio 1.5, advantage 2.0: unclipped 3.0, clipped
  1.2 * 2.0 = 2.4, min = 2.4. Ratio 1.5, advantage -2.0: unclipped
  -3.0, clipped -2.4, min = -3.0 (the minimum favours the unclipped
  branch for negative advantages, which is the PPO rule).
- KL: identical distributions give KL 0. A distribution [1, 0]
  against [0.5, 0.5] gives KL = log(2) = 0.6931 (with the 1e-12
  guard the zero term contributes 0).
- Eval arithmetic: accuracy is correct / n; the confusion matrix
  rows sum to the per-class counts.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))

import grpo_build
import pipeline


class TestCheckpoint(unittest.TestCase):
    def test_round_trip_restores_adapter_and_scores(self):
        dataset = pipeline.make_dataset()
        w_base = dataset["w_base"]
        a, b, _ = pipeline.train_sft(dataset, epochs=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.npz")
            pipeline.save_checkpoint(path, a, b, w_base)
            a2, b2, sidecar = pipeline.load_checkpoint(path, w_base=w_base)
            self.assertTrue(np.array_equal(a, a2))
            self.assertTrue(np.array_equal(b, b2))
            self.assertEqual(sidecar["trainable_parameters"], 40)
            ev1 = pipeline.evaluate(dataset["x_test"], dataset["y_test"], w_base, a, b)
            ev2 = pipeline.evaluate(dataset["x_test"], dataset["y_test"], w_base, a2, b2)
            self.assertEqual(ev1, ev2)

    def test_round_trip_rejects_wrong_base(self):
        dataset = pipeline.make_dataset()
        a, b = pipeline.init_adapter()
        wrong_base = dataset["w_base"] + 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.npz")
            pipeline.save_checkpoint(path, a, b, dataset["w_base"])
            with self.assertRaises(ValueError):
                pipeline.load_checkpoint(path, w_base=wrong_base)


class TestEvalHarness(unittest.TestCase):
    def test_accuracy_and_confusion_arithmetic(self):
        dataset = pipeline.make_dataset()
        a, b = pipeline.init_adapter()
        ev = pipeline.evaluate(dataset["x_test"], dataset["y_test"],
                               dataset["w_base"], a, b)
        self.assertEqual(ev["n"], 128)
        self.assertAlmostEqual(ev["accuracy"], ev["correct"] / 128, places=4)
        row_sums = [sum(row) for row in ev["confusion"]]
        expected = [int((dataset["y_test"] == c).sum()) for c in range(4)]
        self.assertEqual(row_sums, expected)
        self.assertEqual(sum(row_sums), 128)
        self.assertEqual(len(ev["per_class_accuracy"]), 4)

    def test_data_card_counts(self):
        card = pipeline.data_card(pipeline.make_dataset())
        self.assertEqual(card["train"]["n"], 512)
        self.assertEqual(card["val"]["n"], 128)
        self.assertEqual(card["test"]["n"], 128)
        self.assertEqual(sum(card["train"]["class_counts"]), 512)

    def test_batches_cover_all_indices_once(self):
        batches = pipeline.batch_indices(512, 64, epoch=0)
        flat = np.concatenate(batches)
        self.assertEqual(sorted(flat.tolist()), list(range(512)))
        self.assertEqual(len(batches), 8)

    def test_training_improves_held_out_accuracy(self):
        result = pipeline.run_pipeline()
        before = result["before_training"]["test"]["accuracy"]
        after = result["after_training"]["test"]["accuracy"]
        self.assertGreater(after, before)
        self.assertTrue(result["config"]["base_weights_unchanged"])


class TestGRPOClippingAndKL(unittest.TestCase):
    def test_clipped_ratio_by_hand(self):
        self.assertEqual(float(grpo_build.clipped_ratio(np.array(1.5))), 1.2)
        self.assertEqual(float(grpo_build.clipped_ratio(np.array(0.5))), 0.8)
        self.assertEqual(float(grpo_build.clipped_ratio(np.array(1.1))), 1.1)

    def test_surrogate_positive_advantage_uses_clip(self):
        out = grpo_build.surrogate_per_sample(np.array([1.5]), np.array([2.0]))
        self.assertAlmostEqual(float(out[0]), 2.4, places=6)

    def test_surrogate_negative_advantage_uses_min_rule(self):
        out = grpo_build.surrogate_per_sample(np.array([1.5]), np.array([-2.0]))
        self.assertAlmostEqual(float(out[0]), -3.0, places=6)

    def test_kl_identical_is_zero(self):
        p = np.array([[0.25, 0.25, 0.25, 0.25]])
        self.assertAlmostEqual(float(grpo_build.kl_divergence(p, p)[0]), 0.0, places=6)

    def test_kl_by_hand(self):
        p = np.array([[1.0, 0.0]])
        ref = np.array([[0.5, 0.5]])
        self.assertAlmostEqual(float(grpo_build.kl_divergence(p, ref)[0]),
                               0.6931, places=3)

    def test_checker_is_verifiable(self):
        for prompt in range(grpo_build.N_PROMPTS):
            self.assertEqual(grpo_build.checker(prompt, grpo_build.CORRECT[prompt]), 1.0)
            wrong = (int(grpo_build.CORRECT[prompt]) + 1) % 4
            self.assertEqual(grpo_build.checker(prompt, wrong), 0.0)

    def test_grpo_training_improves_reward(self):
        result = grpo_build.run_grpo(iterations=30)
        self.assertGreater(result["after_training"]["mean_reward"],
                           result["before_training"]["mean_reward"])
        self.assertGreaterEqual(result["final_mean_kl_to_ref"], 0.0)

    def test_grpo_determinism(self):
        first = grpo_build.run_grpo(iterations=10)
        second = grpo_build.run_grpo(iterations=10)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
