"""Unit tests for Level 3, Deep: DPO, forgetting, memory study.

Hand-computed expectations:
- DPO loss at zero margin: logits equal to the reference, so
  every margin is 0, sigmoid(0) is 0.5, loss is log(2) = 0.6931.
- DPO gradient at zero margin, beta 0.1, one pair: factor is
  minus 0.1 * (1 - 0.5) = -0.05, so the gradient row is
  -0.05 at the chosen action and +0.05 at the rejected action.
- Memory accounting for a stated model with 1,000,000
  parameters, 2 layers, hidden 64, rank 16: full bytes are
  2,000,000; adapter parameters are 2 layers * 2 matrices *
  2 * 16 * 64 = 8,192, adapter bytes 16,384; 4-bit base bytes
  are 500,000; QLoRA total is 516,384.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))

import dpo_deep
import memory_study


class TestDPOLossAndGradient(unittest.TestCase):
    def test_loss_at_zero_margin_is_log_two(self):
        logits = np.zeros((1, 4))
        loss, _ = dpo_deep.dpo_loss_and_grad(logits, logits.copy(),
                                             [(0, 0, 1)], beta=0.1)
        self.assertAlmostEqual(loss, 0.6931, places=4)

    def test_gradient_hand_computed_case(self):
        logits = np.zeros((1, 4))
        _, grad = dpo_deep.dpo_loss_and_grad(logits, logits.copy(),
                                             [(0, 0, 1)], beta=0.1)
        self.assertAlmostEqual(float(grad[0, 0]), -0.05, places=6)
        self.assertAlmostEqual(float(grad[0, 1]), 0.05, places=6)
        self.assertAlmostEqual(float(grad[0, 2]), 0.0, places=6)
        self.assertAlmostEqual(float(grad[0, 3]), 0.0, places=6)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((2, 4)) * 0.4
        ref = rng.standard_normal((2, 4)) * 0.4
        pairs = [(0, 1, 2), (1, 3, 0)]
        _, grad = dpo_deep.dpo_loss_and_grad(logits, ref, pairs, beta=0.1)
        eps = 1e-6
        numeric = np.zeros_like(logits)
        for i in range(2):
            for j in range(4):
                plus = logits.copy()
                plus[i, j] += eps
                minus = logits.copy()
                minus[i, j] -= eps
                lp, _ = dpo_deep.dpo_loss_and_grad(plus, ref, pairs, beta=0.1)
                lm, _ = dpo_deep.dpo_loss_and_grad(minus, ref, pairs, beta=0.1)
                numeric[i, j] = (lp - lm) / (2 * eps)
        self.assertTrue(np.allclose(grad, numeric, atol=1e-5),
                        msg=f"grad {grad} vs numeric {numeric}")

    def test_pairs_rule(self):
        train_pairs, held_out_pairs = dpo_deep.make_preference_pairs()
        self.assertEqual(len(train_pairs), 8)
        self.assertEqual(len(held_out_pairs), 16)
        train_rejected = {(p, r) for p, _c, r in train_pairs}
        held_rejected = {(p, r) for p, _c, r in held_out_pairs}
        self.assertTrue(train_rejected.isdisjoint(held_rejected))
        for p, c, _r in train_pairs + held_out_pairs:
            self.assertEqual(c, int(dpo_deep.grpo_build.CORRECT[p]))


class TestDPOTraining(unittest.TestCase):
    def test_margin_and_preference_accuracy_improve(self):
        result = dpo_deep.run_deep()
        d = result["dpo"]
        self.assertGreater(d["train_pairs_after"]["mean_margin"],
                           d["train_pairs_before"]["mean_margin"])
        self.assertGreater(d["held_out_pairs_after"]["mean_margin"],
                           d["held_out_pairs_before"]["mean_margin"])
        self.assertGreater(d["held_out_pairs_after"]["preference_accuracy"],
                           d["held_out_pairs_before"]["preference_accuracy"])
        self.assertLess(d["loss_after"], d["loss_before"])

    def test_dpo_determinism(self):
        first = dpo_deep.train_dpo(steps=50)
        second = dpo_deep.train_dpo(steps=50)
        self.assertTrue(np.array_equal(first["logits"], second["logits"]))
        self.assertEqual(first["final_loss"], second["final_loss"])


class TestForgettingChecks(unittest.TestCase):
    def test_forgetting_change_arithmetic(self):
        initial = dpo_deep.grpo_build.init_policy()
        trained = dpo_deep.train_dpo(steps=50)
        checks = dpo_deep.forgetting_checks(initial, trained["logits"])
        self.assertAlmostEqual(
            checks["change"]["greedy_accuracy"],
            round(checks["after"]["greedy_accuracy"]
                  - checks["before"]["greedy_accuracy"], 4), places=4)
        self.assertAlmostEqual(
            checks["change"]["mean_reward"],
            round(checks["after"]["mean_reward"]
                  - checks["before"]["mean_reward"], 4), places=4)
        self.assertGreaterEqual(checks["mean_kl_after_to_before"], 0.0)
        self.assertEqual(checks["untouched_action_order_prompts"], 8)
        self.assertGreaterEqual(checks["untouched_action_order_retention"], 0.0)
        self.assertLessEqual(checks["untouched_action_order_retention"], 1.0)

    def test_identical_policy_has_zero_change(self):
        initial = dpo_deep.grpo_build.init_policy()
        checks = dpo_deep.forgetting_checks(initial, initial.copy())
        self.assertEqual(checks["change"]["greedy_accuracy"], 0.0)
        self.assertEqual(checks["change"]["mean_reward"], 0.0)
        self.assertEqual(checks["mean_kl_after_to_before"], 0.0)
        self.assertEqual(checks["untouched_action_order_retention"], 1.0)


class TestRewardHackingProbe(unittest.TestCase):
    def test_suppression_games_train_but_not_held_out(self):
        result = dpo_deep.run_deep()
        probe = result["reward_hacking_probe"]
        suppressed = probe["rejected_only_suppression_probe"]
        self.assertEqual(suppressed["train_preference_accuracy"], 1.0)
        self.assertLess(suppressed["held_out_preference_accuracy"],
                        suppressed["train_preference_accuracy"])
        constant = probe["constant_policy_probe"]
        self.assertLess(constant["train_preference_accuracy"], 0.5)
        self.assertIn("hacking_or_overfit_flagged",
                      probe["trained_policy_check"])


class TestMemoryStudy(unittest.TestCase):
    def test_memory_arithmetic_hand_case(self):
        row = memory_study.memory_row(
            {"name": "hand", "parameters": 1_000_000,
             "layers": 2, "hidden": 64}, rank=16)
        self.assertEqual(row["full_finetune_bytes"], 2_000_000)
        self.assertEqual(row["lora_parameters"], 8192)
        self.assertEqual(row["lora_adapter_bytes"], 16384)
        self.assertEqual(row["lora_total_bytes"], 2_016_384)
        self.assertEqual(row["qlora_base_4bit_bytes"], 500_000)
        self.assertEqual(row["qlora_total_bytes"], 516_384)

    def test_qlora_total_is_about_a_quarter_of_full(self):
        study = memory_study.memory_study()
        self.assertEqual(len(study["rows"]), 4)
        for row in study["rows"]:
            self.assertLess(row["qlora_total_bytes"], row["lora_total_bytes"])
            self.assertGreater(row["qlora_total_fraction_of_full"], 0.24)
            self.assertLess(row["qlora_total_fraction_of_full"], 0.27)
            self.assertEqual(
                row["qlora_total_bytes"],
                row["qlora_base_4bit_bytes"] + row["lora_adapter_bytes"])


if __name__ == "__main__":
    unittest.main()
