"""Tests for the rigor study (multi-seed, sweeps, baselines).

The key check: each parametrized runner in rigor/rigor_study.py, at
seed offset 0 and the default settings, reproduces the recorded
single-seed results in the level results files exactly (rounded to
4 decimals). So the seed sweep extends the recorded runs, it does
not replace them with a different experiment.
"""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rigor"))

import rigor_study  # noqa: E402


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


BUILD = load("level-2-build/results_build.json")
DEEP = load("level-3-deep/results_deep.json")["deep"]


class TestDefaultSeedReproducesRecorded(unittest.TestCase):
    def test_sft_seed0_matches_build(self):
        r = rigor_study.run_sft(seed_offset=0)
        rec = BUILD["sft_pipeline"]
        self.assertEqual(r["before_test_accuracy"],
                         rec["before_training"]["test"]["accuracy"])
        self.assertEqual(r["final_test_accuracy"],
                         rec["after_training"]["test"]["accuracy"])

    def test_grpo_seed0_matches_build(self):
        r = rigor_study.run_grpo_cfg(seed_offset=0)
        rec = BUILD["grpo_track"]
        self.assertEqual(r["final_mean_reward"], rec["after_training"]["mean_reward"])
        self.assertEqual(r["final_kl_to_ref"], rec["final_mean_kl_to_ref"])
        self.assertEqual(r["final_greedy_accuracy"], rec["after_training"]["greedy_accuracy"])

    def test_dpo_seed0_matches_deep(self):
        r = rigor_study.run_dpo_cfg(seed_offset=0)
        self.assertEqual(r["held_out_margin"], DEEP["dpo"]["held_out_pairs_after"]["mean_margin"])
        self.assertEqual(r["train_margin"], DEEP["dpo"]["train_pairs_after"]["mean_margin"])
        self.assertEqual(r["mean_reward"], DEEP["dpo"]["task_after"]["mean_reward"])


class TestHelpers(unittest.TestCase):
    def test_summary_of_constant_values(self):
        s = rigor_study.summarize([0.5] * 10)
        self.assertEqual(s["mean"], 0.5)
        self.assertEqual(s["std"], 0.0)
        self.assertEqual(s["ci95"], [0.5, 0.5])

    def test_summary_ci_brackets_mean(self):
        s = rigor_study.summarize([0.1, 0.4, 0.2, 0.9, 0.5])
        self.assertLessEqual(s["ci95"][0], s["mean"])
        self.assertGreaterEqual(s["ci95"][1], s["mean"])

    def test_full_finetune_has_64_parameters_and_rank_counts(self):
        self.assertEqual(rigor_study.trainable_parameters("full"), 64)
        self.assertEqual([rigor_study.trainable_parameters(r) for r in (1, 2, 3, 4)],
                         [20, 40, 60, 80])

    def test_misaligned_pairs_flip_chosen_on_first_k_prompts(self):
        pairs = rigor_study.misaligned_pairs(2)
        correct = rigor_study.grpo_build.CORRECT
        self.assertEqual(pairs[0], (0, (int(correct[0]) + 1) % 4, int(correct[0])))
        self.assertEqual(pairs[2][1], int(correct[2]))


class TestGRPOGradientNormalization(unittest.TestCase):
    """The stated loss is -mean(surrogate) + beta * mean KL over prompts."""

    def setUp(self):
        gb = rigor_study.grpo_build
        rng = np.random.default_rng(5)
        self.logits = gb.init_policy() + rng.standard_normal((8, 4)) * 0.3
        self.ref = gb.softmax_rows(gb.init_policy())
        probs = gb.softmax_rows(self.logits)
        self.samples = []
        for p in range(8):
            acts = rng.choice(4, size=8, p=probs[p])
            adv, _, _ = gb.group_advantages(np.array([gb.checker(p, a) for a in acts]))
            self.samples.append((acts, adv, probs[p].copy()))
        num = np.zeros_like(self.logits)
        for i in range(8):
            for j in range(4):
                plus, minus = self.logits.copy(), self.logits.copy()
                plus[i, j] += 1e-6
                minus[i, j] -= 1e-6
                lp = gb.grpo_loss_and_grad(plus, self.ref, self.samples)[0]["loss"]
                lm = gb.grpo_loss_and_grad(minus, self.ref, self.samples)[0]["loss"]
                num[i, j] = (lp - lm) / 2e-6
        self.numeric = num

    def test_corrected_gradient_matches_finite_differences(self):
        grad = rigor_study.grpo_grad_corrected(self.logits, self.ref, self.samples)
        np.testing.assert_allclose(grad, self.numeric, atol=1e-6)

    def test_build_gradient_is_not_the_stated_loss_gradient(self):
        _, grad = rigor_study.grpo_build.grpo_loss_and_grad(self.logits, self.ref, self.samples)
        self.assertGreater(float(np.abs(grad - self.numeric).max()), 1e-3)


class TestSharedWeightDPO(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(rigor_study.run_linear_dpo(3, 2), rigor_study.run_linear_dpo(3, 2))

    def test_aligned_dpo_reaches_unseen_prompts_on_average(self):
        rows = [rigor_study.run_linear_dpo(s, 0) for s in range(10)]
        before = np.mean([r["unseen_greedy_before"] for r in rows])
        after = np.mean([r["unseen_greedy_after"] for r in rows])
        self.assertGreater(after, before)


class TestResultsFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load("rigor/results_rigor.json")

    def test_ten_seeds_everywhere(self):
        self.assertEqual(self.r["seeds"]["n_seeds"], 10)
        for track in ("sft", "grpo", "dpo"):
            self.assertEqual(len(self.r["seeds"][track]["per_seed"]), 10)

    def test_seed0_rows_match_recorded_results(self):
        s = self.r["seeds"]
        self.assertEqual(s["sft"]["per_seed"][0]["final_test_accuracy"], 0.9141)
        self.assertEqual(s["grpo"]["per_seed"][0]["final_mean_reward"], 0.9766)
        self.assertEqual(s["dpo"]["per_seed"][0]["held_out_margin"], 0.8297)

    def test_beta_zero_has_largest_kl(self):
        rows = self.r["grpo_beta_budget"]["rows"]
        final = {row["beta"]: row["checkpoints"][-1]["kl_to_ref"]["mean"] for row in rows}
        self.assertEqual(max(final, key=final.get), 0.0)

    def test_dpo_sweep_grid_is_complete(self):
        g = self.r["dpo_sweep"]
        self.assertEqual(len(g["rows"]),
                         len(g["grid"]["beta"]) * len(g["grid"]["learning_rate"]) * len(g["grid"]["steps"]))

    def test_site_fixture_is_build_init_policy(self):
        fixture = np.array(self.r["site_fixtures"]["init_policy_logits"])
        np.testing.assert_array_equal(fixture, rigor_study.grpo_build.init_policy())


if __name__ == "__main__":
    unittest.main()
