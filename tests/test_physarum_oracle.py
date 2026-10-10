"""Fast sanity tests (unittest, so no extra dependencies; pytest also runs them)."""
import os
import sys
import unittest
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import physarum_oracle as po  # noqa: E402

SMALL = replace(po.Params(), steps=30, batch=8, top_k=10, sel_steps=20)


class TestWorld(unittest.TestCase):
    def test_world_is_connected_and_deterministic(self):
        e1, pos, comm = po.make_world(n=60, seed=3)
        e2, _, _ = po.make_world(n=60, seed=3)
        self.assertTrue(po.is_connected(60, e1))
        np.testing.assert_array_equal(e1, e2)
        self.assertEqual(pos.shape, (60, 2))

    def test_holdout_keeps_graph_connected(self):
        e, _, _ = po.make_world(n=60, seed=1)
        train, held = po.holdout(60, e, 0.15, np.random.default_rng(0))
        self.assertTrue(po.is_connected(60, train))
        self.assertEqual(len(train) + len(held), len(e))
        self.assertGreater(len(held), 0)


class TestMetrics(unittest.TestCase):
    def test_auc_extremes_and_ties(self):
        self.assertEqual(po.auc_score(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0)
        self.assertEqual(po.auc_score(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)
        self.assertEqual(po.auc_score(np.array([1.0]), np.array([1.0])), 0.5)

    def test_resistance_of_a_path(self):
        # unit-conductance path 0-1-2: R(0,2) = 2, R(0,1) = 1
        edges = np.array([[0, 1], [1, 2]])
        R = po.resistance(edges, 3, np.ones(2))
        self.assertAlmostEqual(R[0, 2], 2.0, places=6)
        self.assertAlmostEqual(R[0, 1], 1.0, places=6)


class TestOrganism(unittest.TestCase):
    def test_adapt_conserves_mass_and_stays_positive(self):
        e, _, _ = po.make_world(n=50, seed=2)
        hops = po.hop_matrix(50, e)
        local = po.local_lists(hops, SMALL.local_hops)
        D0 = np.full(len(e), SMALL.d0)
        D, _, _ = po.adapt(e, 50, D0, local, SMALL, np.random.default_rng(0), SMALL.steps)
        self.assertTrue((D > 0).all())
        self.assertAlmostEqual(D.sum(), D0.sum(), delta=0.02 * D0.sum() + len(e) * SMALL.d_floor)

    def test_pipeline_outputs(self):
        e, _, _ = po.make_world(n=50, seed=4)
        train, _ = po.holdout(50, e, 0.15, np.random.default_rng(1))
        res = po.run_oracle(50, train, SMALL, seed=0)
        K = SMALL.top_k
        self.assertEqual(res["cand"].shape, (K, 2))
        self.assertEqual(len(res["vigor"]), K)
        existing = {tuple(x) for x in train.tolist()}
        for a, b in res["cand"].tolist():
            self.assertLess(a, b)
            self.assertNotIn((a, b), existing)  # proposals are genuinely missing links

    def test_evaluate_world_beats_chance(self):
        row = po.evaluate_world(5, SMALL, n=60)
        self.assertGreater(row["auc"]["Physarum resistance"], 0.6)


if __name__ == "__main__":
    unittest.main()
