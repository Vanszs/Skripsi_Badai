import unittest

import numpy as np

from src.config import (
    MAIN_NODE_NAME,
    NODE_NAMES,
    NUM_NODES,
    STAR_EDGE_COUNT,
    STAR_EDGES,
    build_star_edge_attr,
    build_star_edge_index,
)
from src.evaluation.probabilistic_metrics import compute_crps, compute_mae


class ConfigTopologyTest(unittest.TestCase):
    def test_canonical_node_order_and_size(self):
        self.assertEqual(NODE_NAMES, ["MAIN", "UP", "DOWN", "LEFT", "RIGHT"])
        self.assertEqual(NUM_NODES, 5)
        self.assertEqual(MAIN_NODE_NAME, "MAIN")

    def test_star_topology_edges(self):
        edge_index = build_star_edge_index(NODE_NAMES)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], STAR_EDGE_COUNT)
        self.assertEqual(STAR_EDGE_COUNT, 8)
        self.assertEqual(len(STAR_EDGES), 8)

    def test_star_edge_attr_shape_and_positivity(self):
        # FIX #6: per-edge distance attribute must align with edge index.
        # NOTE: distance is constant (0.25 deg) on the equidistant grid by design.
        edge_attr = build_star_edge_attr(NODE_NAMES)
        self.assertEqual(tuple(edge_attr.shape), (STAR_EDGE_COUNT, 1))
        self.assertTrue(bool((edge_attr > 0).all()))


class CRPSNumericTest(unittest.TestCase):
    def test_crps_single_member_equals_mae(self):
        # FIX #11: a deterministic (1-member) forecast must yield CRPS == MAE, not NaN.
        ens = np.array([[3.0], [10.0]], dtype=float)  # [N_timesteps, 1]
        obs = np.array([5.0, 7.0], dtype=float)
        crps = compute_crps(ens, obs)
        mae = compute_mae(ens[:, 0], obs)
        self.assertFalse(np.isnan(crps))
        self.assertAlmostEqual(crps, mae, places=6)

    def test_crps_two_member_fair_estimator(self):
        # FIX #11 (corrected): CRPS = E|X-y| - 0.5*E|X-X'|.
        # samples=[2,4], obs=5 -> term1=2.0, 0.5*E|X-X'|=1.0 -> CRPS=1.0
        ens = np.array([[2.0, 4.0]], dtype=float)
        obs = np.array([5.0], dtype=float)
        self.assertAlmostEqual(compute_crps(ens, obs), 1.0, places=6)
