import unittest

from src.config import (
    MAIN_NODE_NAME,
    NODE_NAMES,
    NUM_NODES,
    STAR_EDGE_COUNT,
    STAR_EDGES,
    build_star_edge_index,
)


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
