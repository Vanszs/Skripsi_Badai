"""Unit tests for src.retrieval.base.RetrievalDatabase."""
import unittest

import numpy as np
import torch

from src.retrieval.base import RetrievalDatabase


class RetrievalDatabaseTest(unittest.TestCase):
    def _make_db(self, n=10, dim=4, data_dim=3):
        db = RetrievalDatabase(embedding_dim=dim)
        embeddings = np.random.RandomState(42).randn(n, dim).astype(np.float32)
        values = np.random.RandomState(43).randn(n, data_dim).astype(np.float32)
        db.add_items(embeddings, values)
        return db, values

    def test_query_shape(self):
        db, _ = self._make_db(n=10, dim=4, data_dim=3)
        query = np.random.randn(2, 4).astype(np.float32)
        out = db.query(query, k=3)
        self.assertEqual(tuple(out.shape), (2, 3, 3))
        self.assertEqual(out.dtype, torch.float32)

    def test_query_k_larger_than_n_raises(self):
        db, _ = self._make_db(n=5, dim=4, data_dim=3)
        query = np.random.randn(1, 4).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "k=10 is larger than the number of stored items"):
            db.query(query, k=10)

    def test_query_no_minus_one_indices(self):
        db, _ = self._make_db(n=10, dim=4, data_dim=3)
        query = np.random.randn(5, 4).astype(np.float32)
        out = db.query(query, k=3)
        # All returned values should be finite; -1 clamping is no longer allowed.
        self.assertTrue(torch.isfinite(out).all())

    def test_cache_invalidated_after_second_add(self):
        db, values1 = self._make_db(n=5, dim=4, data_dim=3)
        query = np.random.randn(1, 4).astype(np.float32)
        out1 = db.query(query, k=2)

        values2 = np.random.RandomState(44).randn(5, 3).astype(np.float32)
        embeddings2 = np.random.RandomState(45).randn(5, 4).astype(np.float32)
        db.add_items(embeddings2, values2)

        out2 = db.query(query, k=2)
        self.assertEqual(out2.shape[0], 1)
        # After adding more items, k=2 should still be valid.
        self.assertEqual(tuple(out2.shape), (1, 2, 3))

    def test_query_empty_database_raises(self):
        db = RetrievalDatabase(embedding_dim=4)
        query = np.random.randn(1, 4).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "RetrievalDatabase is empty"):
            db.query(query, k=1)
