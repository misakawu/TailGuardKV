from __future__ import annotations

import unittest
from types import SimpleNamespace

from profiles.h2o_cache import H2OCache, H2OLayerState, build_h2o_cache


class FakeTensor:
    def __init__(self, values, *, shape=None, element_size=2, dtype="float16"):
        self.values = values
        self.shape = shape or (len(values), len(values[0]) if values and isinstance(values[0], list) else len(values))
        self._element_size = element_size
        self.dtype = dtype

    def index_select(self, dim: int, index):
        if dim != 0:
            raise AssertionError(f"unexpected dim {dim}")
        return FakeTensor([self.values[int(i)] for i in index.values])

    def numel(self):
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    def element_size(self):
        return self._element_size


class FakeIndex:
    def __init__(self, values):
        self.values = list(values)


class H2OCacheTest(unittest.TestCase):
    def test_h2o_cache_separates_logical_and_physical_lengths(self) -> None:
        cache = build_h2o_cache(SimpleNamespace(num_hidden_layers=2), heavy_size=2, recent_size=1)
        cache.update_pruned(
            1,
            H2OLayerState(
                key_states=SimpleNamespace(shape=(1, 1, 3, 4)),
                value_states=SimpleNamespace(shape=(1, 1, 3, 4)),
                hh_score=object(),
                logical_seq_len=9,
            ),
        )

        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.get_seq_length(1), 9)
        self.assertEqual(cache.get_physical_seq_length(1), 3)
        self.assertEqual(cache.h2o_kept_tokens(1), 3)

    def test_h2o_cache_reorder_cache_reorders_tensors_but_keeps_logical_length(self) -> None:
        cache = H2OCache(1, heavy_size=2, recent_size=1)
        cache.update_pruned(
            0,
            H2OLayerState(
                key_states=FakeTensor(["k0", "k1"]),
                value_states=FakeTensor(["v0", "v1"]),
                hh_score=FakeTensor(["s0", "s1"]),
                logical_seq_len=7,
            ),
        )

        reordered = cache.reorder_cache(FakeIndex([1, 0]))
        state = reordered[0]

        self.assertIsNotNone(state)
        self.assertEqual(state.key_states.values, ["k1", "k0"])
        self.assertEqual(state.value_states.values, ["v1", "v0"])
        self.assertEqual(state.hh_score.values, ["s1", "s0"])
        self.assertEqual(state.logical_seq_len, 7)

    def test_h2o_cache_legacy_cache_has_placeholder_for_empty_layers(self) -> None:
        cache = H2OCache(2, heavy_size=2, recent_size=1)
        cache.update_pruned(
            1,
            H2OLayerState(
                key_states="k",
                value_states="v",
                hh_score="s",
                logical_seq_len=5,
            ),
        )

        legacy = cache.to_legacy_cache()

        self.assertEqual(len(legacy), 2)
        self.assertEqual(legacy[0], (None, None, None, 0))
        self.assertEqual(legacy[1], ("k", "v", "s", 5))

    def test_h2o_cache_memory_uses_physical_kept_tokens_not_logical_length(self) -> None:
        cache = H2OCache(1, heavy_size=2, recent_size=1)
        cache.update_pruned(
            0,
            H2OLayerState(
                key_states=FakeTensor([], shape=(1, 1, 3, 4), element_size=2),
                value_states=FakeTensor([], shape=(1, 1, 3, 4), element_size=2),
                hh_score=FakeTensor([], shape=(1, 3), element_size=4),
                logical_seq_len=99,
            ),
        )

        expected_bytes = (1 * 1 * 3 * 4 * 2) + (1 * 1 * 3 * 4 * 2) + (1 * 3 * 4)
        self.assertEqual(cache.h2o_kept_tokens(0), 3)
        self.assertEqual(cache.kv_cache_memory_mib(), expected_bytes / 1024 / 1024)


if __name__ == "__main__":
    unittest.main()
