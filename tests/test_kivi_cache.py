from __future__ import annotations

import unittest
from types import SimpleNamespace

from profiles.kivi_cache import KIVILayerState, KIVICache, build_kivi_cache


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def index_select(self, dim: int, index):
        if dim != 0:
            raise AssertionError(f"unexpected dim {dim}")
        selected = [self.values[int(i)] for i in index.values]
        return FakeTensor(selected)


class FakeIndex:
    def __init__(self, values):
        self.values = list(values)


class KIVICacheTest(unittest.TestCase):
    def test_kivi_cache_tracks_per_layer_seq_length(self) -> None:
        cache = build_kivi_cache(
            SimpleNamespace(num_hidden_layers=3),
            residual_length=32,
            group_size=32,
            k_bits=4,
            v_bits=4,
        )

        cache.update(1, KIVILayerState(None, None, None, None, None, None, None, None, 17))

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.get_seq_length(1), 17)
        self.assertEqual(cache.get_seq_length(0), 0)

    def test_kivi_cache_reorder_cache_reorders_tensor_fields(self) -> None:
        cache = KIVICache(
            1,
            residual_length=32,
            group_size=32,
            k_bits=4,
            v_bits=4,
        )
        cache.update(
            0,
            KIVILayerState(
                key_q=FakeTensor(["k0", "k1"]),
                key_full=FakeTensor(["kf0", "kf1"]),
                key_scale=FakeTensor(["ks0", "ks1"]),
                key_mn=FakeTensor(["km0", "km1"]),
                value_q=FakeTensor(["v0", "v1"]),
                value_full=FakeTensor(["vf0", "vf1"]),
                value_scale=FakeTensor(["vs0", "vs1"]),
                value_mn=FakeTensor(["vm0", "vm1"]),
                kv_seq_len=9,
            ),
        )

        reordered = cache.reorder_cache(FakeIndex([1, 0]))
        state = reordered[0]

        self.assertIsNotNone(state)
        self.assertEqual(state.key_q.values, ["k1", "k0"])
        self.assertEqual(state.value_full.values, ["vf1", "vf0"])
        self.assertEqual(state.kv_seq_len, 9)


if __name__ == "__main__":
    unittest.main()
