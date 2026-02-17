import unittest

import numpy as np


def old_chunk_particles(p_dm, Lbox, N_chunk):
    """Reference implementation from IO_nbody.read()."""
    L_chunk = Lbox / N_chunk
    p_dm_list = []
    for x_min in np.linspace(0, Lbox - L_chunk, N_chunk):
        x_max = x_min + L_chunk
        if x_max == Lbox:
            x_max = 1.00001 * x_max
        for y_min in np.linspace(0, Lbox - L_chunk, N_chunk):
            y_max = y_min + L_chunk
            if y_max == Lbox:
                y_max = 1.00001 * y_max
            for z_min in np.linspace(0, Lbox - L_chunk, N_chunk):
                z_max = z_min + L_chunk
                if z_max == Lbox:
                    z_max = 1.00001 * z_max
                idx = np.where(
                    (p_dm["x"] >= x_min)
                    & (p_dm["x"] < x_max)
                    & (p_dm["y"] >= y_min)
                    & (p_dm["y"] < y_max)
                    & (p_dm["z"] >= z_min)
                    & (p_dm["z"] < z_max)
                )
                p_dm_list.append(p_dm[idx])
    return p_dm_list


def new_chunk_particles(p_dm, Lbox, N_chunk):
    """Proposed O(Ndm) chunk assignment implementation."""
    L_chunk = Lbox / N_chunk
    ix = np.floor(p_dm["x"] / L_chunk).astype(np.int64)
    iy = np.floor(p_dm["y"] / L_chunk).astype(np.int64)
    iz = np.floor(p_dm["z"] / L_chunk).astype(np.int64)
    ix = np.clip(ix, 0, N_chunk - 1)
    iy = np.clip(iy, 0, N_chunk - 1)
    iz = np.clip(iz, 0, N_chunk - 1)

    flat_chunk_id = ((ix * N_chunk) + iy) * N_chunk + iz
    n_total_chunks = int(N_chunk**3)
    order = np.argsort(flat_chunk_id, kind="stable")
    flat_sorted = flat_chunk_id[order]
    starts = np.searchsorted(flat_sorted, np.arange(n_total_chunks + 1), side="left")

    p_dm_list = []
    for chunk_id in range(n_total_chunks):
        i0, i1 = starts[chunk_id], starts[chunk_id + 1]
        if i1 > i0:
            p_dm_list.append(p_dm[order[i0:i1]])
        else:
            p_dm_list.append(p_dm[:0])
    return p_dm_list


def make_particles(n, Lbox, seed, with_boundaries=False):
    rng = np.random.default_rng(seed)
    dt = np.dtype([("id", np.int64), ("x", np.float64), ("y", np.float64), ("z", np.float64)])
    p_dm = np.zeros(n, dtype=dt)
    p_dm["id"] = np.arange(n, dtype=np.int64)
    p_dm["x"] = rng.uniform(0.0, Lbox, n)
    p_dm["y"] = rng.uniform(0.0, Lbox, n)
    p_dm["z"] = rng.uniform(0.0, Lbox, n)

    if with_boundaries and n >= 6:
        p_dm["x"][0] = 0.0
        p_dm["y"][1] = 0.0
        p_dm["z"][2] = 0.0
        p_dm["x"][3] = Lbox
        p_dm["y"][4] = Lbox
        p_dm["z"][5] = Lbox
    return p_dm


class TestDMChunkingEquivalence(unittest.TestCase):
    def assert_chunking_equivalent(self, p_dm, Lbox, N_chunk):
        old = old_chunk_particles(p_dm, Lbox, N_chunk)
        new = new_chunk_particles(p_dm, Lbox, N_chunk)
        self.assertEqual(len(old), len(new))
        self.assertEqual(sum(len(x) for x in old), len(p_dm))
        self.assertEqual(sum(len(x) for x in new), len(p_dm))

        for old_chunk, new_chunk in zip(old, new):
            np.testing.assert_array_equal(old_chunk["id"], new_chunk["id"])

    def test_random_inputs(self):
        Lbox = 128.0
        for N_chunk in (1, 2, 3, 4, 6):
            for seed in range(5):
                p_dm = make_particles(5000, Lbox, seed, with_boundaries=False)
                self.assert_chunking_equivalent(p_dm, Lbox, N_chunk)

    def test_boundary_values(self):
        Lbox = 200.0
        for N_chunk in (1, 2, 5):
            p_dm = make_particles(1000, Lbox, seed=123 + N_chunk, with_boundaries=True)
            self.assert_chunking_equivalent(p_dm, Lbox, N_chunk)


if __name__ == "__main__":
    unittest.main()
