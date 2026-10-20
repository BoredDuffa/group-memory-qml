"""Verify the split unit and the deliberately competitive classical baseline."""

import numpy as np
import pytest

from group_memory_qml.data import (
    classical_features,
    four_views,
    make_sensor_dataset,
    orbit_key,
)


@pytest.mark.parametrize("n", [6, 10])
def test_split_before_augmentation_with_no_shared_exact_orbits(n):
    data = make_sensor_dataset(n, seed=846101)
    assert data.train_x.shape == data.test_x.shape == (32, n)
    assert set(data.train_base_ids).isdisjoint(data.test_base_ids)
    train_orbits = {orbit_key(x) for x in data.train_x}
    test_orbits = {orbit_key(x) for x in data.test_x}
    assert len(train_orbits) == len(test_orbits) == 8
    assert train_orbits.isdisjoint(test_orbits)
    for X, y, ids in (
        (data.train_x, data.train_y, data.train_base_ids),
        (data.test_x, data.test_y, data.test_base_ids),
    ):
        assert np.bincount(y).tolist() == [16, 16]
        assert len(set(ids)) == 8
        for start in range(0, len(X), 4):
            np.testing.assert_array_equal(X[start:start + 4], four_views(X[start]))
            assert len(set(ids[start:start + 4])) == 1
            assert len(set(y[start:start + 4])) == 1


@pytest.mark.parametrize("n", [6, 10])
def test_each_opposite_label_pair_has_identical_amplitude_histogram(n):
    data = make_sensor_dataset(n, seed=846102)
    for X, y in ((data.train_x, data.train_y), (data.test_x, data.test_y)):
        for start in range(0, 32, 8):
            contiguous, alternating = X[start], X[start + 4]
            assert (y[start], y[start + 4]) == (0, 1)
            np.testing.assert_array_equal(np.sort(contiguous), np.sort(alternating))
            assert np.all(contiguous[: n // 2] > 0)
            assert np.all(contiguous[n // 2:] < 0)
            assert np.all(alternating[::2] > 0)
            assert np.all(alternating[1::2] < 0)
            assert orbit_key(contiguous) != orbit_key(alternating)


def test_seed_reproduction_and_changed_seed_identity():
    first = make_sensor_dataset(6, seed=846101)
    repeat = make_sensor_dataset(6, seed=846101)
    different = make_sensor_dataset(6, seed=846102)
    assert first.dataset_sha256 == repeat.dataset_sha256
    assert first.dataset_sha256 != different.dataset_sha256
    np.testing.assert_array_equal(first.train_x, repeat.train_x)
    np.testing.assert_array_equal(first.test_x, repeat.test_x)
    # Independently regenerating data must not depend on NumPy's global RNG.
    np.random.seed(999)
    independent = make_sensor_dataset(6, seed=846101)
    np.testing.assert_array_equal(first.train_x, independent.train_x)


@pytest.mark.parametrize("n", [6, 10])
def test_classical_features_receive_the_same_dihedral_symmetry(n):
    x = np.random.default_rng(431).normal(size=n)
    all_views = np.stack([
        np.roll(v, k) for v in (x, x[::-1]) for k in range(n)
    ])
    features = classical_features(all_views)
    np.testing.assert_allclose(
        features, np.repeat(features[:1], 2 * n, axis=0), rtol=0, atol=1e-12
    )
    # The comparator is informative for the task, not a deliberately weak
    # histogram-only baseline that must collapse the two classes.
    data = make_sensor_dataset(n)
    pair_features = classical_features(data.train_x[[0, 4]])
    assert np.linalg.norm(pair_features[0] - pair_features[1]) > 1e-2


def test_orbit_key_is_exact_and_not_a_near_duplicate_matcher():
    x = np.random.default_rng(437).normal(size=6)
    assert all(orbit_key(view) == orbit_key(x) for view in four_views(x))
    nearby = x.copy()
    nearby[0] = np.nextafter(nearby[0], np.inf)
    assert orbit_key(nearby) != orbit_key(x)
