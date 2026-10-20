"""Independent full-state checks for local cones and finite-grid symmetry."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from group_memory_qml.experiments import local


def _full_ring(x, symmetry=True):
    n = len(x)
    beta = np.full(n, .53) if symmetry else np.linspace(.35, .73, n)
    alpha = np.full(n, -.41) if symmetry else np.linspace(-.6, -.22, n)
    circuit = QuantumCircuit(n)
    for j in range(n):
        circuit.ry(x[j] + .31, j)
    for j in range(n):
        circuit.rzz(beta[j], j, (j + 1) % n)
    for j in range(n):
        circuit.rx(alpha[j], j)
        circuit.ry(.47, j)
    probability = Statevector.from_instruction(circuit).probabilities()
    index = np.arange(2**n, dtype=np.uint64)
    return np.asarray([probability @ (1. - 2. * ((index >> j) & 1)) for j in range(n)])


@pytest.mark.parametrize("n,symmetry", [(8, True), (12, True), (8, False)])
def test_connected_ring_cones_equal_full_state_for_nonuniform_inputs(n, symmetry):
    X = np.random.default_rng(n + 173).uniform(-1., 1., (2, n))
    expected = np.stack([_full_ring(x, symmetry) for x in X])
    for method in local.METHODS:
        actual, receipt = local.ring_features(X, method, symmetry)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        assert receipt["max_simulated_qubits"] <= 3
        if not symmetry:
            assert not receipt["group_enabled"]


@pytest.mark.parametrize("side", [2, 3])
def test_grid_rotation_reflection_reuse_matches_uncompiled_full_state(side):
    n = side * side
    x = np.random.default_rng(73).uniform(-.9, .9, n)
    X = x[local._grid_permutations(side)]
    source, parameters = local._patch_source(side, True)
    expected = []
    indices = np.arange(2**n, dtype=np.uint64)
    signs = np.asarray([1 - 2. * ((indices >> j) & 1) for j in range(n)])
    for row in X:
        probability = Statevector.from_instruction(
            source.assign_parameters(dict(zip(parameters, row)))
        ).probabilities()
        expected.append(np.mean(signs @ probability))
    for method in local.METHODS[:3]:
        actual, receipt = local.patch_features(X, method, side)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        assert receipt["evaluations"] == (8 if method == "qiskit_exact" else 1)


def test_broken_patch_symmetry_falls_back_to_exact_cache():
    x = np.random.default_rng(827).normal(size=4)
    X = x[local._grid_permutations(2)]
    reference, _ = local.patch_features(X, "qiskit_exact", symmetry=False)
    for method in ("group_memory", "qiskit_symmetry"):
        actual, receipt = local.patch_features(X, method, symmetry=False)
        np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=0)
        assert receipt["evaluations"] == 8
        assert not receipt["group_enabled"]
    assert np.ptp(reference) > 1e-4


def test_image_bases_split_before_views_without_whole_image_orbit_leakage():
    train, test = local._texture_data(2, 831)
    def key(image):
        return min(np.rot90(view, j).copy().tobytes()
                   for view in (image, image[:, ::-1]) for j in range(4))
    assert set(train[2]).isdisjoint(test[2])
    assert {key(x) for x in train[0]}.isdisjoint(key(x) for x in test[0])
    for images, labels, ids in (train, test):
        assert len(set(ids)) * 4 == len(images)
        # Perturbations preserve class histograms; mean intensity cannot solve.
        np.testing.assert_array_equal(images.sum(axis=(1, 2)), 32)
        for start in range(0, len(images), 4):
            assert len(set(labels[start:start + 4])) == 1
            assert len(set(ids[start:start + 4])) == 1


def test_ring_data_labels_are_histogram_matched_and_split_by_orbit():
    train, test = local._ring_data(64, 973)
    def key(x):
        return min(np.roll(v, j).tobytes()
                   for v in (x, x[::-1]) for j in range(len(x)))
    assert set(train[2]).isdisjoint(test[2])
    assert {key(x) for x in train[0]}.isdisjoint(key(x) for x in test[0])
    for X, y, _ in (train, test):
        for start in range(0, len(X), 8):
            assert (y[start], y[start + 4]) == (0, 1)
            np.testing.assert_array_equal(np.sort(X[start]), np.sort(X[start + 4]))


def test_three_methods_have_equal_reconstructed_features_at_256_sites():
    base = np.random.default_rng(812).uniform(-1., 1., 8)
    X = np.stack([np.tile(base, 32), np.tile(base[::-1], 32)])
    values = {}
    receipts = {}
    for method in local.METHODS:
        values[method], receipts[method] = local.ring_features(X, method)
    for value in values.values():
        np.testing.assert_allclose(value, values["classical"], atol=1e-10, rtol=0)
    assert receipts["group_memory"]["evaluations"] == 8
    assert receipts["qiskit_symmetry"]["evaluations"] == 8
    assert receipts["qiskit_exact"]["evaluations"] == 16


@pytest.mark.parametrize("experiment,config", [("4A", {"patch_side": 2}),
                                              ("4B", {"n_global": 64})])
def test_actual_trained_head_and_matched_quantum_predictions(experiment, config):
    rows = [local.run(experiment, 846101, method, config) for method in local.METHODS]
    for row in rows:
        assert row["metric_name"] == "heldout_accuracy"
        assert row["metric_value"] > .5  # this fixed small synthetic fixture only
        assert row["diagnostics"]["base_ids_disjoint"]
        assert row["diagnostics"]["learning"].startswith(("Only", "Train-only"))
    reference = np.asarray(rows[2]["validation_vector"])
    for row in rows[:2]:
        np.testing.assert_allclose(row["validation_vector"], reference, rtol=0, atol=1e-10)
        np.testing.assert_allclose(row["diagnostics"]["test_probabilities"],
                                   rows[2]["diagnostics"]["test_probabilities"],
                                   atol=1e-9, rtol=0)
    assert rows[0]["evaluations"] == rows[1]["evaluations"]
    assert rows[0]["total_cx"] == rows[1]["total_cx"]
