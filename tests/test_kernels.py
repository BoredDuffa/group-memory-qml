import numpy as np
from qiskit.quantum_info import Statevector

from group_memory_qml.experiments.kernels import (
    KernelComputer, feature_circuit, make_data, normalize_and_check, permutations,
)


def direct_state(x, symmetry=True):
    circuit, parameters = feature_circuit(len(x), symmetry)
    return Statevector.from_instruction(circuit.assign_parameters(dict(zip(parameters, x)))).data


def direct_fidelity(x, y, symmetry=True):
    return float(abs(np.vdot(direct_state(x, symmetry), direct_state(y, symmetry))) ** 2)


def test_raw_full_kernel_matches_logical_reference_and_joint_rule():
    rng = np.random.default_rng(98102)
    x, y = rng.normal(0, .7, (2, 6))
    p = permutations(6)[2]
    inputs = np.stack([x, x[p], y, y[p]])
    expected = np.asarray([[direct_fidelity(a, b) for b in inputs] for a in inputs])
    for method in ("group_memory", "qiskit_symmetry", "qiskit_exact"):
        computer = KernelComputer(6, method)
        np.testing.assert_allclose(computer.matrix(inputs, inputs), expected, atol=1e-12, rtol=0)
    computer = KernelComputer(6)
    assert computer.pair(x, y) == computer.pair(x[p], y[p])
    assert computer.pair(x, y) != computer.pair(x[p], y)
    # Independent canonicalization would falsely make all self-orbit pairs unity.
    assert direct_fidelity(x, x[p]) < .99


def test_broken_raw_symmetry_disables_reuse():
    rng = np.random.default_rng(123)
    x, y = rng.normal(0, .8, (2, 6))
    p = permutations(6)[1]
    computer = KernelComputer(6, "group_memory", symmetry=False)
    assert not computer.group_enabled
    assert computer.pair(x, y) != computer.pair(x[p], y[p])
    result = computer.matrix(np.stack([x, x[p]]), np.stack([y, y[p]]))
    expected = np.array([[direct_fidelity(a, b, False) for b in (y, y[p])] for a in (x, x[p])])
    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)
    assert abs(expected[0, 0] - expected[1, 1]) > 1e-5


def test_averaged_kernel_matches_full_double_orbit_and_psd():
    rng = np.random.default_rng(213)
    x, y = rng.normal(0, .7, (2, 6))
    p = permutations(6)[1]
    inputs = np.stack([x, y, x[p]])
    # Independent logical states and explicit |G|^2 fidelity average.
    states = [np.asarray([direct_state(item[perm]) for perm in permutations(6)]) for item in inputs]
    expected = np.asarray([[np.mean(abs(a.conj() @ b.T) ** 2) for b in states] for a in states])
    for method in ("group_memory", "qiskit_symmetry", "qiskit_exact"):
        computer = KernelComputer(6, method, averaged=True)
        actual = computer.matrix(inputs, inputs)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        normalized, _, receipt = normalize_and_check(actual, actual, np.diag(actual))
        assert np.linalg.eigvalsh(normalized).min() >= -1e-12
        assert receipt["diagonal_jitter"] < 1e-12
        assert computer.orbit_members_averaged == 12 * computer.density_averages
    assert np.diag(expected).max() < 1
    np.testing.assert_allclose(expected[0], expected[2], atol=1e-12, rtol=0)


def test_averaging_is_invariant_even_when_feature_map_is_broken():
    rng = np.random.default_rng(141)
    x = rng.normal(0, .8, 6)
    p = permutations(6)[3]
    computer = KernelComputer(6, "group_memory", averaged=True, symmetry=False)
    actual = computer.matrix(np.stack([x, x[p]]), np.stack([x, x[p]]))
    np.testing.assert_allclose(actual, np.full((2, 2), actual[0, 0]), atol=1e-12, rtol=0)


def test_self_diagonal_does_not_compute_off_diagonal_pairs():
    rng = np.random.default_rng(312)
    x, y = rng.normal(0, .7, (2, 6))
    inputs = np.stack([x, y, x[permutations(6)[1]]])
    expected = []
    for item in inputs:
        states = np.asarray([direct_state(item[p]) for p in permutations(6)])
        expected.append(np.mean(abs(states.conj() @ states.T) ** 2))
    for method in ("group_memory", "qiskit_symmetry", "qiskit_exact"):
        raw = KernelComputer(6, method)
        np.testing.assert_array_equal(raw.diagonal(inputs), np.ones(3))
        assert raw.state_preparations == raw.kernel_overlaps == 0
        averaged = KernelComputer(6, method, averaged=True)
        np.testing.assert_allclose(averaged.diagonal(inputs), expected, atol=1e-12, rtol=0)
        assert all(a == b for a, b in averaged.values)
        assert averaged.kernel_overlaps == (3 if method == "qiskit_exact" else 2)
        assert averaged.requested_pairs == len(inputs)
        before = averaged.kernel_overlaps
        averaged.diagonal(inputs)
        assert averaged.kernel_overlaps == before


def test_data_splits_are_orbit_disjoint():
    data = make_data(6, 777, orientation_shift=True)
    assert not set(data["train"]["base_ids"]) & set(data["test"]["base_ids"])
    assert not set(data["train"]["orbit_keys"]) & set(data["test"]["orbit_keys"])
    assert len(data["train"]["x"]) == 16
    assert np.bincount(data["test"]["y"]).tolist() == [8, 8]


def test_psd_policy_rejects_large_negative_eigenvalue():
    import pytest
    with pytest.raises(AssertionError, match="not PSD"):
        normalize_and_check(np.array([[1., 2.], [2., 1.]]), np.ones((1, 2)), np.ones(1))
