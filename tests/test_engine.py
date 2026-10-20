"""Check physical outputs against an independent, uncompiled statevector oracle.

The oracle builds the declared circuit directly instead of using the engine's
builder or canonicalization. Finite differences are used only in tests; the
demo's training gradients use Qiskit's reverse-mode implementation.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from group_memory_qml.engine import QMLReuseEngine


THETA = np.array([0.53, -0.41], dtype=np.float64)
METHODS = ("group_memory", "qiskit_same_symmetry", "qiskit_exact_cache")


def sample(n):
    """Distinct, deterministic angles with no accidental stabilizer."""
    return np.random.default_rng(113 + n).uniform(-1.15, 1.15, n)


def views(x):
    return np.stack((x, np.roll(x, 1), x[::-1], np.roll(x[::-1], 2)))


def oracle_local_z(x, theta, symmetry=True):
    n = len(x)
    circuit = QuantumCircuit(n)
    for q in range(n):
        circuit.ry(float(x[q]) + 0.31, q)
    for q in range(n):
        circuit.rzz(float(theta[0]), q, (q + 1) % n)
    gains = np.ones(n) if symmetry else np.linspace(0.65, 1.35, n)
    for q in range(n):
        circuit.rx(float(gains[q] * theta[1]), q)
        circuit.ry(0.47, q)
    probabilities = Statevector.from_instruction(circuit).probabilities()
    indices = np.arange(2**n, dtype=np.uint64)
    return np.asarray(
        [np.dot(probabilities, 1.0 - 2.0 * ((indices >> q) & 1)) for q in range(n)]
    )


@pytest.mark.parametrize("n", [6, 10])
def test_every_rotation_and_reflection_matches_independent_oracle(n):
    x = sample(n)
    permutations = [
        np.roll(np.arange(n), k) for k in range(n)
    ] + [np.roll(np.arange(n)[::-1], k) for k in range(n)]
    X = np.stack([x[p] for p in permutations])
    engine = QMLReuseEngine(n)
    actual, gradient = engine.evaluate(X, THETA)
    expected = np.stack([oracle_local_z(row, THETA) for row in X])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual, oracle_local_z(x, THETA)[permutations],
                               rtol=0, atol=1e-10)
    assert gradient is None
    assert engine.forward_count == 1
    assert engine.gradient_count == 0
    assert engine.requests == 2 * n


@pytest.mark.parametrize("n", [6, 10])
def test_shared_weight_gradients_match_independent_finite_differences(n):
    X = views(sample(n))
    engine = QMLReuseEngine(n, gradients=True)
    _, actual = engine.evaluate(X, THETA, gradient=True)
    # This differentiates the entire circuit: each parameter controls n gates.
    # Applying a single-gate shift to a shared parameter would be incorrect.
    delta = 1e-6
    expected = np.empty((len(X), 2))
    for j in range(2):
        plus, minus = THETA.copy(), THETA.copy()
        plus[j] += delta
        minus[j] -= delta
        for i, x in enumerate(X):
            expected[i, j] = (
                oracle_local_z(x, plus).mean() - oracle_local_z(x, minus).mean()
            ) / (2 * delta)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    assert engine.forward_count == engine.gradient_count == 1


def test_same_symmetry_qiskit_is_equally_informed_and_reuses_equally():
    X = views(sample(6))
    outputs = {}
    for method in METHODS:
        engine = QMLReuseEngine(6, method=method, gradients=True)
        outputs[method] = engine.evaluate(X, THETA, gradient=True)
        expected = 4 if method == "qiskit_exact_cache" else 1
        assert engine.forward_count == expected
        assert engine.gradient_count == expected
        assert engine.requests == 4
        assert engine.group_enabled == (method != "qiskit_exact_cache")
    for Z, J in outputs.values():
        np.testing.assert_allclose(Z, outputs["group_memory"][0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(J, outputs["group_memory"][1], rtol=0, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_exact_repeat_is_cached_and_one_ulp_weight_change_invalidates(method):
    engine = QMLReuseEngine(6, method=method, gradients=True)
    X = views(sample(6))
    first = engine.evaluate(X, THETA, gradient=True)
    count = engine.forward_count
    second = engine.evaluate(X.copy(), THETA.copy(), gradient=True)
    assert engine.forward_count == count
    assert engine.gradient_count == count
    assert engine.invalidations == 0
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    changed = THETA.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    engine.evaluate(X, changed, gradient=True)
    assert engine.forward_count == 2 * count
    assert engine.gradient_count == 2 * count
    assert engine.invalidations == 1


@pytest.mark.parametrize("method", METHODS)
def test_gradient_upgrade_does_not_repeat_forward_simulation(method):
    engine = QMLReuseEngine(6, method=method, gradients=True)
    X = views(sample(6))
    before, no_gradient = engine.evaluate(X, THETA)
    count = engine.forward_count
    assert no_gradient is None
    assert engine.gradient_count == 0
    after, J = engine.evaluate(X, THETA, gradient=True)
    np.testing.assert_array_equal(after, before)
    assert J.shape == (4, 2)
    assert engine.forward_count == count
    assert engine.gradient_count == count
    engine.evaluate(X, THETA, gradient=True)
    assert engine.forward_count == engine.gradient_count == count


@pytest.mark.parametrize("method", METHODS)
def test_broken_symmetry_disables_orbit_reuse_and_preserves_physics(method):
    X = views(sample(6))
    engine = QMLReuseEngine(6, method=method, symmetry=False, gradients=True)
    actual, J = engine.evaluate(X, THETA, gradient=True)
    expected = np.stack([oracle_local_z(x, THETA, symmetry=False) for x in X])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert not engine.group_enabled
    assert engine.forward_count == engine.gradient_count == 4
    # The control really breaks the claimed equivariance, rather than merely
    # turning the cache off while retaining an equivariant architecture.
    assert np.max(np.abs(actual[1] - np.roll(actual[0], 1))) > 1e-4
    delta = 1e-6
    expected_J = np.empty_like(J)
    for p in range(2):
        plus, minus = THETA.copy(), THETA.copy()
        plus[p] += delta
        minus[p] -= delta
        for i, x in enumerate(X):
            expected_J[i, p] = (
                oracle_local_z(x, plus, False).mean()
                - oracle_local_z(x, minus, False).mean()
            ) / (2 * delta)
    np.testing.assert_allclose(J, expected_J, rtol=0, atol=2e-8)


def test_nearby_inputs_are_not_merged_by_rounding():
    x = sample(6)
    changed = x.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    engine = QMLReuseEngine(6)
    engine.evaluate(np.stack([x, changed]), THETA)
    assert engine.forward_count == 2


def test_user_mutations_cannot_poison_cached_values_or_model():
    engine = QMLReuseEngine(6, gradients=True)
    X = views(sample(6))
    Z, J = engine.evaluate(X, THETA, gradient=True)
    expected_Z, expected_J = Z.copy(), J.copy()
    Z[:] = 12345
    J[:] = -12345
    exposed_source, exposed_native = engine.source, engine.native
    exposed_source.x(0)
    exposed_native.x(0)
    gains = engine.gains
    if gains.flags.writeable:
        gains[:] = 12345
    restored_Z, restored_J = engine.evaluate(X, THETA, gradient=True)
    np.testing.assert_array_equal(restored_Z, expected_Z)
    np.testing.assert_array_equal(restored_J, expected_J)
    assert "save_expectation_value" not in engine.native.count_ops()
    assert len(engine.source.data) + 1 == len(exposed_source.data)
    assert len(engine.native.data) + 1 == len(exposed_native.data)
    # Force a cache miss, so this also checks protection of future evaluations.
    new_x = sample(6) * 0.9
    fresh_Z, _ = engine.evaluate(new_x[None, :], THETA)
    np.testing.assert_allclose(fresh_Z[0], oracle_local_z(new_x, THETA),
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "X,theta",
    [
        (np.zeros(6), THETA),
        (np.zeros((2, 5)), THETA),
        (np.zeros((1, 6, 1)), THETA),
        (np.full((1, 6), np.nan), THETA),
        (np.full((1, 6), np.inf), THETA),
        (np.zeros((1, 6)), [0.5]),
        (np.zeros((1, 6)), [[0.5, 0.6]]),
        (np.zeros((1, 6)), [0.5, np.inf]),
    ],
)
def test_invalid_inputs_rejected_without_accounting_or_cache_changes(X, theta):
    engine = QMLReuseEngine(6)
    with pytest.raises(ValueError):
        engine.evaluate(X, theta)
    assert engine.requests == engine.forward_count == engine.invalidations == 0


def test_empty_batch_is_a_noop_even_with_different_weights():
    engine = QMLReuseEngine(6, gradients=True)
    engine.evaluate(sample(6)[None, :], THETA)
    Z, J = engine.evaluate(np.empty((0, 6)), THETA + 0.1, gradient=True)
    assert Z.shape == (0, 6)
    assert J.shape == (0, 2)
    assert engine.requests == engine.forward_count == 1
    assert engine.invalidations == engine.gradient_count == 0


def test_unconfigured_gradient_backend_rejects_request():
    engine = QMLReuseEngine(6, gradients=False)
    with pytest.raises(ValueError):
        engine.evaluate(sample(6)[None, :], THETA, gradient=True)
    assert engine.requests == engine.forward_count == 0


@pytest.mark.parametrize("n", [0, 5, 7, 22])
def test_unsupported_sizes_rejected(n):
    with pytest.raises(ValueError):
        QMLReuseEngine(n)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        QMLReuseEngine(6, method="unknown")
