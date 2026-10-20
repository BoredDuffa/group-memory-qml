"""Independent tests for occurrence and Hessian shift formulas."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from group_memory_qml.experiments.derivatives import ShiftEngine, run


def oracle(engine, X, theta):
    outputs = []
    for vector in X:
        binding = dict(zip(engine.inputs, vector))
        binding.update(zip(engine.weights, theta))
        state = Statevector.from_instruction(engine.source.assign_parameters(binding))
        outputs.append(float(np.real(state.expectation_value(engine.observable))))
    return np.asarray(outputs)


@pytest.mark.parametrize("broken", [False, True])
def test_occurrence_gradients_and_tied_chain_rule(broken):
    engine = ShiftEngine("spin", broken_symmetry=broken)
    X = np.array([[0.43], [1.57]])
    theta = np.full(8, 0.61)
    values, gradient, _ = engine.derivatives(X, theta)
    np.testing.assert_allclose(values, oracle(engine, X, theta), rtol=0, atol=1e-10)
    step = 1e-6
    for j in range(8):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        finite = (oracle(engine, X, plus) - oracle(engine, X, minus)) / (2 * step)
        np.testing.assert_allclose(gradient[:, j], finite, rtol=0, atol=2e-8)
    tied = (oracle(engine, X, theta + step) - oracle(engine, X, theta - step)) / (2 * step)
    np.testing.assert_allclose(gradient.sum(axis=1), tied, rtol=0, atol=2e-8)
    assert engine.last_spatial_reuse == (not broken)
    assert engine.evaluations == (34 if broken else 6)


def test_hessian_matches_independent_finite_differences():
    engine = ShiftEngine("curvature")
    X = np.array([[0.13, -0.27, 0.42, 0.19, -0.33, 0.51]])
    theta = np.array([0.4, 0.3, -0.2, 0.1])
    values, gradient, hessian = engine.derivatives(X, theta, hessian=True)
    base = oracle(engine, X, theta)
    np.testing.assert_allclose(values, base, rtol=0, atol=1e-10)
    epsilon = 2e-4
    for j in range(4):
        a = np.eye(4)[j] * epsilon
        plus, minus = oracle(engine, X, theta + a), oracle(engine, X, theta - a)
        np.testing.assert_allclose(gradient[:, j], (plus - minus) / (2 * epsilon),
                                   rtol=0, atol=2e-8)
        np.testing.assert_allclose(hessian[:, j, j], (plus + minus - 2 * base) / epsilon**2,
                                   rtol=0, atol=2e-7)
        for k in range(j + 1, 4):
            b = np.eye(4)[k] * epsilon
            mixed = (
                oracle(engine, X, theta + a + b) - oracle(engine, X, theta + a - b)
                - oracle(engine, X, theta - a + b) + oracle(engine, X, theta - a - b)
            ) / (4 * epsilon**2)
            np.testing.assert_allclose(hessian[:, j, k], mixed, rtol=0, atol=2e-7)
    assert engine.evaluations == 37


@pytest.mark.parametrize("family", ["spin", "curvature"])
def test_quantum_controls_match_full_derivatives(family):
    X = np.array([[0.39]]) if family == "spin" else np.linspace(-0.5, 0.6, 6)[None, :]
    theta = np.full(8, 0.53) if family == "spin" else np.array([0.4, 0.3, -0.2, 0.1])
    engines = {
        method: ShiftEngine(family, method)
        for method in ("group_memory", "qiskit_symmetry", "qiskit_exact")
    }
    results = {
        method: engine.derivatives(X, theta, hessian=family == "curvature")
        for method, engine in engines.items()
    }
    for result in results.values():
        for actual, expected in zip(result, results["group_memory"]):
            if actual is not None:
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert engines["group_memory"].evaluations == engines["qiskit_symmetry"].evaluations
    assert engines["group_memory"].evaluations < engines["qiskit_exact"].evaluations


def test_one_ulp_unties_weights_and_invalidates_cache():
    engine = ShiftEngine("spin")
    X = np.array([[0.43]])
    theta = np.full(8, 0.61)
    engine.derivatives(X, theta)
    before = engine.evaluations
    engine.derivatives(X, theta)
    assert engine.evaluations == before
    changed = theta.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    values, gradient, _ = engine.derivatives(X, changed)
    assert engine.invalidations == 1
    assert not engine.last_spatial_reuse
    assert engine.evaluations == before + 17
    np.testing.assert_allclose(values, oracle(engine, X, changed), rtol=0, atol=1e-10)


def test_periodic_keys_are_exact_metadata_not_rounded_parameters():
    engine = ShiftEngine("curvature")
    X = np.zeros((2, 6))
    theta = np.array([0.4, 0.3, -0.2, 0.1])
    shifts = np.array([[2, 0, 0, 0], [-2, 0, 0, 0]])
    values = engine.evaluate(X, theta, shifts)
    assert engine.evaluations == 1
    np.testing.assert_allclose(values, oracle(engine, X, theta + np.pi * np.eye(4)[0]),
                               rtol=0, atol=1e-10)
    changed = theta.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    engine.evaluate(X[:1], changed, shifts[:1])
    assert engine.evaluations == 2
    assert engine.invalidations == 1


@pytest.mark.parametrize("experiment", ["2A", "2B"])
def test_small_training_runs_preserve_quantum_trajectories(experiment):
    config = {"steps": 1, "n_train": 4, "n_test": 4}
    outputs = [
        run(experiment, 41, method, config)
        for method in ("group_memory", "qiskit_symmetry", "qiskit_exact")
    ]
    for output in outputs[1:]:
        np.testing.assert_allclose(
            output["validation_vector"], outputs[0]["validation_vector"],
            rtol=0, atol=1e-9,
        )
    for output in outputs:
        assert np.isfinite(output["metric_value"])
        assert output["total_cx"] == output["evaluations"] * output["cx_per_evaluation"]
        assert output["diagnostics"]["final_train_loss"] <= (
            output["diagnostics"]["initial_train_loss"] + 1e-10
        )
