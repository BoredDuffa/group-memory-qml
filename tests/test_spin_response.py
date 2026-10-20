"""Independent full-state checks for persistent spin response coefficients."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from group_memory_qml.experiments.derivatives import ShiftEngine
from group_memory_qml.spin_response import AnalyticSpinEngine, SpinResponseEngine


def oracle(engine, X, theta):
    results = []
    for row in X:
        binding = dict(zip(engine.inputs, row))
        binding.update(zip(engine.weights, theta))
        state = Statevector.from_instruction(engine.source.assign_parameters(binding))
        results.append(float(np.real(state.expectation_value(engine.observable))))
    return np.asarray(results)


@pytest.mark.parametrize("method", ["group_memory_response", "qiskit_response"])
def test_response_retains_coefficients_across_weight_updates(method):
    engine = SpinResponseEngine(method)
    X = np.array([[0.43], [1.57], [0.43]])
    for angle in (0.61, -0.3, 1.9):
        theta = np.full(8, angle)
        actual, gradient, curvature = engine.derivatives(X, theta)
        np.testing.assert_allclose(actual, oracle(engine, X, theta), rtol=0, atol=1e-10)
        assert curvature is None
        step = 1e-6
        tied = (
            oracle(engine, X, theta + step) - oracle(engine, X, theta - step)
        ) / (2 * step)
        np.testing.assert_allclose(gradient.sum(axis=1), tied, rtol=0, atol=2e-8)
    assert engine.evaluations == 6
    assert engine.invalidations == 2
    assert engine.diagnostics["probe_evaluations"] == 4
    assert engine.diagnostics["verification_evaluations"] == 2
    assert engine.diagnostics["cached_input_responses"] == 2
    assert engine.diagnostics["native_cx"] == 6 * engine.cx_per_evaluation
    assert engine.diagnostics["max_response_verification_error"] < 1e-10


def test_individual_occurrence_gradient_is_not_tied_two_query_rule():
    engine = SpinResponseEngine()
    X = np.array([[0.23], [1.61]])
    theta = np.full(8, 0.72)
    _, gradient, _ = engine.derivatives(X, theta)
    step = 1e-6
    for j in range(8):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        finite = (oracle(engine, X, plus) - oracle(engine, X, minus)) / (2 * step)
        np.testing.assert_allclose(gradient[:, j], finite, rtol=0, atol=2e-8)
    reference = ShiftEngine("spin", "qiskit_symmetry")
    _, reference_gradient, _ = reference.derivatives(X, theta)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=0, atol=1e-10)


def test_forward_only_unseen_inputs_do_not_pay_three_probe_cost():
    engine = SpinResponseEngine()
    X = np.array([[0.43], [0.43], [1.57]])
    theta = np.full(8, 0.61)
    actual = engine.evaluate(X, theta)
    np.testing.assert_allclose(actual, oracle(engine, X, theta), rtol=0, atol=1e-10)
    assert engine.evaluations == 2
    assert engine.diagnostics["probe_evaluations"] == 0
    assert engine.diagnostics["cached_input_responses"] == 0
    engine.evaluate(X, theta)
    assert engine.evaluations == 2


@pytest.mark.parametrize("broken", [False, True])
def test_untied_and_broken_weights_fall_back(broken):
    engine = SpinResponseEngine(broken_symmetry=broken)
    X = np.array([[0.43]])
    theta = np.full(8, 0.61)
    if not broken:
        engine.derivatives(X, theta)
        theta[0] = np.nextafter(theta[0], np.inf)
    before = engine.evaluations
    actual, gradient, _ = engine.derivatives(X, theta)
    reference = ShiftEngine("spin", "qiskit_exact", broken_symmetry=broken)
    expected, expected_gradient, _ = reference.derivatives(X, theta)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-10)
    assert not engine.last_spatial_reuse
    assert engine.evaluations == before + 17
    assert engine.diagnostics["fallback_calls"] == 1


def test_negative_zero_disables_tied_reuse():
    engine = SpinResponseEngine()
    theta = np.zeros(8)
    theta[0] = -0.0
    engine.derivatives(np.array([[0.43]]), theta)
    assert not engine.last_spatial_reuse
    assert engine.diagnostics["cached_input_responses"] == 0


def test_input_mutation_does_not_change_saved_response_keys():
    engine = SpinResponseEngine()
    X = np.array([[0.43]])
    theta = np.full(8, 0.61)
    old = engine.derivatives(X, theta)[0]
    X[0, 0] = 1.57
    new = engine.derivatives(X, theta)[0]
    assert engine.diagnostics["cached_input_responses"] == 2
    assert engine.evaluations == 6
    assert abs(old[0] - new[0]) > 0.1
    recovered = engine.evaluate(np.array([[0.43]]), theta)
    np.testing.assert_allclose(recovered, old, rtol=0, atol=1e-10)


def test_arbitrary_integer_shifts_use_local_response():
    engine = SpinResponseEngine()
    X = np.array([[0.43], [1.57]])
    theta = np.full(8, 0.61)
    engine.derivatives(X, theta)
    shifts = np.array([[1, -1, 0, 2, -2, 3, 4, -4], [-16, 16, 2, 1, 3, -2, 1, 0]])
    actual = engine.evaluate(X, theta, shifts)
    expected = np.array([
        oracle(engine, X[index:index + 1], theta + row * np.pi / 2)[0]
        for index, row in enumerate(shifts)
    ])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert engine.evaluations == 6


def test_off_grid_failure_rejects_response_and_keeps_checks(monkeypatch):
    engine = SpinResponseEngine()
    X = np.array([[0.43], [1.57]])
    theta = np.full(8, 0.61)
    original = engine._direct.evaluate

    def damaged_check(vectors, angles, shifts=None):
        values = original(vectors, angles, shifts)
        if np.all(angles == 0.431):
            values[0] += 1e-6
        return values

    monkeypatch.setattr(engine._direct, "evaluate", damaged_check)
    actual, gradient, _ = engine.derivatives(X, theta)
    reference = ShiftEngine("spin", "qiskit_symmetry")
    expected, expected_gradient, _ = reference.derivatives(X, theta)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-10)
    assert engine.diagnostics["verification_failures"] == 1
    assert engine.diagnostics["cached_input_responses"] == 1
    assert engine.diagnostics["rejected_input_responses"] == 1
    assert engine.diagnostics["fallback_calls"] == 1
    probes_before = engine.diagnostics["probe_evaluations"]
    engine.derivatives(X, theta + 0.1)
    assert engine.diagnostics["probe_evaluations"] == probes_before
    assert engine.diagnostics["fallback_calls"] == 2


def test_hessian_request_falls_back_without_changing_semantics():
    engine = SpinResponseEngine()
    reference = ShiftEngine("spin", "qiskit_symmetry")
    X = np.array([[0.43]])
    theta = np.full(8, 0.61)
    actual = engine.derivatives(X, theta, hessian=True)
    expected = reference.derivatives(X, theta, hessian=True)
    for result, target in zip(actual, expected):
        np.testing.assert_allclose(result, target, rtol=0, atol=1e-10)
    assert engine.diagnostics["probe_evaluations"] == 0
    assert engine.evaluations == reference.evaluations


def test_both_planners_execute_identical_native_work():
    engines = [SpinResponseEngine(method) for method in
               ("group_memory_response", "qiskit_response")]
    X = np.array([[1.57], [0.43], [1.57], [0.43]])
    for angle in (0.6, 0.7, -0.2):
        theta = np.full(8, angle)
        results = [engine.derivatives(X, theta) for engine in engines]
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=1e-10)
    assert engines[0].diagnostics == engines[1].diagnostics


@pytest.mark.parametrize("broken", [False, True])
def test_classical_same_model_formula_and_derivatives(broken):
    analytic = AnalyticSpinEngine(broken_symmetry=broken)
    quantum = ShiftEngine("spin", "qiskit_exact", broken_symmetry=broken)
    X = np.array([[-0.8], [0.0], [0.43], [1.57], [2.4]])
    theta = np.linspace(-0.7, 0.9, 8)
    actual, gradients, hessian = analytic.derivatives(X, theta, hessian=True)
    np.testing.assert_allclose(actual, oracle(quantum, X, theta), rtol=0, atol=1e-10)
    step = 1e-6
    for j in range(8):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        finite = (oracle(quantum, X, plus) - oracle(quantum, X, minus)) / (2 * step)
        np.testing.assert_allclose(gradients[:, j], finite, rtol=0, atol=2e-8)
        finite_gradient = (
            analytic.derivatives(X, plus)[1] - analytic.derivatives(X, minus)[1]
        ) / (2 * step)
        np.testing.assert_allclose(hessian[:, :, j], finite_gradient, rtol=0, atol=2e-8)
    assert analytic.evaluations == 0


@pytest.mark.parametrize("factory", [SpinResponseEngine, AnalyticSpinEngine])
@pytest.mark.parametrize("bad", ["shape", "nonfinite", "bool", "shift_type", "shift_bound"])
def test_invalid_inputs_fail_closed(factory, bad):
    engine = factory()
    X, theta, shifts = np.array([[0.43]]), np.full(8, 0.61), np.zeros((1, 8), dtype=int)
    if bad == "shape":
        X = np.ones((1, 2))
    elif bad == "nonfinite":
        theta[0] = np.nan
    elif bad == "bool":
        X = np.array([[True]])
    elif bad == "shift_type":
        shifts = shifts.astype(float)
    else:
        shifts[0, 0] = 17
    with pytest.raises(ValueError):
        engine.evaluate(X, theta, shifts)
