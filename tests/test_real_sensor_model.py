"""Independent full-product-circuit tests for the factorized sensor model."""

import os
import subprocess
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from group_memory_qml.deep_response import _circuit_signature
from group_memory_qml.real_sensor_model import PatchResponseModel


METHODS = (
    "group_spectral", "group_grid", "qiskit_response",
    "qiskit_shift", "classical_adjoint",
)
RESPONSE_METHODS = ("group_spectral", "group_grid", "qiskit_response")


def independent_product(vector, theta, occurrences=2, shifted=None, shift=0.):
    """No engine circuits, matrices or coefficient arrays enter this oracle."""
    circuit = QuantumCircuit(len(vector))
    for patch in range(len(vector)//2):
        offset = 2*patch
        for site in range(2):
            circuit.ry(float(vector[offset+site]), offset+site)
        for layer in range(2*occurrences):
            angle = theta[offset+layer % 2]
            if shifted == (patch, layer):
                angle += shift
            circuit.ry(float(angle), offset+layer % 2)
            for site in range(2):
                circuit.rx(.23+.03*(layer % 2), offset+site)
                circuit.rz(.17-.02*(layer % 2), offset+site)
            circuit.cz(offset, offset+1)
    return circuit


def patch_z(state):
    probabilities = abs(np.asarray(state))**2
    n_qubits = int(np.log2(len(probabilities)))
    indices = np.arange(len(probabilities), dtype=np.uint64)
    local = np.array([
        probabilities @ (1.-2.*((indices >> site) & 1))
        for site in range(n_qubits)
    ])
    return local.reshape(-1, 2).mean(axis=1)


def product_oracle(vectors, theta, occurrences=2, shifted=None, shift=0.):
    return np.stack([
        patch_z(Statevector.from_instruction(
            independent_product(row, theta, occurrences, shifted, shift)
        ).data) for row in vectors
    ])


def occurrence_shift_oracle(vectors, theta, occurrences=2):
    gradient = np.zeros((len(vectors), len(theta)//2, 2))
    for patch in range(len(theta)//2):
        for layer in range(2*occurrences):
            plus = product_oracle(vectors, theta, occurrences, (patch, layer), np.pi/2)
            minus = product_oracle(vectors, theta, occurrences, (patch, layer), -np.pi/2)
            gradient[:, patch, layer % 2] += (plus[:, patch]-minus[:, patch])/2
    return gradient


@pytest.mark.parametrize("method", METHODS)
def test_full_product_features_and_occurrence_summed_jacobian(method):
    model = PatchResponseModel(4, method)
    vectors = np.random.default_rng(69042).normal(size=(2, 4))
    theta = np.array([.31, -.72, .83, .29])
    actual, jacobian = model.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(actual, product_oracle(vectors, theta), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        jacobian, occurrence_shift_oracle(vectors, theta), atol=1e-10, rtol=0
    )
    # Independently perturb all four distinct parameters, including cross-patch zeros.
    for parameter in range(4):
        delta = np.zeros(4)
        delta[parameter] = 1e-6
        finite_difference = (
            product_oracle(vectors, theta+delta)-product_oracle(vectors, theta-delta)
        )/(2e-6)
        expected = np.zeros((2, 2))
        expected[:, parameter//2] = jacobian[:, parameter//2, parameter % 2]
        np.testing.assert_allclose(finite_difference, expected, atol=2e-8, rtol=0)


@pytest.mark.parametrize("occurrences", [1, 2])
def test_full_complex_product_operator_and_source_fingerprint(occurrences):
    model = PatchResponseModel(4, "group_spectral", occurrences=occurrences)
    vector = np.array([.12, -.63, .77, -.41])
    theta = np.array([-.31, .42, .19, .83])
    bindings = dict(zip(model.inputs, vector))
    bindings.update(zip(model.weights, theta))
    actual = Operator(model.source.assign_parameters(bindings)).data
    expected = Operator(independent_product(vector, theta, occurrences)).data
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    assert _circuit_signature(model.source) == model.diagnostics["source_sha256"]


@pytest.mark.parametrize("method", RESPONSE_METHODS)
def test_identical_local_input_reuses_response_with_independent_patch_weights(method):
    model = PatchResponseModel(4, method)
    vectors = np.array([[.24, -.68, .24, -.68]])
    theta = np.array([.43, -.26, -.81, .15])
    actual, _ = model.evaluate(vectors, theta, gradient=True)
    assert abs(actual[0, 0]-actual[0, 1]) > 1e-3
    first = model.diagnostics
    assert first["cached_responses"] == 1
    assert first["check_evaluations"] == 2
    updated = theta+np.array([.11, -.32, -.18, .29])
    values, gradient = model.evaluate(vectors, updated, gradient=True)
    after = model.diagnostics
    assert after["total_circuit_evaluations"] == first["total_circuit_evaluations"]
    assert after["cached_responses"] == 1
    np.testing.assert_allclose(values, product_oracle(vectors, updated), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gradient, occurrence_shift_oracle(vectors, updated), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("method", RESPONSE_METHODS)
def test_forward_only_inputs_promote_to_response_only_for_gradients(method):
    model = PatchResponseModel(4, method)
    vectors = np.array([[.13, -.21, .87, -.52], [.64, -.26, .37, .92]])
    theta = np.array([.21, -.48, -.34, .19])
    forward, none = model.evaluate(vectors, theta)
    assert none is None and model.diagnostics["cached_responses"] == 0
    learned, jacobian = model.evaluate(vectors, theta, gradient=True)
    assert model.diagnostics["cached_responses"] == 4
    assert model.diagnostics["check_evaluations"] == 8
    np.testing.assert_allclose(forward, learned, atol=1e-10, rtol=0)
    assert jacobian.shape == (2, 2, 2)


def test_exact_input_keys_and_copied_predictions_and_sources():
    model = PatchResponseModel(4)
    vectors = np.array([[.13, -.21, .87, -.52]])
    theta = np.array([.21, -.48, -.34, .19])
    expected, expected_gradient = model.evaluate(vectors, theta, gradient=True)
    values, gradient = model.evaluate(vectors, theta, gradient=True)
    values[:] = -100
    gradient[:] = -100
    source = model.source
    source.x(0)
    diagnostics = model.diagnostics
    diagnostics["source_sha256"] = "changed"
    actual, actual_gradient = model.evaluate(vectors, theta, gradient=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_gradient, expected_gradient)
    assert _circuit_signature(model.source) == model.diagnostics["source_sha256"]
    slightly_different = vectors.copy()
    slightly_different[0, 0] = np.nextafter(vectors[0, 0], np.inf)
    model.evaluate(slightly_different, theta, gradient=True)
    assert model.diagnostics["cached_responses"] == 3


def test_sixteen_independent_parameters_match_direct_classical_simulation():
    vectors = np.random.default_rng(9348).normal(size=(2, 16))
    theta = np.linspace(-.73, .49, 16)
    response = PatchResponseModel(16, "group_spectral")
    direct = PatchResponseModel(16, "classical_adjoint")
    result = response.evaluate(vectors, theta, gradient=True)
    expected = direct.evaluate(vectors, theta, gradient=True)
    for actual, wanted in zip(result, expected):
        np.testing.assert_allclose(actual, wanted, atol=1e-10, rtol=0)
    assert result[0].shape == (2, 8) and result[1].shape == (2, 8, 2)
    details = response.diagnostics
    assert details["n_qubits"] == details["n_parameters"] == 16
    assert details["patch_qubits"] == details["patch_parameters"] == 2
    assert details["response_grid_points_per_patch_input"] == 25
    assert details["notional_unfactorized_joint_grid_points"] == 5**16
    assert details["cached_responses"] == 16
    assert "no entanglement between patches" in details["entanglement_scope"]


def test_method_signatures_distinguish_global_model_and_patch_native():
    models = [PatchResponseModel(8, method) for method in METHODS]
    details = [model.diagnostics for model in models]
    for key in ("source_sha256", "architecture_sha256", "patch_source_sha256"):
        assert len({item[key] for item in details}) == 1
    assert details[0]["source_sha256"] != details[0]["patch_source_sha256"]
    for model, item in zip(models, details):
        assert _circuit_signature(model.source) == item["source_sha256"]
        assert item["patch_native_sha256"] == item["native_sha256"]
        assert "two-qubit" in item["native_signature_scope"]


@pytest.mark.parametrize("method", METHODS)
def test_empty_batch(method):
    model = PatchResponseModel(8, method)
    result, gradient = model.evaluate(np.empty((0, 8)), np.zeros(8), gradient=True)
    assert result.shape == (0, 4) and gradient.shape == (0, 4, 2)
    assert model.diagnostics["evaluated_examples"] == 0


@pytest.mark.parametrize("invalid", [0, 1, 3, 7, 17, 18, True, 4.0, "4"])
def test_invalid_parameter_count(invalid):
    with pytest.raises(ValueError):
        PatchResponseModel(invalid)


@pytest.mark.parametrize("invalid", [0, 9, False, 2.0, "2"])
def test_invalid_occurrences(invalid):
    with pytest.raises(ValueError):
        PatchResponseModel(4, occurrences=invalid)


@pytest.mark.parametrize("X,theta,gradient", [
    (np.zeros(4), np.zeros(4), False),
    (np.zeros((1, 3)), np.zeros(4), False),
    (np.zeros((1, 4)), np.zeros(3), False),
    (np.zeros((1, 4)), np.zeros((1, 4)), False),
    (np.full((1, 4), np.nan), np.zeros(4), False),
    (np.zeros((1, 4)), np.full(4, np.inf), False),
    (np.zeros((1, 4), dtype=complex), np.zeros(4), False),
    (np.zeros((1, 4)), np.zeros(4, dtype=complex), False),
    (np.zeros((1, 4)), np.zeros(4), 1),
])
def test_invalid_evaluation_input(X, theta, gradient):
    with pytest.raises(ValueError):
        PatchResponseModel(4).evaluate(X, theta, gradient)


def test_classical_path_has_no_implicit_qiskit_import():
    script = """
import sys
import numpy as np
from group_memory_qml.real_sensor_model import PatchResponseModel
model = PatchResponseModel(8, "classical_adjoint")
model.evaluate(np.ones((2,8)), np.arange(8)/10, gradient=True)
assert not any(name == "qiskit" or name.startswith("qiskit.") or
               name == "qiskit_aer" or name.startswith("qiskit_aer.")
               for name in sys.modules)
assert model.diagnostics["native_evaluations"] == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=os.environ.copy(),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
