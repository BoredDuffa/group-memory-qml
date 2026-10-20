"""Independent circuit and vector-derivative checks for multivariate response reuse."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from group_memory_qml.multi_response import MultiResponseEngine, ResponseResourceError


METHODS = ("group_spectral", "group_grid", "qiskit_response", "qiskit_shift", "classical_adjoint")
RESPONSE_METHODS = ("group_spectral", "group_grid", "qiskit_response")


def independent_circuit(qubits, parameters, occurrences, vector, theta, shifted_layer=None, shift=0):
    """Rebuild the declared gates without using an engine circuit or matrix."""
    circuit = QuantumCircuit(qubits)
    for site, value in enumerate(vector):
        circuit.ry(float(value), site)
    for layer in range(parameters * occurrences):
        angle = theta[layer % parameters] + (shift if layer == shifted_layer else 0)
        circuit.ry(float(angle), layer % qubits)
        kind = layer % 2
        for site in range(qubits):
            circuit.rx(0.23 + 0.03 * kind, site)
            circuit.rz(0.17 - 0.02 * kind, site)
        visited = set()
        for site in range(qubits):
            edge = frozenset((site, (site + 1) % qubits))
            if edge not in visited:
                circuit.cz(site, (site + 1) % qubits)
                visited.add(edge)
    return circuit


def mean_z(state):
    probabilities = np.abs(np.asarray(state)) ** 2
    qubits = int(np.log2(len(probabilities)))
    indices = np.arange(len(probabilities), dtype=np.uint64)
    return float(
        np.mean(
            [
                np.dot(probabilities, 1.0 - 2.0 * ((indices >> site) & 1))
                for site in range(qubits)
            ]
        )
    )


def oracle(qubits, parameters, occurrences, vectors, theta):
    return np.asarray(
        [
            mean_z(
                Statevector.from_instruction(
                    independent_circuit(qubits, parameters, occurrences, row, theta)
                ).data
            )
            for row in vectors
        ]
    )


def shift_oracle(qubits, parameters, occurrences, vectors, theta):
    """A shared parameter's derivative sums shifts of its individual occurrences."""
    gradients = np.zeros((len(vectors), parameters))
    for index, row in enumerate(vectors):
        for layer in range(parameters * occurrences):
            plus = Statevector.from_instruction(
                independent_circuit(
                    qubits, parameters, occurrences, row, theta, layer, np.pi / 2
                )
            )
            minus = Statevector.from_instruction(
                independent_circuit(
                    qubits, parameters, occurrences, row, theta, layer, -np.pi / 2
                )
            )
            gradients[index, layer % parameters] += 0.5 * (
                mean_z(plus.data) - mean_z(minus.data)
            )
    return gradients


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("parameters,occurrences", [(2, 1), (2, 2), (3, 1)])
def test_independent_angles_and_shared_occurrence_derivatives(method, parameters, occurrences):
    qubits = 6
    engine = MultiResponseEngine(qubits, parameters, occurrences, method)
    vectors = np.random.default_rng(9301).normal(0, 0.7, (2, qubits))
    for theta in (
        np.linspace(-0.43, 0.71, parameters),
        np.linspace(0.17, -0.92, parameters),
        np.zeros(parameters),
    ):
        values, gradient = engine.evaluate(vectors, theta, gradient=True)
        assert values.shape == (2,) and gradient.shape == (2, parameters)
        np.testing.assert_allclose(
            values, oracle(qubits, parameters, occurrences, vectors, theta), atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            gradient,
            shift_oracle(qubits, parameters, occurrences, vectors, theta),
            atol=1e-10,
            rtol=0,
        )


def test_each_parameter_gradient_and_joint_direction_are_distinct():
    engine = MultiResponseEngine(6, 2, 3, "group_spectral")
    vectors = np.random.default_rng(524).normal(size=(2, 6))
    theta = np.array([0.41, -0.72])
    _, gradient = engine.evaluate(vectors, theta, gradient=True)
    assert np.max(np.abs(gradient[:, 0] - gradient[:, 1])) > 1e-3
    invalid_shared_shift = np.zeros_like(gradient)
    step = 1e-6
    for parameter in range(2):
        delta = np.zeros(2)
        delta[parameter] = np.pi / 2
        invalid_shared_shift[:, parameter] = 0.5 * (
            oracle(6, 2, 3, vectors, theta + delta)
            - oracle(6, 2, 3, vectors, theta - delta)
        )
        delta[parameter] = step
        finite_difference = (
            oracle(6, 2, 3, vectors, theta + delta)
            - oracle(6, 2, 3, vectors, theta - delta)
        ) / (2 * step)
        np.testing.assert_allclose(gradient[:, parameter], finite_difference, atol=2e-8, rtol=0)
    assert np.max(np.abs(gradient - invalid_shared_shift)) > 1e-3
    joint = (
        oracle(6, 2, 3, vectors, theta + step)
        - oracle(6, 2, 3, vectors, theta - step)
    ) / (2 * step)
    np.testing.assert_allclose(gradient.sum(axis=1), joint, atol=2e-8, rtol=0)


def test_fixture_has_mixed_sign_fourier_terms_that_cannot_be_discarded():
    vector = np.array([0.32, -0.71, 0.84, 0.21])
    grid = 2 * np.pi * np.arange(5) / 5
    sampled = np.array(
        [
            [oracle(4, 2, 2, vector[None, :], np.array([left, right]))[0] for right in grid]
            for left in grid
        ]
    )
    coefficients = np.fft.fft2(sampled) / 25
    # Indices (1,4) and (4,1) represent frequencies (+1,-1) and (-1,+1).
    assert max(abs(coefficients[1, 4]), abs(coefficients[4, 1])) > 1e-5
    engine = MultiResponseEngine(4, 2, 2, "group_spectral")
    theta = np.array([0.413, -0.782])
    values, gradients = engine.evaluate(vector[None, :], theta, gradient=True)
    np.testing.assert_allclose(
        values, oracle(4, 2, 2, vector[None, :], theta), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        gradients, shift_oracle(4, 2, 2, vector[None, :], theta), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("parameters,occurrences", [(2, 1), (2, 2), (3, 1)])
def test_multivariate_state_coefficients_preserve_full_phase(parameters, occurrences):
    qubits = 4
    engine = MultiResponseEngine(qubits, parameters, occurrences, "group_spectral")
    vectors = np.random.default_rng(4231).normal(size=(2, qubits))
    coefficients = engine._spectral_states(vectors)
    exponents = engine.state_exponents
    assert coefficients.shape == (2, (occurrences + 1) ** parameters, 2**qubits)
    assert exponents.shape == ((occurrences + 1) ** parameters, parameters)
    for theta in (
        np.linspace(-0.41, 0.73, parameters),
        np.zeros(parameters),
        np.full(parameters, np.pi),
    ):
        actual = np.einsum("bkd,k->bd", coefficients, np.exp(1j * (exponents @ theta)))
        expected = np.stack(
            [
                Statevector.from_instruction(
                    independent_circuit(qubits, parameters, occurrences, row, theta)
                ).data
                for row in vectors
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    theta = np.linspace(-0.41, 0.73, parameters)
    before = np.einsum("bkd,k->bd", coefficients, np.exp(1j * (exponents @ theta)))
    for parameter in range(parameters):
        shifted = theta.copy()
        shifted[parameter] += 2 * np.pi
        after = np.einsum("bkd,k->bd", coefficients, np.exp(1j * (exponents @ shifted)))
        np.testing.assert_allclose(after, (-1) ** occurrences * before, atol=1e-10, rtol=0)


def test_source_and_native_match_full_operator():
    engine = MultiResponseEngine(4, 2, 2, "group_spectral")
    vector = np.array([0.32, -0.71, 0.84, 0.21])
    theta = np.array([-0.31, 0.47])
    binding = dict(zip(engine.inputs, vector))
    binding.update(zip(engine.weights, theta))
    expected = Operator(independent_circuit(4, 2, 2, vector, theta)).data
    for circuit in (engine.source, engine.native):
        np.testing.assert_allclose(
            Operator(circuit.assign_parameters(binding)).data, expected, atol=1e-10, rtol=0
        )


def test_occurrence_shift_native_preserves_independent_parameter_assignment():
    engine = MultiResponseEngine(4, 2, 2, "qiskit_shift")
    vector = np.array([0.32, -0.71, 0.84, 0.21])
    theta = np.array([-0.31, 0.47])
    expected = Operator(independent_circuit(4, 2, 2, vector, theta)).data
    source_binding = dict(zip(engine.inputs, vector))
    source_binding.update(zip(engine.weights, theta))
    native_binding = dict(zip(engine.inputs, vector))
    native_binding.update(
        (parameter, theta[layer % 2]) for layer, parameter in enumerate(engine.occurrence_weights)
    )
    np.testing.assert_allclose(
        Operator(engine.source.assign_parameters(source_binding)).data, expected, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        Operator(engine.native.assign_parameters(native_binding)).data, expected, atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("method", METHODS)
def test_twelve_qubit_smoke_against_independent_statevector(method):
    engine = MultiResponseEngine(12, 2, 1, method)
    vectors = np.random.default_rng(342).normal(0, 0.6, (1, 12))
    theta = np.array([0.41, -0.83])
    values, gradients = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(values, oracle(12, 2, 1, vectors, theta), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gradients, shift_oracle(12, 2, 1, vectors, theta), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("method", METHODS)
def test_mutated_inputs_outputs_and_parameters_do_not_poison_results(method):
    engine = MultiResponseEngine(6, 2, 2, method)
    vectors = np.random.default_rng(541).normal(size=(2, 6))
    original = vectors.copy()
    theta = np.array([0.31, -0.49])
    values, gradient = engine.evaluate(vectors, theta, gradient=True)
    saved_values, saved_gradient = values.copy(), gradient.copy()
    values[:] = 999
    gradient[:] = -999
    vectors[0, 0] += 0.2
    theta[1] += 0.4
    current_values, current_gradient = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(
        current_values, oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        current_gradient, shift_oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0
    )
    recovered_values, recovered_gradient = engine.evaluate(
        original[[1, 0, 1]], np.array([0.31, -0.49]), gradient=True
    )
    np.testing.assert_allclose(recovered_values, saved_values[[1, 0, 1]], atol=1e-10, rtol=0)
    np.testing.assert_allclose(recovered_gradient, saved_gradient[[1, 0, 1]], atol=1e-10, rtol=0)


@pytest.mark.parametrize("method", RESPONSE_METHODS)
def test_exact_input_keys_do_not_assume_spatial_symmetry(method):
    engine = MultiResponseEngine(6, 2, 2, method)
    vector = np.array([0.17, -0.83, 1.31, 0.42, -0.37, 0.65])
    neighbor = vector.copy()
    neighbor[0] = np.nextafter(neighbor[0], np.inf)
    vectors = np.stack([vector, neighbor, np.roll(vector, 1), vector])
    theta = np.array([0.43, -0.21])
    values, _ = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(values, oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0)
    assert abs(values[0] - values[2]) > 1e-4
    before = engine.diagnostics
    assert before["cached_responses"] == 3
    assert before["check_evaluations"] == 6
    assert before["response_grid_size"] == 25
    assert before["response_state_support_size"] == 9
    for weights in (np.array([0.47, -0.52]), np.array([-0.31, 0.92])):
        values, gradients = engine.evaluate(vectors[::-1], weights, gradient=True)
        np.testing.assert_allclose(
            values, oracle(6, 2, 2, vectors[::-1], weights), atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            gradients, shift_oracle(6, 2, 2, vectors[::-1], weights), atol=1e-10, rtol=0
        )
    after = engine.diagnostics
    for key in (
        "cached_responses", "total_circuit_evaluations", "native_evaluations",
        "check_evaluations", "coefficient_bytes",
    ):
        assert after[key] == before[key]


@pytest.mark.parametrize("method", RESPONSE_METHODS)
def test_off_grid_error_above_unchanged_tolerance_is_transactional(method, monkeypatch):
    engine = MultiResponseEngine(4, 2, 2, method)
    vectors = np.random.default_rng(982).normal(size=(2, 4))
    original = engine._native_values

    def damaged_check(inputs, angles):
        result = original(inputs, angles)
        selected = np.asarray(angles)[:, 0] == -0.583
        result[selected] += 2e-10
        return result

    monkeypatch.setattr(engine, "_native_values", damaged_check)
    with pytest.raises(RuntimeError, match="check|verification|validation"):
        engine.evaluate(vectors, np.array([0.31, -0.72]), gradient=True)
    assert engine.diagnostics["cached_responses"] == 0
    assert engine.diagnostics["check_failures"] > 0
    charged = engine.diagnostics["native_evaluations"]
    assert charged > 0
    monkeypatch.setattr(engine, "_native_values", original)
    values, gradients = engine.evaluate(vectors, np.array([0.31, -0.72]), gradient=True)
    np.testing.assert_allclose(
        values, oracle(4, 2, 2, vectors, np.array([0.31, -0.72])), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        gradients, shift_oracle(4, 2, 2, vectors, np.array([0.31, -0.72])), atol=1e-10, rtol=0
    )
    assert engine.diagnostics["cached_responses"] == 2
    assert engine.diagnostics["native_evaluations"] > charged


@pytest.mark.parametrize("method", RESPONSE_METHODS)
def test_forward_only_unknown_inputs_do_not_calibrate_the_response(method):
    engine = MultiResponseEngine(6, 2, 2, method)
    vectors = np.random.default_rng(821).normal(size=(2, 6))[[0, 1, 0]]
    theta = np.array([0.43, -0.27])
    values, gradient = engine.evaluate(vectors, theta, gradient=False)
    np.testing.assert_allclose(values, oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0)
    assert gradient is None
    before = engine.diagnostics
    assert before["cached_direct_values"] == 2
    assert before["cached_responses"] == before["check_evaluations"] == 0
    engine.evaluate(vectors[::-1], theta, gradient=False)
    assert engine.diagnostics["total_circuit_evaluations"] == before["total_circuit_evaluations"]
    theta[1] = np.nextafter(theta[1], np.inf)
    engine.evaluate(vectors, theta, gradient=False)
    assert engine.diagnostics["cached_responses"] == engine.diagnostics["check_evaluations"] == 0
    if method == "qiskit_response":
        assert engine.diagnostics["total_circuit_evaluations"] == before["total_circuit_evaluations"] + 2
    engine.evaluate(vectors, theta, gradient=True)
    assert engine.diagnostics["cached_responses"] == 2
    assert engine.diagnostics["check_evaluations"] == 4
    charged = engine.diagnostics["total_circuit_evaluations"]
    theta[:] = [-0.63, 0.71]
    updated, _ = engine.evaluate(vectors, theta, gradient=False)
    np.testing.assert_allclose(updated, oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0)
    assert engine.diagnostics["total_circuit_evaluations"] == charged


def test_public_source_and_basis_copies_cannot_change_saved_semantics():
    engine = MultiResponseEngine(4, 2, 2, "group_spectral")
    engine.source.x(0)
    engine.native.x(0)
    for snapshot in (engine.grid, engine.state_exponents):
        try:
            snapshot[...] = 0
        except ValueError:
            assert not snapshot.flags.writeable
    vectors = np.random.default_rng(338).normal(size=(1, 4))
    theta = np.array([0.37, -0.86])
    values, gradients = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(values, oracle(4, 2, 2, vectors, theta), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gradients, shift_oracle(4, 2, 2, vectors, theta), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("method", ("qiskit_response", "qiskit_shift"))
@pytest.mark.parametrize("qubits", (2, 4))
def test_cached_diagonal_preserves_complex_global_phase_and_little_endian_order(method, qubits):
    engine = MultiResponseEngine(qubits, 2, 2, method)
    words = np.arange(2**qubits)
    z_sum = np.sum([1 - 2 * ((words >> site) & 1) for site in range(qubits)], axis=0)
    sign = np.ones(2**qubits)
    visited = set()
    for site in range(qubits):
        other = (site + 1) % qubits
        edge = frozenset((site, other))
        if edge not in visited:
            sign *= 1 - 2 * (((words >> site) & 1) * ((words >> other) & 1))
            visited.add(edge)
    for kind, diagonal in enumerate(engine.diagonals):
        assert diagonal.shape == (2**qubits,)
        expected = sign * np.exp(-0.5j * (0.17 - 0.02 * kind) * z_sum)
        np.testing.assert_allclose(diagonal, expected, atol=1e-10, rtol=0)
        # The |0...0> phase must survive; setting it to one would be incorrect.
        assert abs(diagonal[0] - 1) > 0.1
        np.testing.assert_allclose(np.abs(diagonal), np.ones_like(sign), atol=1e-10, rtol=0)


@pytest.mark.parametrize("method", ("qiskit_response", "qiskit_shift"))
def test_diagonal_execution_circuit_preserves_full_operator_without_dense_instructions(method):
    engine = MultiResponseEngine(4, 2, 2, method)
    vector = np.array([0.37, -0.81, 0.43, 0.69])
    theta = np.array([0.43, -0.72])
    circuit = engine.diagonal_circuit
    bindings = dict(zip(engine.inputs, vector))
    if method == "qiskit_shift":
        bindings.update(
            (parameter, theta[layer % 2])
            for layer, parameter in enumerate(engine.occurrence_weights)
        )
    else:
        bindings.update(zip(engine.weights, theta))
    np.testing.assert_allclose(
        Operator(circuit.assign_parameters(bindings)).data,
        Operator(independent_circuit(4, 2, 2, vector, theta)).data,
        atol=1e-10,
        rtol=0,
    )
    assert circuit.count_ops().get("diagonal", 0) == 4
    assert circuit.count_ops().get("unitary", 0) == 0
    assert circuit.count_ops().get("cx", 0) == 0
    assert engine.diagnostics["dense_block_bytes"] == 0
    for instruction in circuit.data:
        for parameter in instruction.operation.params:
            assert not isinstance(parameter, np.ndarray) or parameter.ndim < 2


@pytest.mark.parametrize("method", ("qiskit_response", "qiskit_shift"))
def test_cached_diagonal_and_circuit_copies_do_not_mutate_execution(method):
    engine = MultiResponseEngine(6, 2, 2, method)
    for diagonal in engine.diagonals:
        try:
            diagonal[:] = 0
        except ValueError:
            assert not diagonal.flags.writeable
    engine.diagonal_circuit.x(0)
    vectors = np.random.default_rng(973).normal(size=(2, 6))
    theta = np.array([0.41, -0.63])
    values, gradients = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(values, oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gradients, shift_oracle(6, 2, 2, vectors, theta), atol=1e-10, rtol=0
    )
    diagnostics = engine.diagnostics
    expected_diagonal = 2 * (25 if method == "qiskit_response" else 1 + 2 * 4)
    expected_native = 4 if method == "qiskit_response" else 0
    assert diagnostics["diagonal_evaluations"] == expected_diagonal
    assert diagnostics["native_evaluations"] == expected_native
    assert diagnostics["total_circuit_evaluations"] == expected_diagonal + expected_native
    assert diagnostics["check_evaluations"] == expected_native
    assert diagnostics["native_cx_work"] == expected_native * (diagnostics["native_cx_per_circuit"] or 0)


@pytest.mark.parametrize("method", RESPONSE_METHODS)
def test_response_grid_guard_fails_before_calibration(method):
    with pytest.raises(ResponseResourceError):
        MultiResponseEngine(4, 3, 2, method, max_grid_points=124)


def test_state_coefficient_pair_guard_is_independent_of_grid_limit():
    # Raising the grid ceiling must not disable the independent contraction guard.
    with pytest.raises(ResponseResourceError):
        MultiResponseEngine(4, 6, 2, "group_spectral", max_grid_points=20000)


@pytest.mark.parametrize("method", METHODS)
def test_working_memory_guard_rejects_impossible_budget(method):
    with pytest.raises(ResponseResourceError):
        MultiResponseEngine(12, 2, 2, method, memory_budget_bytes=1024)


def test_retained_cache_growth_is_bounded_without_changing_existing_values():
    sizing = MultiResponseEngine(4, 2, 1, "group_spectral").diagnostics
    budget = sizing["estimated_base_bytes"] + sizing["estimated_working_bytes_per_input"] + 1000
    engine = MultiResponseEngine(4, 2, 1, "group_spectral", memory_budget_bytes=budget)
    vectors = np.random.default_rng(528).normal(size=(16, 4))
    theta = np.array([0.31, -0.48])
    completed = 0
    first = None
    for row in vectors:
        try:
            values, gradients = engine.evaluate(row[None, :], theta, gradient=True)
        except ResponseResourceError:
            break
        if first is None:
            first = values.copy(), gradients.copy()
        completed += 1
    assert 0 < completed < len(vectors)
    assert engine.diagnostics["cached_responses"] == completed
    assert engine.diagnostics["estimated_peak_workspace_bytes"] <= budget
    recovered = engine.evaluate(vectors[:1], theta, gradient=True)
    for actual, expected in zip(recovered, first):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_failure_in_later_calibration_chunk_rolls_back_the_entire_batch(monkeypatch):
    sizing = MultiResponseEngine(4, 2, 1, "group_spectral").diagnostics
    budget = sizing["estimated_base_bytes"] + sizing["estimated_working_bytes_per_input"] + 2500
    engine = MultiResponseEngine(4, 2, 1, "group_spectral", memory_budget_bytes=budget)
    vectors = np.random.default_rng(927).normal(size=(3, 4))
    original = engine._native_values
    checked_input_counts = []

    def corrupt_last_chunk(inputs, angles):
        checked_input_counts.append(len({row.tobytes() for row in inputs}))
        values = original(inputs, angles)
        selected = (inputs[:, 0] == vectors[-1, 0]) & (np.asarray(angles)[:, 0] == -0.583)
        values[selected] += 2e-10
        return values

    monkeypatch.setattr(engine, "_native_values", corrupt_last_chunk)
    with pytest.raises(RuntimeError, match="check"):
        engine.evaluate(vectors, np.array([0.21, -0.73]), gradient=True)
    assert checked_input_counts == [1, 1, 1]
    assert engine.diagnostics["calibration_chunks"] == 2
    assert engine.diagnostics["check_evaluations"] == 6
    assert engine.diagnostics["cached_responses"] == 0


@pytest.mark.parametrize("method", ("classical_adjoint", "qiskit_shift"))
@pytest.mark.parametrize("parameters", (3, 8))
def test_direct_controls_are_not_subject_to_exponential_response_grid_limit(method, parameters):
    engine = MultiResponseEngine(4, parameters, 2, method, max_grid_points=1)
    vectors = np.random.default_rng(816).normal(size=(1, 4))
    theta = np.linspace(-0.75, 0.92, parameters)
    values, gradients = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(values, oracle(4, parameters, 2, vectors, theta), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gradients, shift_oracle(4, parameters, 2, vectors, theta), atol=1e-10, rtol=0
    )


def test_qiskit_shift_reuses_only_exact_current_weight_results():
    engine = MultiResponseEngine(4, 2, 2, "qiskit_shift")
    vectors = np.random.default_rng(216).normal(size=(2, 4))[[0, 1, 0]]
    theta = np.array([0.31, -0.58])
    values, gradients = engine.evaluate(vectors, theta, gradient=True)
    np.testing.assert_allclose(values, oracle(4, 2, 2, vectors, theta), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        gradients, shift_oracle(4, 2, 2, vectors, theta), atol=1e-10, rtol=0
    )
    assert engine.diagnostics["total_circuit_evaluations"] == 2 * (1 + 2 * 4)
    assert engine.diagnostics["native_evaluations"] == 0
    assert engine.diagnostics["cached_responses"] == 0
    before = engine.diagnostics["total_circuit_evaluations"]
    engine.evaluate(vectors[::-1], theta, gradient=True)
    engine.evaluate(vectors, theta, gradient=False)
    assert engine.diagnostics["total_circuit_evaluations"] == before
    theta[1] = np.nextafter(theta[1], np.inf)
    engine.evaluate(vectors, theta, gradient=True)
    assert engine.diagnostics["total_circuit_evaluations"] == 2 * before
    assert engine.diagnostics["cached_responses"] == 0


@pytest.mark.parametrize(
    "qubits,parameters,occurrences",
    [(1, 2, 1), (17, 2, 1), (True, 2, 1), (4.0, 2, 1),
     (4, 0, 1), (4, 9, 1), (4, True, 1), (4, 2, 0), (4, 2, 9), (4, 2, 1.5)],
)
def test_invalid_architecture_is_rejected(qubits, parameters, occurrences):
    with pytest.raises(ValueError):
        MultiResponseEngine(qubits, parameters, occurrences)


@pytest.mark.parametrize(
    "vectors,theta,gradient",
    [
        (np.zeros(4), np.zeros(2), True),
        (np.zeros((1, 5)), np.zeros(2), True),
        (np.full((1, 4), np.nan), np.zeros(2), True),
        (np.zeros((1, 4)), np.zeros(3), True),
        (np.zeros((1, 4)), 0.3, True),
        (np.zeros((1, 4)), np.full(2, np.inf), True),
        (np.zeros((1, 4)), np.zeros(2, dtype=bool), True),
        (np.zeros((1, 4), dtype=complex), np.zeros(2), True),
        (np.zeros((1, 4)), np.zeros(2), 1),
    ],
)
def test_invalid_inputs_are_rejected_without_simulation(vectors, theta, gradient):
    engine = MultiResponseEngine(4, 2, 1)
    with pytest.raises(ValueError):
        engine.evaluate(vectors, theta, gradient=gradient)
    assert engine.diagnostics["cached_responses"] == engine.diagnostics["native_evaluations"] == 0


@pytest.mark.parametrize("method", METHODS)
def test_empty_input_shapes(method):
    engine = MultiResponseEngine(4, 2, 1, method)
    values, gradients = engine.evaluate(np.empty((0, 4)), np.zeros(2), gradient=True)
    assert values.shape == (0,) and gradients.shape == (0, 2)
    values, gradients = engine.evaluate(np.empty((0, 4)), np.zeros(2), gradient=False)
    assert values.shape == (0,) and gradients is None


def test_classical_adjoint_does_not_import_quantum_libraries():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,numpy as np; "
            "from group_memory_qml.multi_response import MultiResponseEngine; "
            "e=MultiResponseEngine(4,2,2,'classical_adjoint'); "
            "e.evaluate(np.ones((2,4)),np.array([0.2,-0.3]),gradient=True); "
            "assert not any(m.startswith(('qiskit','qiskit_aer')) for m in sys.modules)",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
