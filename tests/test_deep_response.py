"""Independent full-circuit checks for deep shared-parameter response memory."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from group_memory_qml.deep_response import DeepResponseEngine


METHODS = (
    "group_spectral",
    "group_grid",
    "qiskit_response",
    "qiskit_blocks",
    "classical_direct",
)


def mean_z(state):
    """Compute the observable directly, with Qiskit's little-endian convention."""
    probabilities = np.abs(np.asarray(state)) ** 2
    qubits = int(np.log2(len(probabilities)))
    indices = np.arange(len(probabilities), dtype=np.uint64)
    return float(
        np.mean(
            [
                np.dot(probabilities, 1.0 - 2.0 * ((indices >> q) & 1))
                for q in range(qubits)
            ]
        )
    )


def bind_source(engine, vector, angle, occurrence=None, shift=0.0):
    """Build the declared circuit independently, optionally shifting one occurrence."""
    output = QuantumCircuit(engine.n_qubits)
    for site, value in enumerate(vector):
        output.ry(float(value), site)
    for layer in range(engine.layers):
        output.ry(angle + (shift if layer == occurrence else 0.0), layer % engine.n_qubits)
        kind = layer % 2
        for site in range(engine.n_qubits):
            output.rx(0.23 + 0.03 * kind, site)
            output.rz(0.17 - 0.02 * kind, site)
        seen_edges = set()
        for site in range(engine.n_qubits):
            edge = frozenset((site, (site + 1) % engine.n_qubits))
            if edge not in seen_edges:
                output.cz(site, (site + 1) % engine.n_qubits)
                seen_edges.add(edge)
    return output


def oracle(engine, vectors, angle):
    return np.asarray(
        [
            mean_z(Statevector.from_instruction(bind_source(engine, x, angle)).data)
            for x in vectors
        ]
    )


def occurrence_shift_oracle(engine, vectors, angle):
    """Differentiate tied weights by summing valid single-occurrence shifts."""
    derivatives = []
    for vector in vectors:
        derivative = 0.0
        for occurrence in range(engine.layers):
            plus = Statevector.from_instruction(
                bind_source(engine, vector, angle, occurrence, np.pi / 2)
            )
            minus = Statevector.from_instruction(
                bind_source(engine, vector, angle, occurrence, -np.pi / 2)
            )
            derivative += 0.5 * (mean_z(plus.data) - mean_z(minus.data))
        derivatives.append(derivative)
    return np.asarray(derivatives)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("qubits,layers", [(6, 2), (6, 8), (8, 16)])
def test_deep_values_and_shared_derivatives_match_full_source(method, qubits, layers):
    engine = DeepResponseEngine(qubits, layers, method)
    vectors = np.random.default_rng(2901 + layers).normal(0, 0.7, (2, qubits))
    for angle in (0.0, 0.413, -1.207):
        values, derivatives = engine.evaluate(vectors, angle, gradient=True)
        np.testing.assert_allclose(values, oracle(engine, vectors, angle), atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            derivatives,
            occurrence_shift_oracle(engine, vectors, angle),
            atol=1e-10,
            rtol=0,
        )


def test_shared_derivative_requires_sum_of_occurrences():
    engine = DeepResponseEngine(6, 8, "group_spectral")
    vectors = np.random.default_rng(581).normal(0, 0.8, (3, 6))
    angle = 0.713
    _, derivative = engine.evaluate(vectors, angle, gradient=True)
    invalid_two_query = 0.5 * (
        oracle(engine, vectors, angle + np.pi / 2)
        - oracle(engine, vectors, angle - np.pi / 2)
    )
    assert np.max(np.abs(derivative - invalid_two_query)) > 1e-3
    step = 1e-6
    finite_difference = (
        oracle(engine, vectors, angle + step) - oracle(engine, vectors, angle - step)
    ) / (2 * step)
    np.testing.assert_allclose(derivative, finite_difference, atol=2e-8, rtol=0)


def test_source_and_native_preserve_full_operator_including_global_phase():
    engine = DeepResponseEngine(6, 2, "group_spectral")
    vector = np.array([0.31, -0.73, 0.42, 0.05, -0.94, 1.13])
    angle = -0.375
    bindings = dict(zip(engine.inputs, vector))
    bindings[engine.weight] = angle
    expected = Operator(bind_source(engine, vector, angle)).data
    for circuit in (engine.source, engine.native):
        actual = Operator(circuit.assign_parameters(bindings)).data
        # Matrix equality deliberately disallows discarding a global phase.
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    source_copy = engine.source
    native_copy = engine.native
    source_copy.x(0)
    native_copy.x(0)
    block_copy = engine.blocks
    try:
        block_copy[0][:] = 0
    except ValueError:
        # A read-only detached snapshot is also a valid defensive interface.
        assert not block_copy[0].flags.writeable
    values, derivative = engine.evaluate(vector[None, :], angle, gradient=True)
    np.testing.assert_allclose(values, oracle(engine, vector[None, :], angle), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        derivative, occurrence_shift_oracle(engine, vector[None, :], angle), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("qubits,layers", [(2, 1), (6, 2), (6, 8), (8, 16)])
def test_spectral_state_coefficients_preserve_phase_and_half_integer_support(qubits, layers):
    engine = DeepResponseEngine(qubits, layers, "group_spectral")
    vectors = np.random.default_rng(3432 + layers).normal(size=(2, qubits))
    coefficients = engine._spectral_states(vectors)
    assert coefficients.shape == (2, layers + 1, 2**qubits)
    frequencies = np.arange(layers + 1) - layers / 2
    for angle in (0.0, 0.413, -1.207, np.pi, 2 * np.pi):
        actual = np.einsum("bkd,k->bd", coefficients, np.exp(1j * angle * frequencies))
        expected = np.stack(
            [
                Statevector.from_instruction(bind_source(engine, vector, angle)).data
                for vector in vectors
            ]
        )
        # These are complex amplitudes, not only observable or fidelity checks.
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    if layers % 2:
        shifted = np.einsum(
            "bkd,k->bd", coefficients, np.exp(1j * (0.413 + 2 * np.pi) * frequencies)
        )
        ordinary = np.einsum("bkd,k->bd", coefficients, np.exp(1j * 0.413 * frequencies))
        np.testing.assert_allclose(shifted, -ordinary, atol=1e-10, rtol=0)


@pytest.mark.parametrize("method", METHODS)
def test_input_and_return_value_mutation_does_not_poison_cache(method):
    engine = DeepResponseEngine(6, 4, method)
    vectors = np.random.default_rng(625).normal(0, 0.7, (3, 6))
    original = vectors.copy()
    values, derivatives = engine.evaluate(vectors, 0.21, gradient=True)
    saved_values, saved_derivatives = values.copy(), derivatives.copy()
    values[:] = 999.0
    derivatives[:] = -999.0
    vectors[0, 0] += 0.5
    changed, _ = engine.evaluate(vectors, 0.21, gradient=True)
    np.testing.assert_allclose(changed, oracle(engine, vectors, 0.21), atol=1e-10, rtol=0)
    order = [2, 0, 1, 0]
    recovered, gradient = engine.evaluate(original[order], 0.21, gradient=True)
    np.testing.assert_allclose(recovered, saved_values[order], atol=1e-10, rtol=0)
    np.testing.assert_allclose(gradient, saved_derivatives[order], atol=1e-10, rtol=0)


@pytest.mark.parametrize("method", METHODS[:-1])
def test_exact_response_keys_and_weights_reuse(method):
    engine = DeepResponseEngine(6, 4, method)
    vector = np.random.default_rng(312).normal(size=6)
    changed = vector.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    vectors = np.stack([vector, vector, changed])
    engine.evaluate(vectors, 0.31, gradient=True)
    before = engine.diagnostics.copy()
    assert before["cached_responses"] == 2
    assert before["check_evaluations"] == 4
    assert before["response_degree"] == 4
    assert before["coefficient_bytes"] > 0
    for angle in (0.32, -2.1, 0.32 + 2 * np.pi):
        values, gradient = engine.evaluate(vectors[[2, 0, 1]], angle, gradient=True)
        np.testing.assert_allclose(
            values, oracle(engine, vectors[[2, 0, 1]], angle), atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            gradient,
            occurrence_shift_oracle(engine, vectors[[2, 0, 1]], angle),
            atol=1e-10,
            rtol=0,
        )
    after = engine.diagnostics
    assert after["cached_responses"] == before["cached_responses"]
    assert after["native_evaluations"] == before["native_evaluations"]
    assert after["check_evaluations"] == before["check_evaluations"]
    assert after["coefficient_bytes"] == before["coefficient_bytes"]


@pytest.mark.parametrize("method", METHODS[:-1])
def test_off_grid_failure_does_not_commit_response_or_relax_checks(method, monkeypatch):
    engine = DeepResponseEngine(6, 4, method)
    vectors = np.random.default_rng(543).normal(size=(2, 6))
    original = engine._native_values
    assert engine.diagnostics["check_atol"] == 1e-10

    def damaged_check(inputs, angles):
        result = original(inputs, angles)
        for i, angle in enumerate(np.asarray(angles).reshape(-1)):
            if angle == -0.583:
                result[i] += 2e-10
        return result

    monkeypatch.setattr(engine, "_native_values", damaged_check)
    with pytest.raises(RuntimeError, match="check|verification|validation"):
        engine.evaluate(vectors, 0.43, gradient=True)
    assert engine.diagnostics["cached_responses"] == 0
    assert engine.diagnostics["check_failures"] > 0
    charged = engine.diagnostics["native_evaluations"]
    assert charged > 0
    monkeypatch.setattr(engine, "_native_values", original)
    result, derivative = engine.evaluate(vectors, 0.43, gradient=True)
    np.testing.assert_allclose(result, oracle(engine, vectors, 0.43), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        derivative, occurrence_shift_oracle(engine, vectors, 0.43), atol=1e-10, rtol=0
    )
    assert engine.diagnostics["native_evaluations"] > charged
    assert engine.diagnostics["cached_responses"] == 2


def test_qiskit_response_controls_pay_identical_probe_and_check_work():
    engines = [DeepResponseEngine(6, 4, method) for method in ("qiskit_response", "qiskit_blocks")]
    vectors = np.random.default_rng(987).normal(size=(2, 6))
    left, right = [engine.evaluate(vectors, 0.33, gradient=True) for engine in engines]
    for actual, expected in zip(left, right):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    for key in ("check_evaluations", "cached_responses", "response_degree"):
        assert engines[0].diagnostics[key] == engines[1].diagnostics[key]
    for engine in engines:
        diagnostics = engine.diagnostics
        # Dense-block control accounts its numeric unitary circuits separately.
        total = diagnostics["native_evaluations"] + diagnostics["qiskit_block_evaluations"]
        assert total == 2 * (2 * 4 + 1 + 2)


@pytest.mark.parametrize("method", ("group_spectral", "qiskit_response"))
def test_asymmetric_architecture_does_not_reuse_cyclic_input_orbits(method):
    engine = DeepResponseEngine(6, 4, method)
    vector = np.array([0.17, -0.83, 1.31, 0.42, -0.37, 0.65])
    vectors = np.stack([vector, np.roll(vector, 1)])
    angle = np.array(0.43)
    first, _ = engine.evaluate(vectors, angle, gradient=True)
    np.testing.assert_allclose(first, oracle(engine, vectors, float(angle)), atol=1e-10, rtol=0)
    assert abs(first[0] - first[1]) > 1e-4
    assert engine.diagnostics["cached_responses"] == 2
    evaluations = engine.diagnostics["native_evaluations"]
    angle[...] = -0.71
    updated, derivative = engine.evaluate(vectors, angle, gradient=True)
    np.testing.assert_allclose(updated, oracle(engine, vectors, float(angle)), atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        derivative,
        occurrence_shift_oracle(engine, vectors, float(angle)),
        atol=1e-10,
        rtol=0,
    )
    assert engine.diagnostics["native_evaluations"] == evaluations


@pytest.mark.parametrize("method", METHODS)
def test_forward_only_and_empty_batches(method):
    engine = DeepResponseEngine(6, 2, method)
    vectors = np.random.default_rng(111).normal(size=(2, 6))
    values, derivative = engine.evaluate(vectors, -0.314, gradient=False)
    assert derivative is None
    np.testing.assert_allclose(values, oracle(engine, vectors, -0.314), atol=1e-10, rtol=0)
    empty, derivative = engine.evaluate(np.empty((0, 6)), 0.5, gradient=False)
    assert empty.shape == (0,) and derivative is None
    empty, derivative = engine.evaluate(np.empty((0, 6)), 0.5, gradient=True)
    assert empty.shape == derivative.shape == (0,)


@pytest.mark.parametrize("method", METHODS[:-1])
def test_forward_only_cache_is_exact_and_promotes_to_persistent_response(method):
    engine = DeepResponseEngine(6, 4, method)
    unique = np.random.default_rng(774).normal(size=(2, 6))
    vectors = unique[[0, 1, 0]]

    def executed_states():
        diagnostic = engine.diagnostics
        return sum(
            diagnostic[key]
            for key in (
                "native_evaluations",
                "qiskit_block_evaluations",
                "classical_direct_state_evaluations",
            )
        )

    values, _ = engine.evaluate(vectors, 0.43, gradient=False)
    np.testing.assert_allclose(values, oracle(engine, vectors, 0.43), atol=1e-10, rtol=0)
    assert executed_states() == 2
    assert engine.diagnostics["cached_direct_values"] == 2
    assert engine.diagnostics["cached_responses"] == 0
    assert engine.diagnostics["check_evaluations"] == 0
    engine.evaluate(vectors[::-1], 0.43, gradient=False)
    assert executed_states() == 2
    shifted_angle = np.nextafter(0.43, np.inf)
    engine.evaluate(vectors, shifted_angle, gradient=False)
    assert executed_states() == 4
    assert engine.diagnostics["direct_cache_invalidations"] == 1
    _, gradient = engine.evaluate(vectors, shifted_angle, gradient=True)
    np.testing.assert_allclose(
        gradient,
        occurrence_shift_oracle(engine, vectors, shifted_angle),
        atol=1e-10,
        rtol=0,
    )
    assert engine.diagnostics["cached_responses"] == 2
    assert engine.diagnostics["check_evaluations"] == 4
    work_after_calibration = executed_states()
    values, _ = engine.evaluate(vectors, -0.37, gradient=False)
    np.testing.assert_allclose(values, oracle(engine, vectors, -0.37), atol=1e-10, rtol=0)
    assert executed_states() == work_after_calibration
    assert engine.diagnostics["cached_responses"] == 2


@pytest.mark.parametrize(
    "vectors,angle",
    [
        (np.zeros(6), 0.2),
        (np.zeros((2, 5)), 0.2),
        (np.full((1, 6), np.nan), 0.2),
        (np.full((1, 6), np.inf), 0.2),
        (np.zeros((1, 6)), np.nan),
        (np.zeros((1, 6)), np.inf),
        (np.zeros((1, 6)), np.array([0.2, 0.3])),
    ],
)
def test_invalid_inputs_fail_closed(vectors, angle):
    engine = DeepResponseEngine(6, 2, "group_spectral")
    with pytest.raises(ValueError):
        engine.evaluate(vectors, angle, gradient=True)


@pytest.mark.parametrize(
    "qubits,layers",
    [(1, 2), (11, 2), (True, 2), (6.0, 2), (6, 0), (6, 33), (6, True), (6, 2.5)],
)
def test_invalid_architecture_rejected_before_compilation(qubits, layers):
    with pytest.raises(ValueError):
        DeepResponseEngine(qubits, layers, "group_spectral")


@pytest.mark.parametrize(
    "vectors,angle,gradient",
    [
        (np.zeros((1, 6), dtype=bool), 0.2, True),
        (np.zeros((1, 6), dtype=complex), 0.2, True),
        (np.zeros((1, 6)), True, True),
        (np.zeros((1, 6)), 0.2, 1),
    ],
)
def test_unsupported_types_are_not_silently_coerced(vectors, angle, gradient):
    engine = DeepResponseEngine(6, 2, "group_spectral")
    with pytest.raises(ValueError):
        engine.evaluate(vectors, angle, gradient=gradient)
    assert engine.diagnostics["native_evaluations"] == 0
    assert engine.diagnostics["cached_responses"] == 0


def test_direct_control_does_not_import_quantum_runtime():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np; "
            "from group_memory_qml.deep_response import DeepResponseEngine; "
            "e = DeepResponseEngine(6, 4, 'classical_direct'); "
            "e.evaluate(np.ones((2, 6)), 0.3, gradient=True); "
            "assert not any(m.startswith(('qiskit', 'qiskit_aer')) for m in sys.modules)",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
