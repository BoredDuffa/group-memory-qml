"""Exact response memory for genuinely interleaved shared-weight circuits.

There are L occurrences of RY(theta), separated by alternating fixed
RX/RZ/CZ-ring blocks.  The expectation has integer Fourier support [-L, L].
The state has half-angle exponents -L/2, -L/2+1, ..., L/2.  ``group_spectral``
propagates these state coefficients, sharing intermediate work across the
response. ``group_grid`` instead evolves the same 2L+1 response states in a
NumPy batch.  Qiskit controls receive the same grid, degree, persistent exact
input cache and two off-grid checks; ``qiskit_blocks`` also receives the same
dense fixed-block matrices.  All quantum methods pay fresh native compilation.

This is specialized *classical CPU simulation*.  Fourier state coefficients
and intermediate statevectors are not memory that a QPU can expose for free.
Dense blocks use O(4**n) memory and state coefficients use O((L+1)*2**n);
neither construction is intended to scale to large quantum hardware.
``classical_direct`` is a separate local-gate forward-tangent implementation
of the identical model.  Its ordinary execution imports no quantum library
and constructs neither dense block matrices nor Fourier coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from time import perf_counter

import numpy as np


METHODS = (
    "group_spectral", "group_grid", "qiskit_response", "qiskit_blocks",
    "classical_direct",
)
CHECK_ANGLES = np.array([.317, -.583], dtype=np.float64)
CHECK_ANGLES.flags.writeable = False
CHECK_ATOL = 1e-10
_SEED = 6147
_ARCHITECTURE = "deep-ry-single-site-alternating-rx-rz-cz-ring-v1"


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain finite real values")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    return array


def _theta(value):
    value = _real(value, "theta")
    if value.size != 1 or value.ndim > 1:
        raise ValueError("theta must be one real scalar")
    return float(value.reshape(-1)[0])


def _edges(n):
    return tuple(sorted({tuple(sorted((site, (site+1) % n))) for site in range(n)}))


def _block_angles(kind):
    return .23+.03*kind, .17-.02*kind


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _logical_metadata(n, layers):
    """Source fingerprint and depth without importing a quantum package."""
    records = [("ry", [site], [f"input[{site}]"]) for site in range(n)]
    for layer in range(layers):
        records.append(("ry", [layer % n], ["theta"]))
        rx, rz = _block_angles(layer % 2)
        for site in range(n):
            records.extend([("rx", [site], [str(rx)]), ("rz", [site], [str(rz)])])
        records.extend(("cz", list(edge), []) for edge in _edges(n))
    depths = [0]*n
    for _, sites, _ in records:
        new_depth = 1+max(depths[site] for site in sites)
        for site in sites:
            depths[site] = new_depth
    return _fingerprint({"n": n, "phase": "0.0", "operations": records}), max(depths)


def build_deep_circuit(n_qubits, layers):
    """Return a fresh source circuit, input parameters, shared weight, blocks."""
    n = _integer(n_qubits, "n_qubits", 2, 10)
    layers = _integer(layers, "layers", 1, 32)
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector

    inputs = tuple(ParameterVector("input", n))
    weight = Parameter("theta")
    blocks = []
    for kind in range(2):
        block = QuantumCircuit(n, name=f"fixed_{kind}")
        rx, rz = _block_angles(kind)
        for site in range(n):
            block.rx(rx, site)
            block.rz(rz, site)
        for first, second in _edges(n):
            block.cz(first, second)
        blocks.append(block)
    source = QuantumCircuit(n)
    for site, parameter in enumerate(inputs):
        source.ry(parameter, site)
    for layer in range(layers):
        source.ry(weight, layer % n)
        source.compose(blocks[layer % 2], inplace=True)
    return source, inputs, weight, tuple(blocks)


def _circuit_signature(circuit):
    return _fingerprint({
        "n": circuit.num_qubits, "phase": str(circuit.global_phase),
        "operations": [
            (item.operation.name, [circuit.find_bit(q).index for q in item.qubits],
             [str(parameter) for parameter in item.operation.params])
            for item in circuit.data
        ],
    })


@dataclass(frozen=True)
class _InputPlan:
    keys: tuple[bytes, ...]
    vectors: np.ndarray
    inverse: np.ndarray


class DeepResponseEngine:
    """Exact mean-Z forward values and derivatives for one shared angle.

    ``evaluate(X, theta, gradient=False)`` returns a pair of arrays, the second
    being None unless requested.  X has shape (batch, n_qubits); theta is a
    single real value shared by every trainable gate.  Input keys retain every
    float64 bit. No translations, rounded values, or nearby angles are treated
    as equivalent. The cache survives weight updates because it stores the
    complete supported response function.

    All four response methods check *every* newly learned input against two
    independent native Aer executions before committing any part of a batch.
    The two checks supplement the proven frequency bound, not a substitute for
    that bound. A failed or nonfinite check raises and leaves the cache intact.
    This mutable engine is not thread-safe.
    """

    def __init__(self, n_qubits, layers, method="group_spectral"):
        started = perf_counter()
        self.n_qubits = _integer(n_qubits, "n_qubits", 2, 10)
        self.layers = _integer(layers, "layers", 1, 32)
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self._dimension = 1 << self.n_qubits
        words = np.arange(self._dimension)
        self._pairs = tuple(
            (words[(words & (1 << site)) == 0], words[(words & (1 << site)) != 0])
            for site in range(self.n_qubits)
        )
        self._z = np.mean(
            np.array([1.-2.*((words >> site) & 1) for site in range(self.n_qubits)]),
            axis=0,
        )
        self._cz_sign = np.ones(self._dimension)
        for first, second in _edges(self.n_qubits):
            self._cz_sign *= 1-2*(((words >> first) & 1)*((words >> second) & 1))
        self._degree = self.layers
        self._grid = 2*np.pi*np.arange(2*self.layers+1)/(2*self.layers+1)
        self._grid.flags.writeable = False
        self._cache = {}
        self._direct_cache = {}
        self._direct_weight_key = None
        self._direct_invalidations = 0
        self._plans = {}
        self._source = self._native = self._forward = self._block_forward = None
        self._inputs = self._weight = self._block_sources = self._blocks = None
        self._backend = None
        self._source_signature, self._source_depth = _logical_metadata(self.n_qubits, self.layers)
        self._native_signature = None
        self._block_hashes = None
        self._block_gates = None
        self._compile_seconds = self._block_build_seconds = 0.
        self._calibration_seconds = self._evaluate_seconds = self._planning_seconds = 0.
        self._check_seconds = 0.
        self._native_evaluations = self._block_evaluations = 0
        self._grid_state_evaluations = self._direct_state_evaluations = 0
        self._check_evaluations = self._check_failures = 0
        self._fixed_block_state_applications = self._derivative_values = 0
        self._max_check_error = 0.
        self._peak_coefficient_bytes = self._plan_hits = 0
        self._native_cx_per_circuit = None
        self._native_depth = None
        self._architecture_signature = _fingerprint({
            "architecture": _ARCHITECTURE, "n": self.n_qubits, "layers": self.layers,
            "rx": [_block_angles(k)[0] for k in range(2)],
            "rz": [_block_angles(k)[1] for k in range(2)],
            "edges": _edges(self.n_qubits), "observable": "mean_Z",
        })
        if method != "classical_direct":
            self._ensure_quantum()
            if method != "qiskit_response":
                self._ensure_blocks()
            if method == "qiskit_blocks":
                self._build_block_forward()
        self._setup_seconds = perf_counter()-started

    def _ensure_quantum(self):
        if self._native is not None:
            return
        from qiskit import transpile
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator

        self._source, self._inputs, self._weight, self._block_sources = (
            build_deep_circuit(self.n_qubits, self.layers)
        )
        if self._source_signature != _circuit_signature(self._source):
            raise RuntimeError("Logical source fingerprint does not match Qiskit source")
        if self._source_depth != self._source.depth():
            raise RuntimeError("Logical depth does not match Qiskit source")
        started = perf_counter()
        self._native = transpile(
            self._source, basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=3, approximation_degree=1.,
            seed_transpiler=_SEED, num_processes=1,
        )
        self._compile_seconds += perf_counter()-started
        if self._native.num_qubits != self.n_qubits or self._native.layout is not None:
            raise RuntimeError("Unexpected native layout")
        self._native_signature = _circuit_signature(self._native)
        self._native_cx_per_circuit = int(self._native.count_ops().get("cx", 0))
        self._native_depth = int(self._native.depth())
        self._observable = SparsePauliOp.from_sparse_list(
            [("Z", [site], 1/self.n_qubits) for site in range(self.n_qubits)],
            self.n_qubits,
        )
        self._forward = self._native.copy()
        self._forward.save_expectation_value(self._observable, range(self.n_qubits), label="z")
        self._backend = AerSimulator(
            method="statevector", device="CPU", precision="double",
            max_parallel_threads=1, max_parallel_experiments=1,
            max_parallel_shots=1, runtime_parameter_bind_enable=True,
        )

    def _ensure_blocks(self):
        if self._blocks is not None:
            return
        self._ensure_quantum()
        from qiskit.quantum_info import Operator
        started = perf_counter()
        self._blocks = tuple(
            np.ascontiguousarray(Operator(block).data) for block in self._block_sources
        )
        for block in self._blocks:
            block.flags.writeable = False
        self._block_hashes = tuple(hashlib.sha256(b.tobytes()).hexdigest() for b in self._blocks)
        self._block_build_seconds += perf_counter()-started

    def _build_block_forward(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate
        # Operator built each matrix from unitary RX/RZ/CZ gates. Repeating a
        # cubic dense unitarity test at every layer and every bound probe would
        # waste the very cached-block information supplied to this control.
        # Aer 0.17.2 produces wrong answers for this fixture if the gate holds
        # our read-only ndarray. Writable copies preserve the exact matrix
        # while making both ordinary and runtime parameter binding correct.
        self._block_gates = tuple(UnitaryGate(block.copy(), check_input=False) for block in self._blocks)
        circuit = QuantumCircuit(self.n_qubits)
        for site, parameter in enumerate(self._inputs):
            circuit.ry(parameter, site)
        for layer in range(self.layers):
            circuit.ry(self._weight, layer % self.n_qubits)
            circuit.append(self._block_gates[layer % 2], range(self.n_qubits), copy=False)
        circuit.save_expectation_value(self._observable, range(self.n_qubits), label="z")
        self._block_forward = circuit

    @property
    def source(self):
        self._ensure_quantum()
        return self._source.copy()

    @property
    def native(self):
        self._ensure_quantum()
        return self._native.copy()

    @property
    def inputs(self):
        self._ensure_quantum()
        return self._inputs

    @property
    def weight(self):
        self._ensure_quantum()
        return self._weight

    @property
    def blocks(self):
        self._ensure_blocks()
        return tuple(block.copy() for block in self._blocks)

    @property
    def grid(self):
        return self._grid.copy()

    @property
    def diagnostics(self):
        return {
            "method": self.method, "architecture": _ARCHITECTURE,
            "architecture_sha256": self._architecture_signature,
            "source_sha256": self._source_signature, "native_sha256": self._native_signature,
            "source_depth": self._source_depth,
            "fixed_block_sha256": self._block_hashes,
            "n_qubits": self.n_qubits, "layers": self.layers,
            "response_degree": self._degree, "response_grid_size": len(self._grid),
            "response_state_support_size": self.layers+1,
            "cached_responses": len(self._cache),
            "coefficient_bytes": sum(value.nbytes for value in self._cache.values()),
            "peak_coefficient_bytes": self._peak_coefficient_bytes,
            "peak_coefficient_bytes_scope": "Largest individual state-coefficient tensor or retained response table, not process RSS or all temporary arrays",
            "dense_block_bytes": 0 if self._blocks is None else sum(b.nbytes for b in self._blocks),
            "writable_qiskit_block_copy_bytes": (0 if self._block_gates is None
                                                else sum(g.params[0].nbytes for g in self._block_gates)),
            "block_buffers_bytes": ((0 if self._blocks is None else sum(b.nbytes for b in self._blocks))
                                    +(0 if self._block_gates is None else sum(g.params[0].nbytes for g in self._block_gates))),
            "native_evaluations": self._native_evaluations,
            "native_cx_per_circuit": self._native_cx_per_circuit,
            "native_depth": self._native_depth,
            "native_cx_work": (0 if self._native_cx_per_circuit is None
                               else self._native_evaluations*self._native_cx_per_circuit),
            "qiskit_block_evaluations": self._block_evaluations,
            "group_grid_state_evaluations": self._grid_state_evaluations,
            "classical_direct_state_evaluations": self._direct_state_evaluations,
            "fixed_block_state_applications": self._fixed_block_state_applications,
            "derivative_values": self._derivative_values,
            "check_evaluations": self._check_evaluations, "check_failures": self._check_failures,
            "max_check_error": self._max_check_error, "check_atol": CHECK_ATOL,
            "check_angles": CHECK_ANGLES.tolist(), "plan_cache_hits": self._plan_hits,
            "direct_cache_invalidations": self._direct_invalidations,
            "cached_direct_values": len(self._direct_cache),
            "setup_seconds": self._setup_seconds,
            "compile_seconds": self._compile_seconds,
            "block_build_seconds": self._block_build_seconds,
            "calibration_seconds": self._calibration_seconds,
            "planning_seconds": self._planning_seconds,
            "check_seconds": self._check_seconds, "evaluate_seconds": self._evaluate_seconds,
            "compute": "Ideal CPU simulation; no QPU, finite shots, or device noise",
            "cx_scope": "Actual native Aer executions only; dense/spectral propagation is separate classical work",
            "spatial_symmetry_reuse": False,
            "binding_mode": ("runtime_parameter_binding_writable_block_copy" if self.method == "qiskit_blocks"
                             else "native_runtime_parameter_binding" if self.method == "qiskit_response"
                             else "classical_direct_local_gates" if self.method == "classical_direct"
                             else "classical_coefficient_or_state_propagation"),
        }

    def _prepare(self, X):
        started = perf_counter()
        identity = (X.shape, X.tobytes())
        if identity in self._plans:
            self._plan_hits += 1
            self._planning_seconds += perf_counter()-started
            return self._plans[identity]
        positions = {}
        vectors, inverse = [], []
        for row in X:
            key = row.tobytes()
            if key not in positions:
                positions[key] = len(vectors)
                vectors.append(row.copy())
            inverse.append(positions[key])
        array = np.asarray(vectors, dtype=np.float64)
        inverse = np.asarray(inverse, dtype=np.int64)
        array.flags.writeable = inverse.flags.writeable = False
        plan = _InputPlan(tuple(positions), array, inverse)
        self._plans[identity] = plan
        self._planning_seconds += perf_counter()-started
        return plan

    def _initial_state(self, X):
        state = np.ones((len(X), self._dimension), dtype=np.complex128)
        words = np.arange(self._dimension)
        for site in range(self.n_qubits):
            state *= np.where(
                ((words >> site) & 1)[None, :],
                np.sin(X[:, site:site+1]/2), np.cos(X[:, site:site+1]/2),
            )
        return state

    def _ry(self, state, angle, site):
        zero, one = self._pairs[site]
        cosine = np.cos(np.asarray(angle)/2)[..., None]
        sine = np.sin(np.asarray(angle)/2)[..., None]
        output = np.empty_like(state)
        output[..., zero] = cosine*state[..., zero]-sine*state[..., one]
        output[..., one] = sine*state[..., zero]+cosine*state[..., one]
        return output

    def _dry(self, state, angle, site):
        zero, one = self._pairs[site]
        cosine, sine = np.cos(angle/2)/2, np.sin(angle/2)/2
        output = np.empty_like(state)
        output[..., zero] = -sine*state[..., zero]-cosine*state[..., one]
        output[..., one] = cosine*state[..., zero]-sine*state[..., one]
        return output

    def _fixed_local(self, state, kind):
        rx, rz = _block_angles(kind)
        cosine, sine = np.cos(rx/2), np.sin(rx/2)
        lo_phase, hi_phase = np.exp(-.5j*rz), np.exp(.5j*rz)
        for zero, one in self._pairs:
            lo, hi = state[..., zero].copy(), state[..., one].copy()
            state[..., zero] = lo_phase*(cosine*lo-1j*sine*hi)
            state[..., one] = hi_phase*(-1j*sine*lo+cosine*hi)
        state *= self._cz_sign
        return state

    def _direct_values(self, X, angle, gradient):
        state = self._initial_state(X)
        tangent = np.zeros_like(state) if gradient else None
        for layer in range(self.layers):
            site = layer % self.n_qubits
            if gradient:
                tangent = self._ry(tangent, angle, site)+self._dry(state, angle, site)
                tangent = self._fixed_local(tangent, layer % 2)
            state = self._fixed_local(self._ry(state, angle, site), layer % 2)
        values = np.einsum("bd,d,bd->b", state.conj(), self._z, state).real
        derivatives = (2*np.einsum("bd,d,bd->b", tangent.conj(), self._z, state).real
                       if gradient else None)
        self._direct_state_evaluations += len(X)
        return values, derivatives

    def _native_values(self, X, theta_values):
        """Paired native executions; also the independent check-injection hook."""
        return self._aer_values(X, theta_values, blocks=False)

    def _aer_values(self, X, theta_values, blocks=False):
        if not len(X):
            return np.empty(0)
        binds = {parameter: X[:, site].tolist()
                 for site, parameter in enumerate(self._inputs)}
        binds[self._weight] = np.asarray(theta_values, dtype=float).tolist()
        circuit = self._block_forward if blocks else self._forward
        result = self._backend.run(
            circuit, parameter_binds=[binds], shots=None, seed_simulator=_SEED,
        ).result()
        if blocks:
            self._block_evaluations += len(X)
        else:
            self._native_evaluations += len(X)
        if not result.success or len(result.results) != len(X):
            raise RuntimeError("Aer response execution failed")
        values = np.array([float(np.real(result.data(i)["z"])) for i in range(len(X))])
        if not np.isfinite(values).all():
            raise RuntimeError("Aer response execution returned nonfinite values")
        return values

    def _grid_values(self, X):
        count = len(self._grid)
        states = np.repeat(self._initial_state(X)[:, None, :], count, axis=1)
        for layer in range(self.layers):
            states = self._ry(states, self._grid, layer % self.n_qubits)
            states = states @ self._blocks[layer % 2].T
            self._fixed_block_state_applications += len(X)*count
        self._grid_state_evaluations += len(X)*count
        return np.einsum("bkd,d,bkd->bk", states.conj(), self._z, states).real

    def _spectral_states(self, X):
        """State coefficients for exponents k-L/2; exposed privately for tests."""
        coefficients = self._initial_state(X)[:, None, :]
        for layer in range(self.layers):
            zero, one = self._pairs[layer % self.n_qubits]
            y_state = np.empty_like(coefficients)
            y_state[..., zero] = -1j*coefficients[..., one]
            y_state[..., one] = 1j*coefficients[..., zero]
            updated = np.zeros((len(X), coefficients.shape[1]+1, self._dimension),
                               dtype=np.complex128)
            updated[:, :-1, :] += .5*(coefficients+y_state)
            updated[:, 1:, :] += .5*(coefficients-y_state)
            coefficients = updated @ self._blocks[layer % 2].T
            self._fixed_block_state_applications += len(X)*coefficients.shape[1]
            self._peak_coefficient_bytes = max(self._peak_coefficient_bytes,
                                               coefficients.nbytes)
        return coefficients

    def _spectral_coefficients(self, X):
        coefficients = self._spectral_states(X)
        output = np.empty((len(X), self.layers+1), dtype=np.complex128)
        for lag in range(self.layers+1):
            output[:, lag] = np.einsum(
                "bkd,d,bkd->b", coefficients[:, :self.layers+1-lag].conj(),
                self._z, coefficients[:, lag:],
            )
        return output

    @staticmethod
    def _contract(coefficients, angles, gradient=False):
        frequencies = np.arange(1, coefficients.shape[-1])
        phase = np.exp(1j*np.asarray(angles)[..., None]*frequencies)
        values = coefficients[..., 0].real+2*np.real(
            np.sum(coefficients[..., 1:]*phase, axis=-1))
        derivatives = (2*np.real(np.sum(
            coefficients[..., 1:]*(1j*frequencies)*phase, axis=-1))
            if gradient else None)
        return values, derivatives

    def _calibrate(self, keys, X):
        if not keys:
            return
        started = perf_counter()
        try:
            if self.method == "group_spectral":
                coefficients = self._spectral_coefficients(X)
            else:
                if self.method == "group_grid":
                    samples = self._grid_values(X)
                else:
                    samples = self._aer_values(
                        np.repeat(X, len(self._grid), axis=0),
                        np.tile(self._grid, len(X)),
                        blocks=self.method == "qiskit_blocks",
                    ).reshape(len(X), len(self._grid))
                coefficients = (np.fft.fft(samples, axis=1)/len(self._grid))[:, :self.layers+1]
            if not np.isfinite(coefficients).all():
                self._check_failures += 1
                raise RuntimeError("Response coefficients are not finite")
            check_started = perf_counter()
            actual = self._native_values(
                np.repeat(X, len(CHECK_ANGLES), axis=0),
                np.tile(CHECK_ANGLES, len(X)),
            ).reshape(len(X), len(CHECK_ANGLES))
            self._check_evaluations += len(X)*len(CHECK_ANGLES)
            predicted, _ = self._contract(coefficients[:, None, :], CHECK_ANGLES)
            error = float(np.max(np.abs(actual-predicted)))
            self._check_seconds += perf_counter()-check_started
            self._max_check_error = max(self._max_check_error, error)
            if not np.isfinite(error) or error > CHECK_ATOL:
                self._check_failures += 1
                raise RuntimeError(f"Response check failed: {error:.17g} > {CHECK_ATOL}")
            for key, response in zip(keys, coefficients):
                response = response.copy()
                response.flags.writeable = False
                self._cache[key] = response
            self._peak_coefficient_bytes = max(
                self._peak_coefficient_bytes, sum(value.nbytes for value in self._cache.values())
            )
        finally:
            self._calibration_seconds += perf_counter()-started

    def evaluate(self, X, theta, gradient=False):
        started = perf_counter()
        X = _real(X, "X")
        theta = _theta(theta)
        if not isinstance(gradient, (bool, np.bool_)):
            raise ValueError("gradient must be a boolean")
        if X.ndim != 2 or X.shape[1] != self.n_qubits:
            raise ValueError(f"X must have shape (batch, {self.n_qubits})")
        if not len(X):
            return np.empty(0), np.empty(0) if gradient else None
        plan = self._prepare(X)
        if self.method == "classical_direct":
            values, derivatives = self._direct_values(plan.vectors, theta, gradient)
        else:
            weight_key = np.asarray(theta, dtype=np.float64).tobytes()
            if self._direct_weight_key != weight_key:
                self._direct_invalidations += int(self._direct_weight_key is not None)
                self._direct_cache.clear()
                self._direct_weight_key = weight_key
            if gradient:
                missing = [i for i, key in enumerate(plan.keys) if key not in self._cache]
                self._calibrate([plan.keys[i] for i in missing], plan.vectors[missing])
                coefficients = np.array([self._cache[key] for key in plan.keys])
                values, derivatives = self._contract(coefficients, theta, True)
            else:
                values = np.empty(len(plan.keys))
                derivatives = None
                known = [i for i, key in enumerate(plan.keys) if key in self._cache]
                if known:
                    coefficients = np.array([self._cache[plan.keys[i]] for i in known])
                    values[known], _ = self._contract(coefficients, theta, False)
                missing = [i for i, key in enumerate(plan.keys)
                           if key not in self._cache and key not in self._direct_cache]
                if missing:
                    vectors = plan.vectors[missing]
                    if self.method.startswith("group_"):
                        direct, _ = self._direct_values(vectors, theta, False)
                    else:
                        direct = self._aer_values(
                            vectors, np.full(len(vectors), theta),
                            blocks=self.method == "qiskit_blocks",
                        )
                    for i, value in zip(missing, direct):
                        self._direct_cache[plan.keys[i]] = float(value)
                for i, key in enumerate(plan.keys):
                    if key not in self._cache:
                        values[i] = self._direct_cache[key]
        values = np.asarray(values[plan.inverse], dtype=float).copy()
        derivatives = (np.asarray(derivatives[plan.inverse], dtype=float).copy()
                       if gradient else None)
        if not np.isfinite(values).all() or (gradient and not np.isfinite(derivatives).all()):
            raise RuntimeError("Response evaluation returned nonfinite values")
        self._derivative_values += len(X) if gradient else 0
        self._evaluate_seconds += perf_counter()-started
        return values, derivatives
