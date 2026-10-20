"""Matrix-free exact responses with independently trainable quantum angles.

There are p independent parameters, each occurring r times in interleaved RY
gates and fixed RX/RZ/CZ-ring blocks. Expectation frequencies form the *full*
signed Cartesian product [-r,r]**p, including mixed-sign terms. State
coefficients have exponents (k_1-r/2,...,k_p-r/2), k_j in 0..r.

The spectral method propagates these coefficients through local gates, then
forms their observable-weighted Gram matrix and accumulates every frequency
difference. It never constructs a dense 2**n by 2**n circuit operator. The grid
ablation and Qiskit control use the same (2r+1)**p response support and exact
input-cache policy. The direct classical control differentiates the identical
model with an independent local-gate adjoint and needs no response grid.

This is ideal classical CPU simulation, not QPU execution. Response growth is
exponential in the number of independent parameters. Explicit guards bound
the response grid, coefficient pairs, estimated working arrays and retained
caches. The working-array budget is not a hard operating-system RSS limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math

import numpy as np

from .deep_response import (
    _block_angles, _circuit_signature, _edges, _fingerprint, _integer, _real,
)


METHODS = ("group_spectral", "group_grid", "qiskit_response", "qiskit_shift", "classical_adjoint")
_RESPONSE_METHODS = ("group_spectral", "group_grid", "qiskit_response")
CHECK_ATOL = 1e-10
_SEED = 6147
_ARCHITECTURE = "multi-ry-independent-interleaved-local-blocks-v1"
_MAX_COEFFICIENT_PAIRS = 262144


class ResponseResourceError(ValueError):
    """The declared exact response or estimated working set exceeds a guard."""


def _metadata(n, p, repetitions):
    operations = [("ry", [site], [f"input[{site}]"]) for site in range(n)]
    for layer in range(p*repetitions):
        operations.append(("ry", [layer % n], [f"theta[{layer % p}]"]))
        rx, rz = _block_angles(layer % 2)
        for site in range(n):
            operations.extend([("rx", [site], [str(rx)]), ("rz", [site], [str(rz)])])
        operations.extend(("cz", list(edge), []) for edge in _edges(n))
    depths = [0]*n
    for _, sites, _ in operations:
        depth = 1+max(depths[site] for site in sites)
        for site in sites:
            depths[site] = depth
    return _fingerprint({"n": n, "phase": "0.0", "operations": operations}), max(depths)


def build_multi_circuit(n_qubits, n_parameters, occurrences=2):
    """Return a fresh Qiskit source, input tuple and independent-weight tuple."""
    n = _integer(n_qubits, "n_qubits", 2, 16)
    p = _integer(n_parameters, "n_parameters", 1, 8)
    repetitions = _integer(occurrences, "occurrences", 1, 8)
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    inputs, weights = tuple(ParameterVector("input", n)), tuple(ParameterVector("theta", p))
    source = QuantumCircuit(n)
    for site, value in enumerate(inputs):
        source.ry(value, site)
    for layer in range(p*repetitions):
        source.ry(weights[layer % p], layer % n)
        rx, rz = _block_angles(layer % 2)
        for site in range(n):
            source.rx(rx, site)
            source.rz(rz, site)
        for first, second in _edges(n):
            source.cz(first, second)
    return source, inputs, weights


@dataclass(frozen=True)
class _Plan:
    keys: tuple[bytes, ...]
    vectors: np.ndarray
    inverse: np.ndarray


class MultiResponseEngine:
    """Mean-Z predictions and gradients of p independently trainable angles.

    ``evaluate(X, theta, gradient=False)`` accepts X of shape (batch,n) and
    theta of shape (p,). It returns predictions (batch,) and gradients
    (batch,p), or None when no gradient was requested.

    A gradient request learns each missing input's complete exact response.
    A forward-only request on a new input executes it once at the requested
    weights. The response cache survives weight changes; direct values do not.
    Every float64 input bit participates in cache identity. There is no orbit
    canonicalization, approximate matching, or independent-parameter collapse.
    New response batches are committed only after all their checks succeed.
    """

    def __init__(self, n_qubits, n_parameters, occurrences=2,
                 method="group_spectral", memory_budget_bytes=256*1024**2,
                 max_grid_points=4096):
        started = perf_counter()
        self._n = _integer(n_qubits, "n_qubits", 2, 16)
        self._p = _integer(n_parameters, "n_parameters", 1, 8)
        self._r = _integer(occurrences, "occurrences", 1, 8)
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self._method = method
        self._budget = _integer(memory_budget_bytes, "memory_budget_bytes", 1024, 16*1024**3)
        self._max_grid = _integer(max_grid_points, "max_grid_points", 1, 1000000)
        self._d = 1 << self._n
        self._length = self._p*self._r
        self._grid_size = (2*self._r+1)**self._p
        self._state_size = (self._r+1)**self._p
        self._grid_shape = (2*self._r+1,)*self._p
        self._state_shape = (self._r+1,)*self._p
        self._check_angles = np.array([
            .317+.071*np.arange(self._p), -.583+.043*np.arange(self._p),
        ])
        self._check_angles.flags.writeable = False
        self._base_bytes = (8*self._n+32)*self._d
        self._working_per_input = self._working_bytes(method)
        if method in _RESPONSE_METHODS:
            if self._grid_size > self._max_grid:
                raise ResponseResourceError(
                    f"Exact response needs {self._grid_size} grid points; limit is {self._max_grid}"
                )
            if self._state_size**2 > _MAX_COEFFICIENT_PAIRS:
                raise ResponseResourceError(
                    f"Exact response needs {self._state_size**2} coefficient pairs; "
                    f"limit is {_MAX_COEFFICIENT_PAIRS}"
                )
            self._base_bytes += (
                (3*self._p+1)*self._state_size**2*8
                + (self._grid_size+self._state_size)*self._p*24
            )
        if self._base_bytes+self._working_per_input > self._budget:
            raise ResponseResourceError(
                f"Estimated one-input workspace {self._base_bytes+self._working_per_input} "
                f"bytes exceeds budget {self._budget}"
            )
        self._source_sha, self._source_depth = _metadata(self._n, self._p, self._r)
        self._native_sha = None
        self._architecture_sha = _fingerprint({
            "architecture": _ARCHITECTURE, "n": self._n, "p": self._p,
            "occurrences": self._r, "observable": "mean_Z",
        })
        words = np.arange(self._d)
        self._pairs = tuple(
            (words[(words & (1 << q)) == 0], words[(words & (1 << q)) != 0])
            for q in range(self._n)
        )
        self._z = np.mean([1.-2.*((words >> q) & 1) for q in range(self._n)], axis=0)
        self._cz = np.ones(self._d)
        for first, second in _edges(self._n):
            self._cz *= 1-2*(((words >> first) & 1)*((words >> second) & 1))
        self._grid = self._frequencies = self._state_indices = self._differences = None
        if method in _RESPONSE_METHODS:
            self._initialize_response_basis()
        self._source = self._native = self._forward = self._inputs = self._weights = None
        self._occurrence_weights = None
        self._backend = None
        self._native_cx = self._native_depth = None
        self._cache, self._direct_cache, self._plans, self._shift_cache = {}, {}, {}, {}
        self._weight_key = None
        self._plan_bytes = self._cache_bytes = 0
        self._compile_s = self._calibration_s = self._evaluation_s = self._check_s = 0.
        self._planning_s = 0.
        self._native_evaluations = self._checks = self._check_failures = 0
        self._grid_evaluations = self._direct_evaluations = self._derivative_values = 0
        self._max_error = 0.
        self._invalidations = self._plan_hits = self._calibration_chunks = 0
        self._max_chunk = 0
        self._estimated_peak = self._base_bytes
        if method != "classical_adjoint":
            self._ensure_quantum()
        self._setup_s = perf_counter()-started

    @property
    def n_qubits(self): return self._n
    @property
    def n_parameters(self): return self._p
    @property
    def occurrences(self): return self._r
    @property
    def layers(self): return self._length
    @property
    def method(self): return self._method

    def _working_bytes(self, method):
        complex_bytes = 16
        if method == "group_spectral":
            return complex_bytes*(6*self._state_size*self._d
                                  + 3*self._state_size**2+4*self._grid_size)
        if method == "group_grid":
            return complex_bytes*(6*self._grid_size*self._d+4*self._grid_size)
        if method == "qiskit_response":
            return (complex_bytes*(6*self._d+4*self._grid_size)
                    +self._grid_size*(self._n+self._p)*8)
        if method == "qiskit_shift":
            return complex_bytes*(6*self._d+8*self._length+4*self._p)
        return complex_bytes*((self._length+6)*self._d+self._p)

    def _initialize_response_basis(self):
        if self._grid is not None:
            return
        if self._grid_size > self._max_grid or self._state_size**2 > _MAX_COEFFICIENT_PAIRS:
            raise ResponseResourceError("Response basis exceeds the declared resource guards")
        indices = np.indices(self._grid_shape).reshape(self._p, -1).T
        self._grid = 2*np.pi*indices/(2*self._r+1)
        self._frequencies = np.where(indices <= self._r, indices, indices-(2*self._r+1))
        self._state_indices = np.indices(self._state_shape).reshape(self._p, -1).T
        delta = self._state_indices[None, :, :]-self._state_indices[:, None, :]
        self._differences = np.ravel_multi_index(
            np.moveaxis(delta % (2*self._r+1), -1, 0), self._grid_shape,
        ).reshape(-1)
        for array in (self._grid, self._frequencies, self._state_indices, self._differences):
            array.flags.writeable = False

    def _ensure_quantum(self):
        if self._native is not None:
            return
        from qiskit import transpile, QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
        self._source, self._inputs, self._weights = build_multi_circuit(self._n, self._p, self._r)
        if _circuit_signature(self._source) != self._source_sha:
            raise RuntimeError("Independent logical source signature mismatch")
        operational = self._source
        if self._method == "qiskit_shift":
            self._occurrence_weights = tuple(ParameterVector("occurrence", self._length))
            operational = QuantumCircuit(self._n)
            for q, value in enumerate(self._inputs):
                operational.ry(value, q)
            for layer, parameter in enumerate(self._occurrence_weights):
                operational.ry(parameter, layer % self._n)
                rx, rz = _block_angles(layer % 2)
                for q in range(self._n):
                    operational.rx(rx, q)
                    operational.rz(rz, q)
                for first, second in _edges(self._n):
                    operational.cz(first, second)
        started = perf_counter()
        self._native = transpile(
            operational, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3,
            approximation_degree=1., seed_transpiler=_SEED, num_processes=1,
        )
        self._compile_s += perf_counter()-started
        if self._native.layout is not None or self._native.num_qubits != self._n:
            raise RuntimeError("Unexpected native layout")
        self._native_sha = _circuit_signature(self._native)
        self._native_cx = int(self._native.count_ops().get("cx", 0))
        self._native_depth = int(self._native.depth())
        self._observable = SparsePauliOp.from_sparse_list(
            [("Z", [q], 1/self._n) for q in range(self._n)], self._n,
        )
        self._forward = self._native.copy()
        self._forward.save_expectation_value(self._observable, range(self._n), label="z")
        self._backend = AerSimulator(
            method="statevector", precision="double", device="CPU",
            max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1,
            runtime_parameter_bind_enable=True, max_memory_mb=max(1, self._budget//1024**2),
        )

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
    def weights(self):
        self._ensure_quantum()
        return self._weights
    @property
    def occurrence_weights(self):
        """Native binding parameters for the occurrence-shift control."""
        self._ensure_quantum()
        return self._occurrence_weights
    @property
    def grid(self):
        self._initialize_response_basis()
        return self._grid.copy()
    @property
    def state_exponents(self):
        self._initialize_response_basis()
        return self._state_indices.copy()-self._r/2
    @property
    def check_angles(self):
        return self._check_angles.copy()

    @property
    def diagnostics(self):
        return {
            "method": self._method, "architecture": _ARCHITECTURE,
            "architecture_sha256": self._architecture_sha,
            "source_sha256": self._source_sha, "native_sha256": self._native_sha,
            "source_depth": self._source_depth, "native_depth": self._native_depth,
            "n_qubits": self._n, "n_parameters": self._p, "occurrences": self._r,
            "layers": self._length, "response_degree_per_parameter": [self._r]*self._p,
            "response_shape": list(self._grid_shape), "response_grid_size": self._grid_size,
            "response_state_support_size": self._state_size,
            "response_coefficient_pairs": self._state_size**2,
            "full_signed_frequency_tensor": True,
            "cached_responses": len(self._cache), "cached_direct_values": len(self._direct_cache),
            "cached_shift_gradients": len(self._shift_cache),
            "coefficient_bytes": self._cache_bytes,
            "estimated_plan_bytes": self._plan_bytes,
            "memory_budget_bytes": self._budget,
            "estimated_base_bytes": self._base_bytes,
            "estimated_working_bytes_per_input": self._working_per_input,
            "estimated_peak_workspace_bytes": self._estimated_peak,
            "memory_budget_scope": "Conservative controlled-array/cache estimate and Aer max_memory_mb; not a hard process RSS bound",
            "max_grid_points": self._max_grid, "max_coefficient_pairs": _MAX_COEFFICIENT_PAIRS,
            "calibration_chunks": self._calibration_chunks, "maximum_input_chunk": self._max_chunk,
            "dense_block_bytes": 0, "native_evaluations": self._native_evaluations,
            "native_cx_per_circuit": self._native_cx,
            "native_cx_work": 0 if self._native_cx is None else self._native_evaluations*self._native_cx,
            "group_grid_state_evaluations": self._grid_evaluations,
            "classical_direct_state_evaluations": self._direct_evaluations,
            "derivative_values": self._derivative_values,
            "check_evaluations": self._checks, "check_failures": self._check_failures,
            "max_check_error": self._max_error, "check_atol": CHECK_ATOL,
            "check_angles": self._check_angles.tolist(),
            "direct_cache_invalidations": self._invalidations, "plan_cache_hits": self._plan_hits,
            "setup_seconds": self._setup_s, "compile_seconds": self._compile_s,
            "calibration_seconds": self._calibration_s, "evaluate_seconds": self._evaluation_s,
            "planning_seconds": self._planning_s, "check_seconds": self._check_s,
            "spatial_symmetry_reuse": False,
            "compute": "Ideal CPU simulation; independent parameters; no QPU or finite shots",
            "gradient_rule": ("sum of valid single-occurrence parameter shifts"
                              if self._method == "qiskit_shift"
                              else "reverse statevector adjoint" if self._method == "classical_adjoint"
                              else "derivative of the full signed exact Fourier response"),
            "cx_scope": "Native Aer circuits only; matrix-free spectral/grid/adjoint work is classical",
        }

    def _chunk_size(self, count, extra_persistent=0, method=None, cap=32):
        per_input = self._working_per_input if method is None else self._working_bytes(method)
        held = self._base_bytes+self._plan_bytes+self._cache_bytes+extra_persistent
        available = self._budget-held
        size = min(count, cap, available//max(1, per_input))
        if size < 1:
            raise ResponseResourceError(
                f"Estimated cache plus one-input workspace exceeds {self._budget} byte budget"
            )
        self._max_chunk = max(self._max_chunk, int(size))
        self._estimated_peak = max(self._estimated_peak, held+int(size)*per_input)
        return int(size)

    def _prepare(self, X):
        started = perf_counter()
        identity = (X.shape, X.tobytes())
        if identity in self._plans:
            self._plan_hits += 1
            self._planning_s += perf_counter()-started
            return self._plans[identity]
        positions, vectors, inverse = {}, [], []
        for row in X:
            key = row.tobytes()
            if key not in positions:
                positions[key] = len(vectors)
                vectors.append(row.copy())
            inverse.append(positions[key])
        vectors, inverse = np.asarray(vectors), np.asarray(inverse, dtype=np.int64)
        cost = len(identity[1])+vectors.nbytes+inverse.nbytes+len(positions)*(self._n*8+128)
        if self._base_bytes+self._cache_bytes+self._plan_bytes+cost+self._working_per_input > self._budget:
            raise ResponseResourceError("Input plan would leave insufficient bounded working memory")
        vectors.flags.writeable = inverse.flags.writeable = False
        plan = _Plan(tuple(positions), vectors, inverse)
        self._plans[identity] = plan
        self._plan_bytes += cost
        self._planning_s += perf_counter()-started
        return plan

    def _initial_state(self, X):
        state = np.ones((len(X), self._d), dtype=np.complex128)
        words = np.arange(self._d)
        for q in range(self._n):
            state *= np.where(((words >> q) & 1)[None, :],
                              np.sin(X[:, q:q+1]/2), np.cos(X[:, q:q+1]/2))
        return state

    def _ry(self, state, angle, site):
        lo, hi = self._pairs[site]
        c, s = np.cos(np.asarray(angle)/2)[..., None], np.sin(np.asarray(angle)/2)[..., None]
        result = np.empty_like(state)
        result[..., lo] = c*state[..., lo]-s*state[..., hi]
        result[..., hi] = s*state[..., lo]+c*state[..., hi]
        return result

    def _dry(self, state, angle, site):
        lo, hi = self._pairs[site]
        c, s = np.cos(angle/2)/2, np.sin(angle/2)/2
        result = np.empty_like(state)
        result[..., lo] = -s*state[..., lo]-c*state[..., hi]
        result[..., hi] = c*state[..., lo]-s*state[..., hi]
        return result

    def _fixed(self, state, kind, inverse=False):
        rx, rz = _block_angles(kind)
        c, s = np.cos(rx/2), np.sin(rx/2)
        if inverse:
            state *= self._cz
        for lo, hi in self._pairs:
            a, b = state[..., lo].copy(), state[..., hi].copy()
            if inverse:
                a *= np.exp(.5j*rz)
                b *= np.exp(-.5j*rz)
                state[..., lo], state[..., hi] = c*a+1j*s*b, 1j*s*a+c*b
            else:
                state[..., lo] = np.exp(-.5j*rz)*(c*a-1j*s*b)
                state[..., hi] = np.exp(.5j*rz)*(-1j*s*a+c*b)
        if not inverse:
            state *= self._cz
        return state

    def _adjoint_values(self, X, theta, gradient):
        state = self._initial_state(X)
        before = []
        for layer in range(self._length):
            if gradient:
                before.append(state.copy())
            state = self._fixed(self._ry(state, theta[layer % self._p], layer % self._n),
                                layer % 2)
        values = np.einsum("bd,d,bd->b", state.conj(), self._z, state).real
        derivatives = None
        if gradient:
            derivatives = np.zeros((len(X), self._p))
            adjoint = state*self._z
            for layer in reversed(range(self._length)):
                angle, site = theta[layer % self._p], layer % self._n
                adjoint = self._fixed(adjoint, layer % 2, inverse=True)
                derivatives[:, layer % self._p] += 2*np.einsum(
                    "bd,bd->b", adjoint.conj(), self._dry(before[layer], angle, site)).real
                adjoint = self._ry(adjoint, -angle, site)
        self._direct_evaluations += len(X)
        return values, derivatives

    def _native_values(self, X, theta):
        """Independent native checks and fast runtime-bound Qiskit grid probes."""
        if self._method == "qiskit_shift":
            return self._run_native(X, theta[:, np.arange(self._length) % self._p],
                                    self._occurrence_weights)
        return self._run_native(X, theta, self._weights)

    def _run_native(self, X, theta, parameters):
        if not len(X):
            return np.empty(0)
        # This cap also bounds the parameter/result metadata held by one Aer job.
        step = self._chunk_size(len(X), method="qiskit_shift", cap=128)
        outputs = []
        for start in range(0, len(X), step):
            vectors, angles = X[start:start+step], theta[start:start+step]
            bindings = {parameter: vectors[:, q].tolist() for q, parameter in enumerate(self._inputs)}
            bindings.update({parameter: angles[:, j].tolist() for j, parameter in enumerate(parameters)})
            result = self._backend.run(
                self._forward, parameter_binds=[bindings], shots=None, seed_simulator=_SEED,
            ).result()
            self._native_evaluations += len(vectors)
            if not result.success or len(result.results) != len(vectors):
                raise RuntimeError("Native Aer evaluation failed")
            outputs.extend(float(np.real(result.data(i)["z"])) for i in range(len(vectors)))
        values = np.asarray(outputs)
        if not np.isfinite(values).all():
            raise RuntimeError("Native Aer returned nonfinite outputs")
        return values

    def _shift_values(self, X, theta):
        multiplier = 1+2*self._length
        settings = np.tile(theta[np.arange(self._length) % self._p], (multiplier, 1))
        for occurrence in range(self._length):
            settings[1+2*occurrence, occurrence] += np.pi/2
            settings[2+2*occurrence, occurrence] -= np.pi/2
        chunk = self._chunk_size(len(X), method="qiskit_shift")
        forwards, jacobians = [], []
        for start in range(0, len(X), chunk):
            vectors = X[start:start+chunk]
            values = self._run_native(
                np.repeat(vectors, multiplier, axis=0), np.tile(settings, (len(vectors), 1)),
                self._occurrence_weights,
            ).reshape(len(vectors), multiplier)
            derivatives = np.zeros((len(vectors), self._p))
            for occurrence in range(self._length):
                derivatives[:, occurrence % self._p] += (
                    values[:, 1+2*occurrence]-values[:, 2+2*occurrence])/2
            forwards.append(values[:, 0])
            jacobians.append(derivatives)
        return np.concatenate(forwards), np.concatenate(jacobians)

    def _spectral_states(self, X):
        """Return coefficients (batch,C,2**n) in ``state_exponents`` order."""
        if self._chunk_size(len(X)) < len(X):
            raise ResponseResourceError("Spectral state request exceeds the bounded input chunk")
        coefficients = self._initial_state(X).reshape((len(X),)+(1,)*self._p+(self._d,))
        for layer in range(self._length):
            parameter, site = layer % self._p, layer % self._n
            lo, hi = self._pairs[site]
            y_state = np.empty_like(coefficients)
            y_state[..., lo], y_state[..., hi] = -1j*coefficients[..., hi], 1j*coefficients[..., lo]
            shape = list(coefficients.shape)
            shape[1+parameter] += 1
            updated = np.zeros(shape, dtype=np.complex128)
            previous = [slice(None)]*len(shape)
            following = list(previous)
            previous[1+parameter], following[1+parameter] = slice(None, -1), slice(1, None)
            updated[tuple(previous)] += .5*(coefficients+y_state)
            updated[tuple(following)] += .5*(coefficients-y_state)
            coefficients = self._fixed(updated, layer % 2)
        return coefficients.reshape(len(X), self._state_size, self._d)

    def _spectral_coefficients(self, X):
        states = self._spectral_states(X)
        output = np.empty((len(X), self._grid_size), dtype=np.complex128)
        for i, coefficients in enumerate(states):
            gram = (coefficients.conj()*self._z) @ coefficients.T
            output[i] = (
                np.bincount(self._differences, weights=gram.real.reshape(-1),
                            minlength=self._grid_size)
                +1j*np.bincount(self._differences, weights=gram.imag.reshape(-1),
                               minlength=self._grid_size)
            )
        return output

    def _grid_coefficients(self, X):
        if self._method == "group_grid":
            states = np.repeat(self._initial_state(X)[:, None, :], self._grid_size, axis=1)
            for layer in range(self._length):
                states = self._fixed(
                    self._ry(states, self._grid[:, layer % self._p], layer % self._n), layer % 2,
                )
            samples = np.einsum("bkd,d,bkd->bk", states.conj(), self._z, states).real
            self._grid_evaluations += len(X)*self._grid_size
        else:
            # Native batches are separately capped; no dense state grid is held.
            samples = self._native_values(
                np.repeat(X, self._grid_size, axis=0), np.tile(self._grid, (len(X), 1)),
            ).reshape(len(X), self._grid_size)
        tensor = samples.reshape((len(X),)+self._grid_shape)
        axes = tuple(range(1, self._p+1))
        return (np.fft.fftn(tensor, axes=axes)/self._grid_size).reshape(len(X), self._grid_size)

    def _contract(self, coefficients, theta, gradient):
        phase = np.exp(1j*(self._frequencies @ theta))
        weighted = coefficients*phase
        values = weighted.sum(axis=-1).real
        derivatives = (np.real(weighted @ (1j*self._frequencies)) if gradient else None)
        return values, derivatives

    def _calibrate(self, keys, X):
        if not keys:
            return
        started = perf_counter()
        pending_bytes = len(keys)*self._grid_size*16
        chunk = self._chunk_size(len(keys), extra_persistent=pending_bytes)
        pending = []
        try:
            for start in range(0, len(keys), chunk):
                vectors = X[start:start+chunk]
                coefficients = (self._spectral_coefficients(vectors)
                                if self._method == "group_spectral"
                                else self._grid_coefficients(vectors))
                if not np.isfinite(coefficients).all():
                    self._check_failures += 1
                    raise RuntimeError("Nonfinite exact-response coefficients")
                check_started = perf_counter()
                actual = self._native_values(
                    np.repeat(vectors, 2, axis=0), np.tile(self._check_angles, (len(vectors), 1)),
                ).reshape(len(vectors), 2)
                self._checks += 2*len(vectors)
                predicted = np.column_stack([
                    self._contract(coefficients, angle, False)[0] for angle in self._check_angles
                ])
                error = float(np.max(np.abs(actual-predicted)))
                self._check_s += perf_counter()-check_started
                if np.isfinite(error):
                    self._max_error = max(self._max_error, error)
                if not np.isfinite(error) or error > CHECK_ATOL:
                    self._check_failures += 1
                    raise RuntimeError(f"Response check failed: {error:.17g} > {CHECK_ATOL}")
                pending.extend(row.copy() for row in coefficients)
                self._calibration_chunks += 1
            for key, coefficient in zip(keys, pending):
                coefficient.flags.writeable = False
                self._cache[key] = coefficient
                self._cache_bytes += coefficient.nbytes
        finally:
            self._calibration_s += perf_counter()-started

    def _direct(self, X, theta, gradient=False):
        chunk = self._chunk_size(len(X), method="classical_adjoint")
        values, derivatives = [], []
        for start in range(0, len(X), chunk):
            output, jacobian = self._adjoint_values(X[start:start+chunk], theta, gradient)
            values.append(output)
            if gradient:
                derivatives.append(jacobian)
        return np.concatenate(values), np.concatenate(derivatives) if gradient else None

    def evaluate(self, X, theta, gradient=False):
        started = perf_counter()
        X, theta = _real(X, "X"), _real(theta, "theta")
        if X.ndim != 2 or X.shape[1] != self._n:
            raise ValueError(f"X must have shape (batch,{self._n})")
        if theta.shape != (self._p,):
            raise ValueError(f"theta must have shape ({self._p},)")
        if not isinstance(gradient, (bool, np.bool_)):
            raise ValueError("gradient must be a boolean")
        if not len(X):
            return np.empty(0), np.empty((0, self._p)) if gradient else None
        plan = self._prepare(X)
        if self._method == "classical_adjoint":
            values, derivatives = self._direct(plan.vectors, theta, gradient)
        else:
            weight_key = theta.tobytes()
            if self._weight_key != weight_key:
                self._invalidations += int(self._weight_key is not None)
                self._direct_cache.clear()
                self._shift_cache.clear()
                self._weight_key = weight_key
            if self._method == "qiskit_shift":
                values = np.empty(len(plan.keys))
                derivatives = np.empty((len(plan.keys), self._p)) if gradient else None
                if gradient:
                    missing = [i for i, key in enumerate(plan.keys) if key not in self._shift_cache]
                    if missing:
                        output, jacobian = self._shift_values(plan.vectors[missing], theta)
                        for i, value, derivative in zip(missing, output, jacobian):
                            key = plan.keys[i]
                            self._direct_cache[key] = float(value)
                            self._shift_cache[key] = derivative.copy()
                    for i, key in enumerate(plan.keys):
                        values[i], derivatives[i] = self._direct_cache[key], self._shift_cache[key]
                else:
                    missing = [i for i, key in enumerate(plan.keys) if key not in self._direct_cache]
                    if missing:
                        direct = self._native_values(plan.vectors[missing],
                                                     np.tile(theta, (len(missing), 1)))
                        for i, value in zip(missing, direct):
                            self._direct_cache[plan.keys[i]] = float(value)
                    values[:] = [self._direct_cache[key] for key in plan.keys]
            elif gradient:
                missing = [i for i, key in enumerate(plan.keys) if key not in self._cache]
                self._calibrate([plan.keys[i] for i in missing], plan.vectors[missing])
                coefficients = np.array([self._cache[key] for key in plan.keys])
                values, derivatives = self._contract(coefficients, theta, True)
            else:
                values, derivatives = np.empty(len(plan.keys)), None
                known = [i for i, key in enumerate(plan.keys) if key in self._cache]
                if known:
                    coefficients = np.array([self._cache[plan.keys[i]] for i in known])
                    values[known], _ = self._contract(coefficients, theta, False)
                missing = [i for i, key in enumerate(plan.keys)
                           if key not in self._cache and key not in self._direct_cache]
                if missing:
                    vectors = plan.vectors[missing]
                    if self._method == "qiskit_response":
                        direct = self._native_values(vectors, np.tile(theta, (len(vectors), 1)))
                    else:
                        direct, _ = self._direct(vectors, theta)
                    for i, value in zip(missing, direct):
                        self._direct_cache[plan.keys[i]] = float(value)
                for i, key in enumerate(plan.keys):
                    if key not in self._cache:
                        values[i] = self._direct_cache[key]
        values = np.asarray(values[plan.inverse], dtype=float).copy()
        derivatives = (np.asarray(derivatives[plan.inverse], dtype=float).copy()
                       if gradient else None)
        if not np.isfinite(values).all() or (gradient and not np.isfinite(derivatives).all()):
            raise RuntimeError("Nonfinite prediction or gradient")
        self._derivative_values += len(X)*self._p if gradient else 0
        self._evaluation_s += perf_counter()-started
        return values, derivatives
