"""Exact, circuit-calibrated parameter-response reuse for the sensor ring.

This is CPU simulation, not quantum hardware or an approximate surrogate. For
uniform gains the local-Z response belongs to the four-dimensional span
``cos(beta), sin(beta)*sin(2*gamma), 1, cos(2*gamma)``. Four freshly compiled
Qiskit/Aer probe instances identify its coefficients; a fifth off-grid probe
checks each input orbit at the unchanged absolute tolerance 1e-10. The basis
follows from conjugating local observables through this particular shallow
circuit. A numerical probe alone would not certify an arbitrary circuit.

Both quantum modes receive exactly the same basis, probes, input plans, cache
policy and backend. They differ only in dictionary versus NumPy deduplication.
The explicitly classical control evaluates the same model analytically.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Keep the analytic control free of quantum-library import overhead. Validation
# is deliberately lightweight here; the quantum constructor also delegates to
# the original engine's architecture validation.
def _validate_n_qubits(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 6 <= int(value) <= 20 or int(value) % 2):
        raise ValueError("n_qubits must be an even integer between 6 and 20")
    return int(value)


def _validate_bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _float64_array(value, name):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError("expected real numeric values")
        array = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numeric values") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array

METHODS = ("group_memory_response", "qiskit_response")
_PROBES = np.array([[0., 0.], [0., np.pi], [np.pi/2, np.pi/2],
                    [np.pi/4, np.pi/2], [.317, -.583]])
_PROBES.flags.writeable = False
_TOL = 1e-10


def _response(coefficients, theta, gradient):
    """Evaluate exact local responses and mean-local-Z shared derivatives."""
    gamma, beta = theta
    a, b, c, d = np.moveaxis(coefficients, -2, 0)
    sg, cg = np.sin(2*gamma), np.cos(2*gamma)
    sb, cb = np.sin(beta), np.cos(beta)
    z = a*cb + b*sb*sg + c + d*cg
    jac = (np.stack([(2*b*sb*cg-2*d*sg).mean(axis=-1),
                     (-a*sb+b*cb*sg).mean(axis=-1)], axis=-1)
           if gradient else None)
    return z, jac


@dataclass(frozen=True)
class _Plan:
    keys: tuple[bytes, ...]
    vectors: NDArray[np.float64]
    inverse: NDArray[np.int64]
    restore: NDArray[np.int64]


class SensorResponseEngine:
    """Persistent exact parameter functions learned from native probe circuits.

    ``gradient=True`` calibrates each previously unseen input orbit once. Later
    weights use the cached functions, including exact derivatives. An unseen
    forward-only input receives a single direct simulation instead of five
    calibration probes. Unsupported nonuniform-gain architectures delegate to
    the original engine without response reuse. All construction, preparation,
    calibration, validation and evaluation must be included in runtime totals.

    ``gradient_count`` counts reverse-gradient jobs (zero for the supported
    response basis); ``diagnostics()`` separately counts function derivatives.
    ``forward_count`` includes all four fit probes, the validation probe, and
    direct forward instances, including probes rejected by validation.
    """
    def __init__(self, n_qubits: int, method: str = "group_memory_response",
                 symmetry: bool = True, gradients: bool = True):
        started = perf_counter()
        self._n = _validate_n_qubits(n_qubits)
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self._method = method
        self._symmetry = _validate_bool(symmetry, "symmetry")
        self._gradients = _validate_bool(gradients, "gradients")
        mapped = "group_memory" if method == METHODS[0] else "qiskit_same_symmetry"
        from .engine import QMLReuseEngine
        self._engine = QMLReuseEngine(self._n, mapped, self._symmetry,
                                     gradients=self._gradients and not self._symmetry)
        self._coefficients: dict[bytes, NDArray[np.float64]] = {}
        self._plans: dict[tuple[tuple[int, ...], bytes], _Plan] = {}
        self._direct: dict[bytes, NDArray[np.float64]] = {}
        self._weight_key: bytes | None = None
        self._forward_count = self._requests = self._response_derivatives = 0
        self._direct_invalidations = self._plan_hits = self._probe_failures = 0
        self._calibration_seconds = self._planning_seconds = self._response_seconds = 0.
        self._max_probe_error = 0.
        self._setup_seconds = perf_counter()-started

    @property
    def n(self): return self._n
    @property
    def n_qubits(self): return self._n
    @property
    def method(self): return self._method
    @property
    def symmetry(self): return self._symmetry
    @property
    def group_enabled(self): return self._engine.group_enabled
    @property
    def forward_count(self):
        return self._forward_count if self._symmetry else self._engine.forward_count
    @property
    def gradient_count(self):
        return 0 if self._symmetry else self._engine.gradient_count
    @property
    def invalidations(self):
        """Coefficient invalidations; ordinary fallback follows its old policy."""
        return 0 if self._symmetry else self._engine.invalidations
    @property
    def requests(self):
        return self._requests if self._symmetry else self._engine.requests
    @property
    def gates(self): return self._engine.gates
    @property
    def source(self): return self._engine.source
    @property
    def native(self): return self._engine.native
    @property
    def gains(self): return self._engine.gains
    @property
    def signature(self): return self._engine.signature

    def _prepare(self, inputs):
        started = perf_counter()
        identity = (inputs.shape, inputs.tobytes())
        if identity in self._plans:
            self._plan_hits += 1
            self._planning_seconds += perf_counter()-started
            return self._plans[identity]
        canonical, permutations, keys = self._engine._canonicalize(inputs)
        if self._method == METHODS[0]:
            unique = {}
            for key, row in zip(keys, canonical):
                unique.setdefault(key, row)
        else:
            packed = np.ascontiguousarray(canonical).view(
                np.dtype((np.void, self._n*8))).ravel()
            _, indices = np.unique(packed, return_index=True)
            unique = {keys[i]: canonical[i] for i in indices}
        ordered = tuple(sorted(unique))
        indices = {key: i for i, key in enumerate(ordered)}
        vectors = np.array([unique[key] for key in ordered], copy=True)
        inverse = np.array([indices[key] for key in keys])
        restore = np.argsort(permutations, axis=1)
        for array in (vectors, inverse, restore):
            array.flags.writeable = False
        plan = _Plan(ordered, vectors, inverse, restore)
        self._plans[identity] = plan
        self._planning_seconds += perf_counter()-started
        return plan

    def _simulate(self, vectors, weights):
        circuits = []
        for vector, theta in zip(vectors, weights):
            binding = dict(zip(self._engine._inputs, vector))
            binding.update(zip(self._engine._weights, theta))
            circuits.append(self._engine._forward.assign_parameters(binding))
        result = self._engine._backend.run(circuits, shots=None,
                                          seed_simulator=6147).result()
        self._forward_count += len(circuits)
        if not result.success:
            raise RuntimeError(f"Response probe simulation failed: {result.status}")
        values = np.array([[float(np.real(result.data(i)[f"z{q}"]))
                            for q in range(self._n)] for i in range(len(circuits))])
        if not np.isfinite(values).all():
            raise RuntimeError("Response probe simulation produced nonfinite outputs")
        return values

    def _calibrate(self, keys, vectors):
        if not keys:
            return
        started = perf_counter()
        try:
            values = self._simulate(np.repeat(vectors, 5, axis=0),
                                    np.tile(_PROBES, (len(keys), 1)))
            values = values.reshape(len(keys), 5, self._n)
            p0, p1, p2, p3, check = np.moveaxis(values, 1, 0)
            a = (p0-p1)/2
            c = ((p0+p1)/2+p2)/2
            d = ((p0+p1)/2-p2)/2
            b = p3-c
            coefficients = np.stack([a, b, c, d], axis=1)
            prediction, _ = _response(coefficients, _PROBES[-1], False)
            error = float(np.max(np.abs(prediction-check)))
            self._max_probe_error = max(self._max_probe_error, error)
            if not np.isfinite(error) or error > _TOL:
                self._probe_failures += 1
                raise RuntimeError(f"Parameter-response verification failed: {error:.17g} > {_TOL}")
            # Transactional: no coefficients survive a failed batch check.
            for key, coeff in zip(keys, coefficients):
                self._coefficients[key] = coeff.copy()
        finally:
            self._calibration_seconds += perf_counter()-started

    def evaluate(self, X: ArrayLike, theta: ArrayLike, gradient: bool = False):
        if not self._symmetry:
            return self._engine.evaluate(X, theta, gradient)
        inputs, weights = _float64_array(X, "X"), _float64_array(theta, "theta")
        gradient = _validate_bool(gradient, "gradient")
        if inputs.ndim != 2 or inputs.shape[1] != self._n:
            raise ValueError(f"X must have shape (batch, {self._n})")
        if weights.shape != (2,):
            raise ValueError("theta must have shape (2,)")
        if gradient and not self._gradients:
            raise ValueError("Gradient evaluation requires gradients=True at construction")
        if not len(inputs):
            return np.empty((0, self._n)), np.empty((0, 2)) if gradient else None
        plan = self._prepare(inputs)
        weight_key = weights.tobytes()
        if weight_key != self._weight_key:
            self._direct_invalidations += int(self._weight_key is not None)
            self._direct.clear()
            self._weight_key = weight_key
        if gradient:
            missing = [i for i, key in enumerate(plan.keys) if key not in self._coefficients]
            self._calibrate([plan.keys[i] for i in missing], plan.vectors[missing])
        direct = [i for i, key in enumerate(plan.keys)
                  if key not in self._coefficients and key not in self._direct]
        if direct:
            values = self._simulate(plan.vectors[direct], np.tile(weights, (len(direct), 1)))
            for i, value in zip(direct, values):
                self._direct[plan.keys[i]] = value.copy()
        started = perf_counter()
        response_ids = [i for i, key in enumerate(plan.keys) if key in self._coefficients]
        z = np.empty((len(plan.keys), self._n))
        jac = np.empty((len(plan.keys), 2)) if gradient else None
        if response_ids:
            coefficients = np.stack([self._coefficients[plan.keys[i]] for i in response_ids])
            zr, jr = _response(coefficients, weights, gradient)
            z[response_ids] = zr
            if gradient:
                jac[response_ids] = jr
                self._response_derivatives += len(response_ids)
        for i, key in enumerate(plan.keys):
            if key not in self._coefficients:
                z[i] = self._direct[key]
        restored = np.take_along_axis(z[plan.inverse], plan.restore, axis=1)
        derivatives = jac[plan.inverse].copy() if gradient else None
        self._response_seconds += perf_counter()-started
        self._requests += len(inputs)
        return restored, derivatives

    def diagnostics(self) -> dict[str, Any]:
        return {"response_supported": self._symmetry,
                "response_algorithm": "sensor-four-term-native-probes-v1",
                "architecture_signature": self.signature,
                "response_coefficients_from": "five native Qiskit/Aer probes per input orbit",
                "basis": ["cos(beta)", "sin(beta)*sin(2*gamma)", "1", "cos(2*gamma)"],
                "response_orbits": len(self._coefficients),
                "response_derivative_instances": self._response_derivatives,
                "plan_count": len(self._plans), "plan_cache_hits": self._plan_hits,
                "direct_cache_invalidations": self._direct_invalidations,
                "probe_failures": self._probe_failures,
                "max_probe_abs_error": self._max_probe_error,
                "probe_tolerance": _TOL, "setup_seconds": self._setup_seconds,
                "planning_seconds": self._planning_seconds,
                "calibration_seconds": self._calibration_seconds,
                "response_evaluation_seconds": self._response_seconds,
                "forward_count_includes_validation_probes": True,
                "compute": "Ideal CPU simulation and exact parameter-response arithmetic; no QPU"}


class ClassicalSensorEngine:
    """Analytic classical evaluation of the identical shallow sensor model.

    The known local-observable formula has a constant-width light cone. This
    control performs no Qiskit compilation, statevector simulation or probes.
    It also supports nonuniform gains, where shared-beta derivatives acquire
    each site's gain. It provides the same ``evaluate`` contract.
    """
    def __init__(self, n_qubits, method="classical_same_model", symmetry=True,
                 gradients=True):
        self._n = _validate_n_qubits(n_qubits)
        self._symmetry = _validate_bool(symmetry, "symmetry")
        self._gradients = _validate_bool(gradients, "gradients")
        self._gains = np.ones(self._n) if self._symmetry else np.linspace(.65, 1.35, self._n)
        self._plans = {}
        self._requests = 0
    @property
    def n(self): return self._n
    @property
    def n_qubits(self): return self._n
    @property
    def group_enabled(self): return False
    @property
    def forward_count(self): return 0
    @property
    def gradient_count(self): return 0
    @property
    def invalidations(self): return 0
    @property
    def requests(self): return self._requests
    @property
    def gates(self): return {}
    def evaluate(self, X, theta, gradient=False):
        inputs, weights = _float64_array(X, "X"), _float64_array(theta, "theta")
        gradient = _validate_bool(gradient, "gradient")
        if inputs.ndim != 2 or inputs.shape[1] != self._n:
            raise ValueError(f"X must have shape (batch, {self._n})")
        if weights.shape != (2,):
            raise ValueError("theta must have shape (2,)")
        if gradient and not self._gradients:
            raise ValueError("Gradient evaluation requires gradients=True at construction")
        if not len(inputs):
            return np.empty((0, self._n)), np.empty((0, 2)) if gradient else None
        identity = (inputs.shape, inputs.tobytes())
        if identity not in self._plans:
            si, ci = np.sin(inputs+.31), np.cos(inputs+.31)
            left, right = np.roll(ci, 1, axis=1), np.roll(ci, -1, axis=1)
            a = np.cos(.47)*ci
            b = .5*np.cos(.47)*si*(left+right)
            c = -.5*np.sin(.47)*si*(1-left*right)
            d = -.5*np.sin(.47)*si*(1+left*right)
            self._plans[identity] = np.stack([a,b,c,d],axis=1)
        coeff = self._plans[identity]
        if self._symmetry:
            z, jac = _response(coeff, weights, gradient)
        else:
            gamma, beta = weights
            a,b,c,d = np.moveaxis(coeff,1,0)
            cb, sb = np.cos(beta*self._gains), np.sin(beta*self._gains)
            cg, sg = np.cos(2*gamma), np.sin(2*gamma)
            z = a*cb+b*sb*sg+c+d*cg
            jac = (np.stack([(2*b*sb*cg-2*d*sg).mean(axis=-1),
                             ((-a*sb+b*cb*sg)*self._gains).mean(axis=-1)],axis=-1)
                   if gradient else None)
        self._requests += len(inputs)
        return z.copy(), jac.copy() if gradient else None
    def diagnostics(self):
        return {"compute": "Exact classical analytic evaluation of the identical model",
                "quantum_evaluations": 0, "plan_count": len(self._plans)}


# Explicit analytic spelling used by the benchmark harness.
AnalyticSensorEngine = ClassicalSensorEngine
