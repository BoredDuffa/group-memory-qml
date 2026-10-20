"""Persistent exact responses for the fixed eight-qubit spin classifier.

This is a restricted model-specific response representation, not a general
two-query shift rule for repeated parameters. The preparation and RZZ ring do
not depend on the learned weights. Every weight is a terminal RY gate and the
observable is mean Z, so

    f(x, theta) = sum_j (a_j(x) cos(theta_j) + b_j(x) sin(theta_j)) / n.

Translation symmetry makes the local coefficients identical. Two global probe
settings determine them, and a third off-grid execution checks the result.
The coefficients can then survive weight updates. Untied weights, broken
symmetry and failed checks use the original occurrence-shift implementation.

Both quantum wrappers execute the same probes, verify with the same tolerance,
and cache the same exact input keys. Their only planning difference is a Python
dictionary versus NumPy unique. All probes and validation executions count in
native evaluations and CNOT work. These engines use ideal CPU Aer, not a QPU.
"""

from __future__ import annotations

import numpy as np

from .experiments.derivatives import ShiftEngine, _real_array


_METHODS = ("group_memory_response", "qiskit_response")
_VERIFY_ANGLE = 0.431
_VERIFY_ATOL = 1e-10


def _arguments(X, theta, shifts=None):
    X = _real_array(X, "X")
    theta = _real_array(theta, "theta")
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Incorrect input shape")
    if theta.shape != (8,):
        raise ValueError("Incorrect parameter shape")
    if shifts is None:
        shifts = np.zeros((len(X), 8), dtype=np.int64)
    shifts = np.asarray(shifts)
    if shifts.shape != (len(X), 8) or shifts.dtype.kind not in "iu":
        raise ValueError("shifts must be an integer matrix")
    if np.any(shifts > 16) or np.any(shifts < -16):
        raise ValueError("Shift metadata is bounded to [-16,16]")
    return X, theta, shifts.astype(np.int64)


def _tied(theta):
    words = theta.view(np.dtype((np.void, 8)))
    return bool(np.all(words == words[0]))


class SpinResponseEngine:
    """ShiftEngine-compatible spin responses with conservative fallbacks.

    Forward-only unseen inputs execute directly once and are not proactively
    expanded into three probes. Calling ``derivatives`` learns responses for
    unseen inputs. Coefficients use immutable float-byte keys, with no rounding,
    and remain cached after learned weights change. An input failing the fixed
    verification check is permanently excluded from response reuse.
    """

    family = "spin"
    n_qubits = n_parameters = 8
    input_width = 1

    def __init__(self, method="group_memory_response", broken_symmetry=False):
        if method not in _METHODS:
            raise ValueError("Unknown response method")
        if type(broken_symmetry) is not bool:
            raise ValueError("broken_symmetry must be boolean")
        self.method = method
        self.broken_symmetry = broken_symmetry
        direct_method = "group_memory" if method == _METHODS[0] else "qiskit_symmetry"
        self._direct = ShiftEngine("spin", direct_method, broken_symmetry)
        self.inputs = self._direct.inputs
        self.weights = self._direct.weights
        self.cx_per_evaluation = self._direct.cx_per_evaluation
        self._coefficients = {}
        self._rejected = set()
        self._weight_key = None
        self.requests = 0
        self.invalidations = 0
        self.last_spatial_reuse = False
        self.probe_evaluations = 0
        self.verification_evaluations = 0
        self.response_calls = 0
        self.fallback_calls = 0
        self.direct_forward_calls = 0
        self.verification_failures = 0
        self.max_verification_error = 0.0

    @property
    def source(self):
        return self._direct.source

    @property
    def observable(self):
        return self._direct.observable

    @property
    def evaluations(self):
        return self._direct.evaluations

    @property
    def diagnostics(self):
        return {
            "native_evaluations": self.evaluations,
            "native_cx": self.evaluations * self.cx_per_evaluation,
            "probe_evaluations": self.probe_evaluations,
            "verification_evaluations": self.verification_evaluations,
            "response_calls": self.response_calls,
            "fallback_calls": self.fallback_calls,
            "direct_forward_calls": self.direct_forward_calls,
            "cached_input_responses": len(self._coefficients),
            "rejected_input_responses": len(self._rejected),
            "verification_failures": self.verification_failures,
            "max_response_verification_error": self.max_verification_error,
            "response_verification_atol": _VERIFY_ATOL,
            "native_cache_invalidations": self._direct.invalidations,
            "model_weight_changes": self.invalidations,
            "response_basis": "a(x)*cos(theta)+b(x)*sin(theta); terminal mean-Z",
            "quantum_probe_settings_per_new_derivative_input": 3,
        }

    def _begin(self, theta):
        key = theta.tobytes()
        if self._weight_key is not None and key != self._weight_key:
            self.invalidations += 1
        self._weight_key = key
        self.last_spatial_reuse = not self.broken_symmetry and _tied(theta)
        return self.last_spatial_reuse

    def _unique(self, X):
        if self.method == "group_memory_response":
            unique = {}
            for row in X:
                unique.setdefault(row.tobytes(), row)
            return sorted(unique.items())
        if not len(X):
            return []
        words = X.view(np.dtype((np.void, X.shape[1] * 8))).ravel()
        _, indices = np.unique(words, return_index=True)
        return sorted((X[index].tobytes(), X[index]) for index in indices)

    def _learn(self, X):
        missing = [
            (key, row) for key, row in self._unique(X)
            if key not in self._coefficients and key not in self._rejected
        ]
        if not missing:
            return
        vectors = np.asarray([row for _, row in missing])
        before = self.evaluations
        a = self._direct.evaluate(vectors, np.zeros(8))
        b = self._direct.evaluate(vectors, np.full(8, np.pi / 2))
        self.probe_evaluations += self.evaluations - before
        before = self.evaluations
        check = self._direct.evaluate(vectors, np.full(8, _VERIFY_ANGLE))
        self.verification_evaluations += self.evaluations - before
        predicted = a * np.cos(_VERIFY_ANGLE) + b * np.sin(_VERIFY_ANGLE)
        errors = np.abs(check - predicted)
        for (key, _), ai, bi, error in zip(missing, a, b, errors):
            if np.isfinite(error):
                self.max_verification_error = max(
                    self.max_verification_error, float(error)
                )
            if not np.isfinite(error) or error > _VERIFY_ATOL:
                self._rejected.add(key)
                self.verification_failures += 1
            else:
                self._coefficients[key] = (float(ai), float(bi))

    def evaluate(self, X, theta, shifts=None):
        X, theta, shifts = _arguments(X, theta, shifts)
        if not len(X):
            return np.empty(0)
        eligible = self._begin(theta)
        self.requests += len(X)
        keys = [row.tobytes() for row in X]
        reused = np.asarray([
            eligible and key in self._coefficients for key in keys
        ], dtype=bool)
        values = np.empty(len(X))
        if reused.any():
            coefficients = np.asarray([
                self._coefficients[key] for key, reuse in zip(keys, reused) if reuse
            ])
            # Exact integer shift metadata is reduced modulo the proven period.
            angles = theta + (shifts[reused] % 4) * (np.pi / 2)
            values[reused] = (
                coefficients[:, 0] * np.cos(angles).mean(axis=1)
                + coefficients[:, 1] * np.sin(angles).mean(axis=1)
            )
            self.response_calls += int(reused.sum())
        if (~reused).any():
            values[~reused] = self._direct.evaluate(X[~reused], theta, shifts[~reused])
            rejected = sum(
                key in self._rejected for key, reuse in zip(keys, reused) if not reuse
            )
            self.fallback_calls += len(X[~reused]) if not eligible else rejected
            self.direct_forward_calls += len(X[~reused])
        return values

    def derivatives(self, X, theta, hessian=False):
        X, theta, _ = _arguments(X, theta)
        if type(hessian) is not bool:
            raise ValueError("hessian must be boolean")
        if not len(X):
            curvature = np.empty((0, 8, 8)) if hessian else None
            return np.empty(0), np.empty((0, 8)), curvature
        eligible = self._begin(theta) and not hessian
        request_multiplier = 1 + 2 * 8
        if hessian:
            request_multiplier += 2 * 8 + 4 * (8 * 7 // 2)
        self.requests += len(X) * request_multiplier
        if not eligible:
            self.fallback_calls += len(X)
            return self._direct.derivatives(X, theta, hessian)
        self._learn(X)
        keys = [row.tobytes() for row in X]
        reused = np.asarray([key in self._coefficients for key in keys], dtype=bool)
        predictions = np.empty(len(X))
        gradient = np.empty((len(X), 8))
        if reused.any():
            coefficients = np.asarray([
                self._coefficients[key] for key, reuse in zip(keys, reused) if reuse
            ])
            a, b = coefficients.T
            angle = theta[0]
            predictions[reused] = a * np.cos(angle) + b * np.sin(angle)
            local = (-a * np.sin(angle) + b * np.cos(angle)) / 8
            gradient[reused] = local[:, None]
            self.response_calls += int(reused.sum()) * request_multiplier
        if (~reused).any():
            predictions[~reused], gradient[~reused], _ = self._direct.derivatives(
                X[~reused], theta
            )
            self.fallback_calls += int((~reused).sum())
        return predictions, gradient, None


class AnalyticSpinEngine:
    """Exact classical evaluation of this fixed model, with zero quantum probes.

    The RZZ ring has just two neighbors per measured site. For its product RY
    preparation, a=cos(x) and b=-sin(x)*(cos(phi)^2-cos(x)^2*sin(phi)^2).
    This computes the same quantum model, including gradients and optional
    broken-site RZ, without statevector simulation. It is a dequantization
    control, not an implementation of quantum group-memory acceleration.
    """

    family = "spin"
    method = "classical_analytic"
    n_qubits = n_parameters = 8
    input_width = 1
    evaluations = cx_per_evaluation = 0

    def __init__(self, broken_symmetry=False):
        if type(broken_symmetry) is not bool:
            raise ValueError("broken_symmetry must be boolean")
        self.broken_symmetry = broken_symmetry
        self.requests = 0
        self.invalidations = 0
        self.last_spatial_reuse = False
        self._weight_key = None

    @property
    def diagnostics(self):
        return {
            "native_evaluations": 0, "native_cx": 0, "probe_evaluations": 0,
            "verification_evaluations": 0, "response_calls": self.requests,
            "fallback_calls": 0,
            "response_basis": "closed-form exact classical spin expectation",
        }

    def _begin(self, theta):
        key = theta.tobytes()
        if self._weight_key is not None and key != self._weight_key:
            self.invalidations += 1
        self._weight_key = key
        self.last_spatial_reuse = not self.broken_symmetry and _tied(theta)

    def _coefficients(self, X):
        x = X[:, 0]
        cx, sx = np.cos(x), np.sin(x)
        c, s = np.cos(0.37), np.sin(0.37)
        mean_x = sx * (c * c - cx * cx * s * s)
        a = np.repeat(cx[:, None], 8, axis=1)
        b = np.repeat(-mean_x[:, None], 8, axis=1)
        if self.broken_symmetry:
            mean_y = 2 * sx * cx * c * s
            b[:, 0] = -(np.cos(0.29) * mean_x - np.sin(0.29) * mean_y)
        return a, b

    def evaluate(self, X, theta, shifts=None):
        X, theta, shifts = _arguments(X, theta, shifts)
        self._begin(theta)
        self.requests += len(X)
        a, b = self._coefficients(X)
        angles = theta + (shifts % 4) * (np.pi / 2)
        return (a * np.cos(angles) + b * np.sin(angles)).mean(axis=1)

    def derivatives(self, X, theta, hessian=False):
        X, theta, _ = _arguments(X, theta)
        if type(hessian) is not bool:
            raise ValueError("hessian must be boolean")
        self._begin(theta)
        self.requests += len(X) * 17
        a, b = self._coefficients(X)
        local = a * np.cos(theta) + b * np.sin(theta)
        gradient = (-a * np.sin(theta) + b * np.cos(theta)) / 8
        curvature = None
        if hessian:
            curvature = np.zeros((len(X), 8, 8))
            indices = np.arange(8)
            curvature[:, indices, indices] = -local / 8
        return local.mean(axis=1), gradient, curvature
