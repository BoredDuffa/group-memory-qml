"""Exactly factorized two-qubit quantum features for real sensor training.

With K patches, this model has 2K independently trained angles and 2K total
logical qubits. Its unitary is a tensor product of K two-qubit circuits:

    U(x,theta) = tensor_product_j U_patch(x[2j:2j+2],theta[2j:2j+2]).

Each patch contains noncommuting trainable RY gates interleaved with fixed
RX/RZ/CZ blocks, but there is NO entanglement between patches. The output is K
features, each the mean of its patch's two local Z expectations. A classical
head can combine those features. Gradients returned here are the exact local
blocks of the full, block-diagonal feature Jacobian.

Every method receives this factorization. A single compiled two-qubit engine
and its exact response cache are shared across positions with identical patch
architecture. This is not a fully entangled 8/16-qubit response calculation;
one two-qubit simulator or device can evaluate the patches sequentially.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .deep_response import _block_angles, _fingerprint, _integer, _real
from .multi_response import METHODS, MultiResponseEngine


_ARCHITECTURE = "independent-two-qubit-sensor-patches-v1"


def _source_metadata(total_parameters, occurrences):
    """Full product-circuit signature, without importing Qiskit."""
    operations = []
    for patch in range(total_parameters//2):
        offset = 2*patch
        for site in range(2):
            operations.append(("ry", [offset+site], [f"input[{offset+site}]"]))
        for layer in range(2*occurrences):
            operations.append(("ry", [offset+layer % 2],
                               [f"theta[{offset+layer % 2}]"]))
            rx, rz = _block_angles(layer % 2)
            for site in range(2):
                operations.extend([
                    ("rx", [offset+site], [str(rx)]),
                    ("rz", [offset+site], [str(rz)]),
                ])
            operations.append(("cz", [offset, offset+1], []))
    return _fingerprint({"n": total_parameters, "phase": "0.0", "operations": operations})


class PatchResponseModel:
    """Shared exact patch engine with independently trainable local parameters.

    ``PatchResponseModel(n_parameters, method, occurrences=2)`` supports even
    parameter counts from 2 to 16. Total logical qubits equal that count, while
    each independent circuit uses two qubits and two trainable parameters.

    ``evaluate(X, theta, gradient=False)`` accepts real finite X shaped
    (batch,n_parameters), theta shaped (n_parameters,), and returns features
    (batch,n_parameters//2) plus local Jacobians
    (batch,n_parameters//2,2), or None. Cross-patch derivatives are exactly
    zero and are not stored.

    Parameters and inputs are never rounded or tied across patches. Exact
    cached local response functions may be reused at any patch position
    because the patch circuit and feature observable are identical.
    """

    def __init__(self, n_parameters, method="group_spectral", occurrences=2,
                 memory_budget_bytes=256*1024**2):
        started = perf_counter()
        self._n = _integer(n_parameters, "n_parameters", 2, 16)
        if self._n % 2:
            raise ValueError("n_parameters must be even")
        self._r = _integer(occurrences, "occurrences", 1, 8)
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self._method = method
        self._patches = self._n//2
        self._source_sha = _source_metadata(self._n, self._r)
        self._model_sha = _fingerprint({
            "architecture": _ARCHITECTURE, "source_sha256": self._source_sha,
            "feature": "mean_Z_on_each_independent_two_qubit_patch",
            "local_parameter_mapping": [[2*j, 2*j+1] for j in range(self._patches)],
        })
        self._engine = MultiResponseEngine(
            2, 2, self._r, method=method, memory_budget_bytes=memory_budget_bytes,
        )
        self._source = self._inputs = self._weights = None
        self._evaluations = self._gradient_examples = 0
        self._evaluate_s = 0.
        self._setup_s = perf_counter()-started

    @property
    def n_qubits(self): return self._n
    @property
    def n_parameters(self): return self._n
    @property
    def n_patches(self): return self._patches
    @property
    def patch_qubits(self): return 2
    @property
    def occurrences(self): return self._r
    @property
    def method(self): return self._method

    def _build_source(self):
        if self._source is not None:
            return
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from .deep_response import _circuit_signature
        self._inputs = tuple(ParameterVector("input", self._n))
        self._weights = tuple(ParameterVector("theta", self._n))
        source = QuantumCircuit(self._n)
        patch_source = self._engine.source
        for patch in range(self._patches):
            offset = 2*patch
            bindings = dict(zip(self._engine.inputs, self._inputs[offset:offset+2]))
            bindings.update(zip(self._engine.weights, self._weights[offset:offset+2]))
            source.compose(patch_source.assign_parameters(bindings),
                           qubits=[offset, offset+1], inplace=True)
        if _circuit_signature(source) != self._source_sha:
            raise RuntimeError("Full product-circuit source fingerprint mismatch")
        self._source = source

    @property
    def source(self):
        """A fresh complete product circuit; built lazily for external checks."""
        self._build_source()
        return self._source.copy()

    @property
    def inputs(self):
        self._build_source()
        return self._inputs

    @property
    def weights(self):
        self._build_source()
        return self._weights

    @property
    def diagnostics(self):
        local = self._engine.diagnostics
        return {
            **local,
            "architecture": _ARCHITECTURE,
            "architecture_sha256": self._model_sha,
            "model_sha256": self._model_sha,
            "source_sha256": self._source_sha,
            "patch_source_sha256": local["source_sha256"],
            "patch_native_sha256": local["native_sha256"],
            "native_signature_scope": "One shared two-qubit operational circuit, not a compiled full tensor-product circuit",
            "n_qubits": self._n, "n_parameters": self._n,
            "n_patches": self._patches,
            "patch_qubits": 2, "patch_parameters": 2,
            "parameter_occurrences_per_patch": self._r,
            "trainable_gate_occurrences_per_patch": 2*self._r,
            "trainable_gate_occurrences_total": self._n*self._r,
            "response_grid_points_per_patch_input": (2*self._r+1)**2,
            "response_state_vectors_per_patch_input": (self._r+1)**2,
            "notional_unfactorized_joint_grid_points": (2*self._r+1)**self._n,
            "entanglement_scope": "Two qubits within each patch; no entanglement between patches",
            "factorization_scope": "Exact tensor-product circuit and patch-local observables, supplied identically to every method",
            "cache_scope": "One exact-input local response cache shared across identical patch positions; each patch has independent angles",
            "feature_jacobian_shape": ["batch", self._patches, 2],
            "cross_patch_derivatives": "Exactly zero by tensor-product structure",
            "evaluated_examples": self._evaluations,
            "gradient_examples": self._gradient_examples,
            "setup_seconds": self._setup_s,
            "patch_engine_setup_seconds": local["setup_seconds"],
            "model_evaluate_seconds": self._evaluate_s,
            "compute": "Ideal CPU simulation of independent two-qubit patches; no QPU execution",
        }

    def evaluate(self, X, theta, gradient=False):
        started = perf_counter()
        X, theta = _real(X, "X"), _real(theta, "theta")
        if X.ndim != 2 or X.shape[1] != self._n:
            raise ValueError(f"X must have shape (batch,{self._n})")
        if theta.shape != (self._n,):
            raise ValueError(f"theta must have shape ({self._n},)")
        if not isinstance(gradient, (bool, np.bool_)):
            raise ValueError("gradient must be a boolean")
        features = np.empty((len(X), self._patches))
        jacobian = np.empty((len(X), self._patches, 2)) if gradient else None
        for patch in range(self._patches):
            offset = 2*patch
            values, local = self._engine.evaluate(
                X[:, offset:offset+2], theta[offset:offset+2], gradient=gradient,
            )
            features[:, patch] = values
            if gradient:
                jacobian[:, patch, :] = local
        self._evaluations += len(X)
        self._gradient_examples += len(X) if gradient else 0
        self._evaluate_s += perf_counter()-started
        return features, jacobian
