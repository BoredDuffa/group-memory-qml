"""Exact-input and known-symmetry reuse for a small sensor-circuit family.

This module is a CPU statevector experiment, not a QPU execution interface.
The supported symmetry is supplied by the architecture: uniform gates on a
ring are equivariant under the dihedral group. No symmetry is inferred for an
arbitrary circuit. Input keys retain every float64 bit; there is no rounding,
approximate match, or reuse across different parameter values.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_algorithms.gradients import ReverseEstimatorGradient

METHODS = ("group_memory", "qiskit_same_symmetry", "qiskit_exact_cache")
_MAX_QUBITS = 20
_SEED = 6147
_ARCHITECTURE = "sensor-ring-ry-rzz-rx-ry-v1"


def _validate_n_qubits(n_qubits: int) -> int:
    if (
        isinstance(n_qubits, (bool, np.bool_))
        or not isinstance(n_qubits, (int, np.integer))
        or not 6 <= int(n_qubits) <= _MAX_QUBITS
        or int(n_qubits) % 2
    ):
        raise ValueError("n_qubits must be an even integer between 6 and 20")
    return int(n_qubits)


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _float64_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
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


def _json_hash(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _job_hash(keys: list[bytes]) -> str:
    return _json_hash([key.hex() for key in keys])


def build_sensor_circuit(
    n_qubits: int, symmetry: bool = True
) -> tuple[
    QuantumCircuit,
    tuple[Parameter, ...],
    tuple[Parameter, ...],
    NDArray[np.float64],
]:
    """Build the supported ring circuit and its input/weight parameters.

    Each site has ``RY(input + 0.31)``, each ring edge has ``RZZ(weight[0])``,
    and each site ends with ``RX(gain * weight[1]); RY(0.47)``. There are two
    shared trainable weights. For ``symmetry=True`` all gains are one; otherwise
    site-dependent gains intentionally break ring symmetry.

    The even 6--20-qubit bound is the supported demo range, not a hardware or
    mathematical limit. The returned circuit is a new independently mutable
    object. This function itself does not run or transpile the circuit.
    """
    n_qubits = _validate_n_qubits(n_qubits)
    symmetry = _validate_bool(symmetry, "symmetry")
    inputs = ParameterVector("input", n_qubits)
    weights = ParameterVector("weight", 2)
    circuit = QuantumCircuit(n_qubits)
    gains = np.ones(n_qubits) if symmetry else np.linspace(0.65, 1.35, n_qubits)
    for site in range(n_qubits):
        circuit.ry(inputs[site] + 0.31, site)
    for site in range(n_qubits):
        circuit.rzz(weights[0], site, (site + 1) % n_qubits)
    for site in range(n_qubits):
        circuit.rx(gains[site] * weights[1], site)
    for site in range(n_qubits):
        circuit.ry(0.47, site)
    gains.flags.writeable = False
    return circuit, tuple(inputs), tuple(weights), gains


class QMLReuseEngine:
    """Reuse local-Z predictions and shared-weight mean-Z gradients.

    ``group_memory`` uses a streaming dictionary to deduplicate canonical
    inputs. ``qiskit_same_symmetry`` uses a NumPy unique table with the same
    canonicalization, cache policy, simulator, circuit and gradient algorithm.
    It is an application-aware Qiskit baseline supplied by this package, not
    an automatic symmetry-discovery feature in stock Qiskit.
    ``qiskit_exact_cache`` caches exact input vectors without orbit reduction.

    Correctness contract:

    * With uniform gains, a rotation/reflection of an input permutes its local
      Z outputs in the same way, while mean Z and its shared-weight derivatives
      are invariant. The engine restores the original local-output ordering.
    * With site-dependent gains, orbit reuse is disabled for every method.
    * A bitwise change in either float64 weight invalidates all cached results.
      No approximate-key matching is performed.
    * Public circuit/configuration accessors cannot mutate the private
      architecture. Returned prediction, derivative and metadata arrays are
      copies. The engine is stateful and is not thread-safe.
    * Gradients are classical reverse-mode derivatives of mean Z with respect
      to the two shared weights. They are not parameter-shift circuits or QPU
      jobs. Numerical equality is subject to statevector floating-point error.

    Construction includes fresh level-3 Qiskit compilation. A complete
    benchmark must time construction, canonicalization, simulation, gradients
    and the classical training/readout together.
    """

    def __init__(
        self,
        n_qubits: int,
        method: str = "group_memory",
        symmetry: bool = True,
        gradients: bool = True,
    ) -> None:
        self._n = _validate_n_qubits(n_qubits)
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self._method = method
        self._symmetry = _validate_bool(symmetry, "symmetry")
        self._gradients = _validate_bool(gradients, "gradients")
        self._source, self._inputs, self._weights, self._gains = build_sensor_circuit(
            self._n, self._symmetry
        )
        sites = np.arange(self._n)
        self._permutations = np.asarray(
            [np.roll(sites, shift) for shift in range(self._n)]
            + [np.roll(sites[::-1], shift) for shift in range(self._n)]
        )
        self._group_enabled = self._symmetry and method != "qiskit_exact_cache"
        self._signature = _json_hash(
            {
                "architecture": _ARCHITECTURE,
                "n_qubits": self._n,
                "gains": self._gains.tolist(),
                "observable": "local_Z_and_mean_Z",
                "precision": "float64",
                "basis": ["rz", "sx", "x", "cx"],
                "optimization_level": 3,
                "seed_transpiler": _SEED,
            }
        )
        self._native = transpile(
            self._source,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=3,
            approximation_degree=1.0,
            seed_transpiler=_SEED,
            num_processes=1,
        )
        if self._native.num_qubits != self._n or self._native.layout is not None:
            raise RuntimeError("Unexpected transpiler layout; output restoration is unsafe")
        self._gates = dict(self._native.count_ops())
        self._forward = self._native.copy()
        for site in range(self._n):
            observable = SparsePauliOp.from_sparse_list([("Z", [site], 1.0)], self._n)
            self._forward.save_expectation_value(
                observable, range(self._n), label=f"z{site}"
            )
        self._observable = SparsePauliOp.from_sparse_list(
            [("Z", [site], 1.0 / self._n) for site in range(self._n)], self._n
        )
        self._backend = AerSimulator(
            method="statevector",
            device="CPU",
            precision="double",
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )
        self._reverse = ReverseEstimatorGradient() if self._gradients else None
        self._parameters = tuple(self._native.parameters)
        self._cache: dict[bytes, tuple[NDArray[np.float64], NDArray[np.float64] | None]] = {}
        self._weight_key: bytes | None = None
        self._forward_count = 0
        self._gradient_count = 0
        self._invalidations = 0
        self._requests = 0
        self._calls: list[dict[str, Any]] = []

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def n(self) -> int:
        """Number of sites; an alias of ``n_qubits``."""
        return self._n

    @property
    def method(self) -> str:
        return self._method

    @property
    def symmetry(self) -> bool:
        return self._symmetry

    @property
    def group_enabled(self) -> bool:
        return self._group_enabled

    @property
    def source(self) -> QuantumCircuit:
        """An independent copy of the uncompiled circuit."""
        return self._source.copy()

    @property
    def native(self) -> QuantumCircuit:
        """An independent copy of the compiled circuit, without save instructions."""
        return self._native.copy()

    @property
    def x(self) -> tuple[Parameter, ...]:
        return self._inputs

    @property
    def theta(self) -> tuple[Parameter, ...]:
        return self._weights

    @property
    def gains(self) -> NDArray[np.float64]:
        return self._gains.copy()

    @property
    def gates(self) -> dict[str, int]:
        """Native gate counts per forward circuit; excludes gradient work."""
        return self._gates.copy()

    @property
    def signature(self) -> str:
        """Identity of the supported architecture and compilation configuration."""
        return self._signature

    @property
    def forward_count(self) -> int:
        """Number of explicitly simulated forward circuit instances."""
        return self._forward_count

    @property
    def gradient_count(self) -> int:
        """Number of classical reverse-gradient instances, each with two derivatives."""
        return self._gradient_count

    @property
    def invalidations(self) -> int:
        return self._invalidations

    @property
    def requests(self) -> int:
        """Number of requested examples across successful nonempty calls."""
        return self._requests

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Independent copy of per-call work accounting."""
        return deepcopy(self._calls)

    def _canonicalize(
        self, inputs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], list[bytes]]:
        canonical = []
        permutations = []
        keys = []
        for vector in inputs:
            if self._group_enabled:
                orbit = np.ascontiguousarray(vector[self._permutations])
                candidates = [row.tobytes() for row in orbit]
                index = min(range(len(candidates)), key=candidates.__getitem__)
                canonical.append(orbit[index])
                permutations.append(self._permutations[index])
                keys.append(candidates[index])
            else:
                canonical.append(vector)
                permutations.append(np.arange(self._n))
                keys.append(vector.tobytes())
        return np.asarray(canonical), np.asarray(permutations), keys

    def _bindings(
        self, keys: list[bytes], unique: dict[bytes, NDArray[np.float64]], weights: NDArray[np.float64]
    ) -> list[dict[Parameter, float]]:
        bindings = []
        for key in keys:
            binding = dict(zip(self._inputs, unique[key]))
            binding.update(zip(self._weights, weights))
            bindings.append(binding)
        return bindings

    def evaluate(
        self, X: ArrayLike, theta: ArrayLike, gradient: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Evaluate local Z and optionally the two shared-weight mean-Z derivatives.

        ``X`` must be a finite real matrix of shape ``(batch, n_qubits)`` and
        ``theta`` a finite real vector of shape ``(2,)``. Values are converted
        to float64 before exact keys are constructed. An empty, correctly shaped
        batch is a no-op: it returns ``(0, n_qubits)`` and optionally ``(0, 2)``
        arrays without changing the cache, weights or counters.

        A forward-only cached result can be upgraded to include its gradient
        without repeating the separately counted forward simulation. The
        reverse-gradient algorithm itself still performs its own state
        propagation, which is included in its runtime.
        """
        inputs = _float64_array(X, "X")
        weights = _float64_array(theta, "theta")
        gradient = _validate_bool(gradient, "gradient")
        if inputs.ndim != 2 or inputs.shape[1] != self._n:
            raise ValueError(f"X must have shape (batch, {self._n})")
        if weights.shape != (2,):
            raise ValueError("theta must have shape (2,)")
        if gradient and self._reverse is None:
            raise ValueError("Gradient evaluation requires gradients=True at construction")
        if len(inputs) == 0:
            return np.empty((0, self._n)), np.empty((0, 2)) if gradient else None

        weight_key = weights.tobytes()
        if self._weight_key != weight_key:
            if self._weight_key is not None:
                self._invalidations += 1
            self._cache.clear()
            self._weight_key = weight_key

        canonical, permutations, keys = self._canonicalize(inputs)
        if self._method == "group_memory":
            unique = {}
            for key, vector in zip(keys, canonical):
                unique.setdefault(key, vector)
        else:
            packed = np.ascontiguousarray(canonical).view(
                np.dtype((np.void, self._n * np.dtype(np.float64).itemsize))
            ).ravel()
            _, indices = np.unique(packed, return_index=True)
            unique = {keys[index]: canonical[index] for index in indices}

        forward_keys = sorted(key for key in unique if key not in self._cache)
        gradient_keys = (
            sorted(
                key
                for key in unique
                if key not in self._cache or self._cache[key][1] is None
            )
            if gradient
            else []
        )
        if forward_keys:
            bindings = self._bindings(forward_keys, unique, weights)
            circuits = [self._forward.assign_parameters(binding) for binding in bindings]
            result = self._backend.run(circuits, shots=None, seed_simulator=_SEED).result()
            if not result.success:
                raise RuntimeError(f"Statevector evaluation failed: {result.status}")
            values = np.asarray(
                [
                    [float(np.real(result.data(index)[f"z{site}"])) for site in range(self._n)]
                    for index in range(len(forward_keys))
                ]
            )
            if not np.isfinite(values).all():
                raise RuntimeError("Statevector evaluation produced nonfinite outputs")
            for key, value in zip(forward_keys, values):
                self._cache[key] = (value.copy(), None)
            self._forward_count += len(forward_keys)

        if gradient_keys:
            bindings = self._bindings(gradient_keys, unique, weights)
            parameter_values = [
                [float(binding[parameter]) for parameter in self._parameters]
                for binding in bindings
            ]
            jacobian = np.asarray(
                self._reverse.run(
                    [self._native] * len(gradient_keys),
                    [self._observable] * len(gradient_keys),
                    parameter_values=parameter_values,
                    parameters=[list(self._weights)] * len(gradient_keys),
                ).result().gradients,
                dtype=np.float64,
            )
            if jacobian.shape != (len(gradient_keys), 2) or not np.isfinite(jacobian).all():
                raise RuntimeError("Gradient evaluation produced invalid outputs")
            for key, derivative in zip(gradient_keys, jacobian):
                self._cache[key] = (self._cache[key][0], derivative.copy())
            self._gradient_count += len(gradient_keys)

        restored = np.asarray(
            [self._cache[key][0][np.argsort(perm)] for key, perm in zip(keys, permutations)]
        )
        derivatives = (
            np.asarray([self._cache[key][1] for key in keys]) if gradient else None
        )
        self._requests += len(inputs)
        self._calls.append(
            {
                "requested": len(inputs),
                "new_forward_states": len(forward_keys),
                "reverse_gradient_instances": len(gradient_keys),
                "canonical_jobs_sha256": _job_hash(sorted(set(forward_keys + gradient_keys))),
                "forward_jobs_sha256": _job_hash(forward_keys),
                "gradient_jobs_sha256": _job_hash(gradient_keys),
                "weights_sha256": sha256(weight_key).hexdigest(),
                "gradient": gradient,
            }
        )
        return restored, derivatives
