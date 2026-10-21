"""Small exact-shift derivative experiments; all execution is ideal CPU Aer.

2A differentiates individual gate occurrences, then sums their derivatives for
one tied weight. 2B differentiates four single-occurrence Pauli rotations and
uses an exact parameter-shift Hessian. Neither applies the usual two-query
shift rule directly to a parameter occurring in several gates.

The cache's periodic equivalence uses integer shift metadata, never rounding
nearby parameter values. Quantum methods share circuits and derivative rules.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

METHODS = ("group_memory", "qiskit_symmetry", "qiskit_exact", "classical")


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


class ShiftEngine:
    """Evaluate a fixed architecture using exact inputs and integer shift keys.

    ``spin`` has translation/reflection-invariant preparation and mean-Z
    readout. Spatial shift reuse is enabled only when all occurrence weights
    are bitwise identical and ``broken_symmetry`` is false. ``curvature`` has
    four single-occurrence rotations and only their proven 2*pi expectation
    periodicity. The equally informed Qiskit wrapper uses vectorized unique
    planning; the GroupMemory wrapper uses a dictionary.
    """

    def __init__(self, family, method="group_memory", broken_symmetry=False):
        if family not in ("spin", "curvature"):
            raise ValueError("family must be spin or curvature")
        if method not in METHODS[:-1]:
            raise ValueError("Unknown quantum method")
        if not isinstance(broken_symmetry, bool):
            raise ValueError("broken_symmetry must be boolean")
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator

        self.family = family
        self.method = method
        self.broken_symmetry = broken_symmetry
        self.n_qubits = 8 if family == "spin" else 6
        self.n_parameters = 8 if family == "spin" else 4
        self.input_width = 1 if family == "spin" else 6
        self.inputs = tuple(ParameterVector("x", self.input_width))
        self.weights = tuple(ParameterVector("theta", self.n_parameters))
        source = QuantumCircuit(self.n_qubits)
        if family == "spin":
            for site in range(self.n_qubits):
                source.ry(self.inputs[0], site)
            for site in range(self.n_qubits):
                source.rzz(0.37, site, (site + 1) % self.n_qubits)
            if broken_symmetry:
                source.rz(0.29, 0)
            for site, weight in enumerate(self.weights):
                source.ry(weight, site)
            observable = SparsePauliOp.from_sparse_list(
                [("Z", [site], 1 / self.n_qubits) for site in range(self.n_qubits)],
                self.n_qubits,
            )
        else:
            for site, angle in enumerate(self.inputs):
                source.ry(angle, site)
            for site in range(self.n_qubits - 1):
                source.cx(site, site + 1)
            source.ry(self.weights[0], 0)
            source.cx(0, 1)
            source.rx(self.weights[1], 1)
            source.cx(1, 0)
            source.rzz(self.weights[2], 0, 2)
            source.ry(self.weights[3], 0)
            if broken_symmetry:
                # Does not invalidate 2*pi periodicity of single-occurrence gates.
                source.rx(0.29, 0)
            observable = SparsePauliOp.from_sparse_list(
                [("Z", [0], 1.0)], self.n_qubits
            )
        self._source = source
        self._observable = observable
        self._native = transpile(
            source, basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=3, approximation_degree=1.0,
            seed_transpiler=6147, num_processes=1,
        )
        if self._native.layout is not None:
            raise RuntimeError("Unexpected layout in derivative experiment")
        self.cx_per_evaluation = int(self._native.count_ops().get("cx", 0))
        self._execution = self._native.copy()
        self._execution.save_expectation_value(
            observable, range(self.n_qubits), label="prediction"
        )
        self._backend = AerSimulator(
            method="statevector", precision="double", device="CPU",
            max_parallel_threads=1, max_parallel_experiments=1,
            max_parallel_shots=1,
        )
        self._cache = {}
        self._weight_key = None
        self.evaluations = 0
        self.requests = 0
        self.invalidations = 0
        self.last_spatial_reuse = False
        self._permutations = [
            np.roll(np.arange(self.n_parameters), shift)
            for shift in range(self.n_parameters)
        ] + [
            np.roll(np.arange(self.n_parameters)[::-1], shift)
            for shift in range(self.n_parameters)
        ]

    @property
    def source(self):
        return self._source.copy()

    @property
    def observable(self):
        return self._observable.copy()

    def evaluate(self, X, theta, shifts=None):
        """Return expectations for paired rows ``X[i], theta + pi/2*shifts[i]``."""
        X = _real_array(X, "X")
        theta = _real_array(theta, "theta")
        if X.ndim != 2 or X.shape[1] != self.input_width:
            raise ValueError("Incorrect input shape")
        if theta.shape != (self.n_parameters,):
            raise ValueError("Incorrect parameter shape")
        if shifts is None:
            shifts = np.zeros((len(X), self.n_parameters), dtype=np.int64)
        shifts = np.asarray(shifts)
        if shifts.shape != (len(X), self.n_parameters) or shifts.dtype.kind not in "iu":
            raise ValueError("shifts must be an integer matrix")
        if np.any(shifts > 16) or np.any(shifts < -16):
            raise ValueError("Shift metadata is bounded to [-16,16]")
        shifts = shifts.astype(np.int64)
        if not len(X):
            return np.empty(0)

        weight_key = theta.tobytes()
        if self._weight_key != weight_key:
            if self._weight_key is not None:
                self.invalidations += 1
            self._cache.clear()
            self._weight_key = weight_key
        # Equality is bitwise, including the distinction between +0 and -0.
        weight_bytes = theta.view(np.dtype((np.void, 8)))
        tied = bool(np.all(weight_bytes == weight_bytes[0]))
        self.last_spatial_reuse = (
            self.family == "spin" and tied and not self.broken_symmetry
            and self.method != "qiskit_exact"
        )
        keys, jobs = [], []
        for vector, shift in zip(X, shifts):
            if self.method == "qiskit_exact":
                angles = theta + shift * (np.pi / 2)
            else:
                reduced = shift % 4
                if self.last_spatial_reuse:
                    reduced = np.asarray(min(
                        tuple(int(v) for v in reduced[permutation])
                        for permutation in self._permutations
                    ))
                angles = theta + reduced * (np.pi / 2)
            key = vector.tobytes() + angles.tobytes()
            keys.append(key)
            jobs.append((vector, angles))
        if self.method == "group_memory":
            unique = {}
            for key, job in zip(keys, jobs):
                unique.setdefault(key, job)
        else:
            packed = np.asarray(keys, dtype=np.dtype((np.void, len(keys[0]))))
            _, indices = np.unique(packed, return_index=True)
            unique = {keys[index]: jobs[index] for index in indices}
        missing = sorted(key for key in unique if key not in self._cache)
        if missing:
            circuits = []
            for key in missing:
                vector, angles = unique[key]
                binding = dict(zip(self.inputs, vector))
                binding.update(zip(self.weights, angles))
                circuits.append(self._execution.assign_parameters(binding))
            result = self._backend.run(circuits, shots=None, seed_simulator=6147).result()
            if not result.success:
                raise RuntimeError(f"Aer derivative execution failed: {result.status}")
            values = [
                float(np.real(result.data(i)["prediction"])) for i in range(len(missing))
            ]
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite derivative expectation")
            self._cache.update(zip(missing, values))
            self.evaluations += len(missing)
        self.requests += len(X)
        return np.asarray([self._cache[key] for key in keys])

    def derivatives(self, X, theta, hessian=False):
        """Exact occurrence-wise gradient and optional single-occurrence Hessian."""
        X = _real_array(X, "X")
        p = self.n_parameters
        zero = np.zeros(p, dtype=np.int64)
        shifts = [zero]
        gradient_indices = []
        diagonal_indices = []
        mixed_indices = []
        for j in range(p):
            unit = zero.copy()
            unit[j] = 1
            gradient_indices.append((len(shifts), len(shifts) + 1))
            shifts.extend([unit, -unit])
        if hessian:
            for j in range(p):
                unit = zero.copy()
                unit[j] = 2
                diagonal_indices.append((len(shifts), len(shifts) + 1))
                shifts.extend([unit, -unit])
            for j in range(p):
                for k in range(j + 1, p):
                    start = len(shifts)
                    for sj, sk in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        shift = zero.copy()
                        shift[j], shift[k] = sj, sk
                        shifts.append(shift)
                    mixed_indices.append((j, k, start))
        values = self.evaluate(
            np.repeat(X, len(shifts), axis=0), theta,
            np.tile(np.asarray(shifts), (len(X), 1)),
        ).reshape(len(X), len(shifts))
        predictions = values[:, 0]
        gradient = np.asarray([
            (values[:, plus] - values[:, minus]) / 2
            for plus, minus in gradient_indices
        ]).T
        curvature = None
        if hessian:
            curvature = np.zeros((len(X), p, p))
            for j, (plus, minus) in enumerate(diagonal_indices):
                curvature[:, j, j] = (
                    values[:, plus] + values[:, minus] - 2 * predictions
                ) / 4
            for j, k, start in mixed_indices:
                mixed = (
                    values[:, start] - values[:, start + 1]
                    - values[:, start + 2] + values[:, start + 3]
                ) / 4
                curvature[:, j, k] = curvature[:, k, j] = mixed
        return predictions, gradient, curvature


def make_data(experiment, seed, n_train=12, n_test=12):
    """Independent base preparations; no augmented views or train/test reuse."""
    rng = np.random.default_rng(seed)
    if experiment == "2A":
        def split(count):
            labels = np.repeat([0, 1], count // 2)
            angles = np.where(
                labels, rng.uniform(0.15, 0.75, count), rng.uniform(1.3, 1.9, count)
            )
            order = rng.permutation(count)
            return angles[order, None], labels[order].astype(float)
    elif experiment == "2B":
        def split(count):
            X = rng.uniform(-0.8, 0.8, (count, 6))
            target = (
                0.4 * np.cos(X[:, 0]) - 0.3 * np.sin(X[:, 1])
                + 0.2 * np.sin(X[:, 2]) * np.cos(X[:, 3])
            )
            return X, target
    else:
        raise ValueError("experiment must be 2A or 2B")
    return (*split(n_train), *split(n_test))


def _classification_loss(values, labels, bias):
    logits = 4 * values + bias
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    loss = float(np.mean(np.logaddexp(0, logits) - labels * logits))
    return probabilities, loss


def _classical(experiment, train, labels, test, target, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    if experiment == "2A":
        features = lambda X: np.concatenate([X, np.sin(X), np.cos(X)], axis=1)
        scaler = StandardScaler().fit(features(train))
        model = LogisticRegression(C=10, max_iter=200, random_state=seed).fit(
            scaler.transform(features(train)), labels
        )
        predictions = model.predict_proba(scaler.transform(features(test)))[:, 1]
        train_predictions = model.predict_proba(scaler.transform(features(train)))[:, 1]
        metric = float(np.mean((predictions >= 0.5) == target))
        loss = float(-np.mean(
            labels * np.log(np.clip(train_predictions, 1e-15, 1))
            + (1 - labels) * np.log(np.clip(1 - train_predictions, 1e-15, 1))
        ))
        name = "logistic regression on preparation angle and sin/cos features"
    else:
        model = KernelRidge(alpha=1e-3, kernel="rbf", gamma=0.25).fit(train, labels)
        predictions = model.predict(test)
        loss = float(np.mean((model.predict(train) - labels) ** 2))
        metric = float(np.mean((predictions - target) ** 2))
        name = "RBF kernel ridge on the six classical input angles"
    return predictions, metric, loss, name


def run(experiment: str, seed: int, method: str, config: dict | None = None) -> dict:
    """Run 2A (8-qubit spin classifier) or 2B (6-qubit Hessian regression).

    Configuration supports ``steps`` (default 3), ``n_train``/``n_test`` (12),
    and ``broken_symmetry`` (false). Parent harness times imports, this entire
    function, serialization and process overhead. Nothing runs on import.
    """
    if experiment not in ("2A", "2B") or method not in METHODS:
        raise ValueError("Unsupported experiment or method")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cfg = {"steps": 3, "n_train": 12, "n_test": 12, "broken_symmetry": False}
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        cfg.update(config)
    if type(cfg["steps"]) is not int or not 1 <= cfg["steps"] <= 8:
        raise ValueError("steps must be an integer from 1 through 8")
    for key in ("n_train", "n_test"):
        if type(cfg[key]) is not int or not 4 <= cfg[key] <= 64 or cfg[key] % 2:
            raise ValueError(f"{key} must be an even integer from 4 through 64")
    if type(cfg["broken_symmetry"]) is not bool:
        raise ValueError("broken_symmetry must be boolean")
    X, y, test, target = make_data(
        experiment, int(seed), cfg["n_train"], cfg["n_test"]
    )
    data_hash = hashlib.sha256(
        X.tobytes() + y.tobytes() + test.tobytes() + target.tobytes()
    ).hexdigest()
    history = []
    diagnostics = {
        "data_sha256": data_hash,
        "independent_train_bases": len(X),
        "independent_test_bases": len(test),
        "compute": "ideal CPU simulation; zero QPU jobs",
        "synthetic_task": True,
        "shots": None,
        "config": cfg,
    }
    if method == "classical":
        predictions, metric, loss, name = _classical(experiment, X, y, test, target, seed)
        evaluations = cx = 0
        diagnostics.update(classical_model=name, final_train_loss=loss)
        validation = predictions.tolist()
        final_parameters = []
    else:
        engine = ShiftEngine(
            "spin" if experiment == "2A" else "curvature",
            method, cfg["broken_symmetry"],
        )
        if experiment == "2A":
            shared = 1.1
            bias = 0.0
            for step in range(cfg["steps"]):
                theta = np.full(8, shared)
                values, occurrence_jacobian, _ = engine.derivatives(X, theta)
                probabilities, loss = _classification_loss(values, y, bias)
                tied_jacobian = occurrence_jacobian.sum(axis=1)
                gradient = float(np.mean(4 * (probabilities - y) * tied_jacobian))
                bias_gradient = float(np.mean(probabilities - y))
                history.append({
                    "step": step, "loss": loss, "parameters": [shared, bias],
                    "gradient": [gradient, bias_gradient],
                })
                shared -= 0.3 * gradient
                bias -= 0.3 * bias_gradient
            theta = np.full(8, shared)
            train_values = engine.evaluate(X, theta)
            _, final_loss = _classification_loss(train_values, y, bias)
            predictions, _ = _classification_loss(engine.evaluate(test, theta), target, bias)
            metric = float(np.mean((predictions >= 0.5) == target))
            final_parameters = [shared, bias]
            diagnostics.update(
                gradient_rule="valid occurrence shifts, summed by chain rule for tied weight",
                spatial_reuse=engine.last_spatial_reuse,
                symmetry="translation/reflection of uniform prepared spin state and mean-Z",
                stronger_baseline_scope=(
                    "Equally informed Qiskit uses the same occurrence-orbit reduction. "
                    "Generalized shared-parameter shift schedules and reverse/adjoint CPU "
                    "gradients are not benchmarked; no best-gradient or unique advantage claim."
                ),
            )
        else:
            theta = np.array([0.4, 0.3, 0.2, 0.1])
            for step in range(cfg["steps"]):
                values, jacobian, hessian = engine.derivatives(X, theta, hessian=True)
                residual = values - y
                loss = float(np.mean(residual ** 2))
                gradient = 2 * np.mean(residual[:, None] * jacobian, axis=0)
                loss_hessian = 2 * np.mean(
                    jacobian[:, :, None] * jacobian[:, None, :]
                    + residual[:, None, None] * hessian, axis=0
                )
                damping = max(0.2, 0.05 - float(np.linalg.eigvalsh(loss_hessian).min()))
                direction = np.linalg.solve(
                    loss_hessian + damping * np.eye(4), -gradient
                )
                direction *= min(1.0, 0.5 / max(np.linalg.norm(direction), 1e-15))
                proposed = theta + direction
                proposed_loss = float(np.mean((engine.evaluate(X, proposed) - y) ** 2))
                accepted = proposed_loss <= loss
                history.append({
                    "step": step, "loss": loss, "parameters": theta.tolist(),
                    "gradient": gradient.tolist(), "hessian": loss_hessian.tolist(),
                    "damping": damping, "accepted": bool(accepted),
                })
                if accepted:
                    theta = proposed
            final_loss = float(np.mean((engine.evaluate(X, theta) - y) ** 2))
            predictions = engine.evaluate(test, theta)
            metric = float(np.mean((predictions - target) ** 2))
            final_parameters = theta.tolist()
            diagnostics.update(
                gradient_rule="single-occurrence parameter-shift gradient and Hessian",
                symmetry="2*pi expectation periodicity using integer shift metadata",
                raw_derivative_requests_per_example=41,
                canonical_derivative_requests_per_example=37,
                stronger_baseline_scope=(
                    "Equally informed Qiskit includes periodic Hessian query reduction. "
                    "Adjoint/reverse CPU Hessians and other generalized derivative schedules "
                    "are not benchmarked; no best-second-order or unique advantage claim."
                ),
            )
        evaluations = engine.evaluations
        cx = engine.cx_per_evaluation
        diagnostics.update(
            requested_queries=engine.requests, weight_invalidations=engine.invalidations,
            final_train_loss=final_loss, initial_train_loss=history[0]["loss"],
            parameter_count=1 if experiment == "2A" else 4,
            learned_classical_bias=experiment == "2A",
            native_cx_per_executed_circuit=cx,
            all_native_cx_includes_derivative_queries=True,
        )
        validation = predictions.tolist() + list(final_parameters)
        for point in history:
            validation += [point["loss"]] + point["parameters"] + point["gradient"]
            if "hessian" in point:
                validation += np.asarray(point["hessian"]).ravel().tolist()
    return {
        "experiment": experiment, "seed": int(seed), "method": method,
        "metric_name": "heldout_accuracy" if experiment == "2A" else "heldout_mse",
        "metric_value": metric, "higher_is_better": experiment == "2A",
        "n_qubits": 8 if experiment == "2A" else 6,
        "evaluations": int(evaluations), "cx_per_evaluation": int(cx),
        "total_cx": int(evaluations * cx),
        "diagnostics": diagnostics, "history": history,
        "predictions": predictions.tolist(), "targets": target.tolist(),
        "final_parameters": final_parameters, "validation_vector": validation,
        "validation_fingerprint": hashlib.sha256(
            json.dumps(validation, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest(),
    }
