"""Small CPU fidelity-kernel experiments with exact, explicitly scoped reuse.

3A: raw fidelity is invariant under a JOINT permutation of the two inputs.
3B: full-orbit density features make the averaged kernel independently invariant.
No overlap-circuit execution, sampling, QPU timing, or hardware CNOTs are claimed.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from ..data import classical_features, four_views, orbit_key

METHODS = ("group_memory", "qiskit_symmetry", "qiskit_exact", "classical")


def permutations(n):
    sites = np.arange(n)
    return np.asarray([np.roll(sites, k) for k in range(n)] +
                      [np.roll(sites[::-1], k) for k in range(n)])


def feature_circuit(n, symmetry=True):
    """Two input layers separated by entanglers; fidelity is not product-only."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    inputs = ParameterVector("x", n)
    circuit = QuantumCircuit(n)
    gains = np.ones(n) if symmetry else np.linspace(.55, 1.45, n)
    for j in range(n):
        circuit.ry(inputs[j], j)
    for j in range(n):
        circuit.rzz(.71, j, (j + 1) % n)
    for j in range(n):
        circuit.rz(gains[j] * inputs[j], j)
        circuit.ry(.23, j)
    return circuit, tuple(inputs)


def make_data(n=6, seed=846101, pairs=2, views=4, orientation_shift=False):
    """Split histogram pairs before views; 2 pairs => 16 examples per split."""
    if n not in (6, 8):
        raise ValueError("Kernel examples support 6 or 8 qubits")
    if pairs not in (2, 4) or views not in (1, 2, 4):
        raise ValueError("pairs must be 2 or 4; views must be 1, 2 or 4")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    splits, seen = {}, set()
    for split, namespace in (("train", 7931), ("test", 7963)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, namespace]))
        x, y, ids, keys = [], [], [], []
        for pair in range(pairs):
            positive = .85 + rng.uniform(-.18, .18, n // 2)
            negative = -.85 + rng.uniform(-.18, .18, n // 2)
            contiguous = np.r_[positive, negative]
            alternate = np.empty(n)
            alternate[::2], alternate[1::2] = positive, negative
            for label, base in enumerate((contiguous, alternate)):
                key = orbit_key(base)
                if key in seen:
                    raise RuntimeError("Train/test orbit leakage")
                seen.add(key)
                # Apply a deterministic held-out orientation shift, not a label rule.
                if orientation_shift and split == "test":
                    base = np.roll(base[::-1], 2)
                for view in four_views(base)[:views]:
                    x.append(view)
                    y.append(label)
                    ids.append(f"{split}-pair{pair}-class{label}")
                    keys.append(key.hex())
        splits[split] = {"x": np.asarray(x), "y": np.asarray(y), "base_ids": ids, "orbit_keys": keys}
    return splits


def _key(x):
    return np.ascontiguousarray(x, dtype="<f8").tobytes()


class KernelComputer:
    """One immutable feature map, exact state cache, and fresh kernel cache.

    Dictionary and vectorized-unique planners use identical representatives and
    sorted missing jobs. qiskit_exact has ordinary exact-state and pair caching.
    """

    def __init__(self, n=6, method="group_memory", averaged=False, symmetry=True):
        if n not in (6, 8) or method not in METHODS[:-1]:
            raise ValueError("Unsupported qubit count or quantum method")
        from qiskit import transpile
        self.n, self.method, self.averaged, self.symmetry = n, method, bool(averaged), bool(symmetry)
        self.perms = permutations(n)
        self.source, self.inputs = feature_circuit(n, symmetry)
        self.native = transpile(self.source, basis_gates=["rz", "sx", "x", "cx"],
                                optimization_level=3, approximation_degree=1.,
                                seed_transpiler=6147, num_processes=1)
        if self.native.layout is not None:
            raise RuntimeError("Unexpected layout")
        self.cx_per_preparation = int(self.native.count_ops().get("cx", 0))
        self.states, self.densities, self.values, self.orbits = {}, {}, {}, {}
        self.state_preparations = self.kernel_overlaps = self.density_averages = 0
        self.requested_pairs = self.orbit_members_averaged = 0
        # Averaging proves independent invariance even for a nonsymmetric map.
        self.group_enabled = method != "qiskit_exact" and (symmetry or averaged)

    def _orbit(self, x):
        key = _key(x)
        if key not in self.orbits:
            self.orbits[key] = [_key(x[p]) for p in self.perms]
        return self.orbits[key]

    def pair(self, x, y):
        if self.averaged and self.group_enabled:
            a, b = min(self._orbit(x)), min(self._orbit(y))
            return tuple(sorted((a, b)))
        if self.group_enabled:
            # Same group element acts on BOTH inputs. Independent canonicalization
            # would change this raw fidelity kernel and is deliberately absent.
            return min(tuple(sorted((a, b))) for a, b in zip(self._orbit(x), self._orbit(y)))
        return tuple(sorted((_key(x), _key(y))))

    def _unique(self, jobs):
        if self.method == "group_memory":
            return sorted(dict.fromkeys(jobs))
        if not jobs:
            return []
        packed = np.frombuffer(b"".join(a + b for a, b in jobs),
                               dtype=np.dtype((np.void, self.n * 16)))
        indices = np.unique(packed, return_index=True)[1]
        return sorted(jobs[int(index)] for index in indices)

    def _prepare_states(self, keys):
        from qiskit.quantum_info import Statevector
        for key in sorted(set(keys) - self.states.keys()):
            x = np.frombuffer(key, dtype="<f8")
            circuit = self.native.assign_parameters(dict(zip(self.inputs, x)))
            self.states[key] = np.asarray(Statevector.from_instruction(circuit).data)
            self.state_preparations += 1

    def matrix(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        for data in (x, y):
            if data.ndim != 2 or data.shape[1] != self.n or not np.isfinite(data).all():
                raise ValueError("Expected finite matrices with n input columns")
        jobs = [self.pair(a, b) for a in x for b in y]
        return self._evaluate_jobs(jobs).reshape(len(x), len(y))

    def diagonal(self, x):
        """Compute only self entries, with the same policy for every method.

        Raw normalized pure-state fidelity has a known unit diagonal. An
        averaged density feature instead needs its explicitly computed purity.
        No unused off-diagonal test/test overlaps are evaluated.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n or not np.isfinite(x).all():
            raise ValueError("Expected a finite matrix with n input columns")
        if not self.averaged:
            self.requested_pairs += len(x)
            return np.ones(len(x))
        return self._evaluate_jobs([self.pair(item, item) for item in x])

    def _evaluate_jobs(self, jobs):
        self.requested_pairs += len(jobs)
        missing = [job for job in self._unique(jobs) if job not in self.values]
        features = sorted({key for job in missing for key in job})
        if self.averaged:
            for key in features:
                if key in self.densities:
                    continue
                members = self._orbit(np.frombuffer(key, dtype="<f8"))
                self._prepare_states(members)
                states = np.asarray([self.states[member] for member in members])
                # Mean over ALL 2n group elements, retaining multiplicities.
                density = states.T @ states.conj() / len(members)
                self.densities[key] = density.reshape(-1)
                self.density_averages += 1
                self.orbit_members_averaged += len(members)
            store = self.densities
        else:
            self._prepare_states(features)
            store = self.states
        # Chunking bounds temporary memory for the 8-qubit averaged features.
        for start in range(0, len(missing), 64):
            chunk = missing[start:start+64]
            left = np.asarray([store[a] for a, _ in chunk])
            right = np.asarray([store[b] for _, b in chunk])
            overlap = np.einsum("ij,ij->i", left.conj(), right, optimize=False)
            values = overlap.real if self.averaged else np.abs(overlap) ** 2
            for job, value in zip(chunk, values):
                self.values[job] = float(value)
        self.kernel_overlaps += len(missing)
        return np.asarray([self.values[job] for job in jobs])

    def diagnostics(self):
        return {
            "state_preparations": self.state_preparations,
            "kernel_overlaps": self.kernel_overlaps,
            "group_density_averages": self.density_averages,
            "orbit_members_averaged": self.orbit_members_averaged,
            "requested_pairs": self.requested_pairs,
            "cx_per_state_preparation": self.cx_per_preparation,
            "simulated_state_preparation_cx": self.state_preparations * self.cx_per_preparation,
            "group_reuse_enabled": self.group_enabled,
            "kernel_definition": "full orbit density inner product" if self.averaged else "raw state fidelity",
            "pair_reduction": "independent orbit representatives only AFTER full averaging" if self.averaged
                              else "joint simultaneous permutation only; exact fallback if symmetry broken",
            "overlap_note": "Classical vector/density inner products; zero QPU overlap circuits or shots.",
        }


def normalize_and_check(train, test, test_diagonal):
    """Charge identical diagonal normalization and numerical PSD checks."""
    train = np.asarray(train, float)
    diagonal = np.diag(train)
    test_diagonal = np.asarray(test_diagonal, float)
    if np.any(diagonal <= 0) or np.any(test_diagonal <= 0):
        raise AssertionError("Kernel diagonal must be positive")
    normalized = train / np.sqrt(diagonal[:, None] * diagonal[None, :])
    cross = test / np.sqrt(test_diagonal[:, None] * diagonal[None, :])
    normalized = (normalized + normalized.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    minimum = float(eigenvalues.min())
    if minimum < -1e-9:
        raise AssertionError(f"Kernel not PSD: {minimum}")
    # A common, tiny numerical diagonal repair preserves the cross-kernel.
    # Large negative eigenvalues fail; none are silently projected away.
    repair = max(0., -minimum)
    normalized += np.eye(len(normalized)) * repair
    return normalized, cross, {"minimum_eigenvalue_before_repair": minimum,
                              "diagonal_jitter": repair,
                              "diagonal_normalization": True,
                              "psd_policy": "Reject eigenvalue below -1e-9; add only tiny roundoff diagonal jitter."}


def run(experiment: str, seed: int, method: str, config: dict | None = None) -> dict:
    """One complete fresh data/setup/kernel/SVM invocation; root also times it."""
    start = time.perf_counter()
    if experiment not in ("3A", "3B") or method not in METHODS:
        raise ValueError("Expected experiment 3A/3B and a supported method")
    config = dict(config or {})
    allowed = {"n_qubits", "pairs", "views", "symmetry"}
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(config)-allowed)}")
    n, pairs, views = config.get("n_qubits", 6), config.get("pairs", 2), config.get("views", 4)
    symmetry = config.get("symmetry", True)
    if not isinstance(symmetry, (bool, np.bool_)):
        raise ValueError("symmetry must be boolean")
    data = make_data(n, seed, pairs, views, orientation_shift=experiment == "3B")
    train, test = data["train"], data["test"]
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import rbf_kernel
    if method == "classical":
        scaler = StandardScaler()
        a = scaler.fit_transform(classical_features(train["x"]))
        b = scaler.transform(classical_features(test["x"]))
        gamma = 1. / a.shape[1]
        gram = rbf_kernel(a, gamma=gamma)
        cross = rbf_kernel(b, a, gamma=gamma)
        diagonal = np.ones(len(b))
        diagnostics = {"kernel_definition": "RBF on train-scaled invariant Fourier power/circular correlations",
                       "state_preparations": 0, "kernel_overlaps": 0,
                       "classical_kernel_entries": int(gram.size + cross.size + len(b)),
                       "simulated_state_preparation_cx": 0, "group_reuse_enabled": False}
    else:
        computer = KernelComputer(n, method, averaged=experiment == "3B", symmetry=symmetry)
        gram = computer.matrix(train["x"], train["x"])
        cross = computer.matrix(test["x"], train["x"])
        # In particular, the averaged kernel diagonal is NOT assumed to be one.
        diagonal = computer.diagonal(test["x"])
        diagnostics = computer.diagnostics()
    normalized, cross_normalized, psd = normalize_and_check(gram, cross, diagonal)
    classifier = SVC(kernel="precomputed", C=1., tol=1e-8, shrinking=False)
    classifier.fit(normalized, train["y"])
    predictions = classifier.predict(cross_normalized)
    score = float(np.mean(predictions == test["y"]))
    fingerprint = hashlib.sha256()
    for array in (train["x"], train["y"], test["x"], test["y"]):
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    vector = np.r_[normalized.ravel(), cross_normalized.ravel(), predictions.astype(float)]
    record = {
        "experiment": experiment, "seed": int(seed), "method": method,
        "metric_name": "heldout_accuracy", "value": score, "higher_is_better": True,
        "n_qubits": n, "evaluations": diagnostics["kernel_overlaps"],
        "total_cx": diagnostics["simulated_state_preparation_cx"],
        "cx_scope": "CX gates in actually simulated state preparations only; classical kernel overlaps are NOT QPU circuits.",
        "diagnostics": {**diagnostics, "psd": psd, "symmetry": bool(symmetry),
                        "train_rows": len(train["x"]), "test_rows": len(test["x"]),
                        "independent_histogram_pairs_per_split": pairs,
                        "base_patterns_per_split": pairs * 2, "views_per_base": views,
                        "chance_accuracy": .5, "full_group_average_charged": experiment == "3B" and method != "classical",
                        "orientation_shift": experiment == "3B"},
        "validation_fingerprint": fingerprint.hexdigest(),
        "validation_vector": vector.tolist(),
        "outputs": {"train_kernel": normalized.tolist(), "test_kernel": cross_normalized.tolist(),
                    "unnormalized_train_diagonal": np.diag(gram).tolist(),
                    "unnormalized_test_diagonal": diagonal.tolist(),
                    "predictions": predictions.tolist(), "labels": test["y"].tolist()},
        "data": {"train_base_ids": train["base_ids"], "test_base_ids": test["base_ids"],
                 "train_orbit_keys": train["orbit_keys"], "test_orbit_keys": test["orbit_keys"]},
        "compute": "Ideal CPU statevectors and classical inner products/SVM; no QPU execution.",
    }
    json.dumps(record, allow_nan=False)
    record["internal_total_s"] = time.perf_counter() - start
    return record
