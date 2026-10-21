"""Small quantum image filters and exact causal cones of a connected ring.

All quantum execution is ideal CPU statevector simulation of compiled 4/9-qubit
patches or three-qubit cones. The global ring's state is never constructed by
``run``. Its commuting ZZ layer generally entangles the connected ring.
"""
from __future__ import annotations

import hashlib

import numpy as np

METHODS = ("group_memory", "qiskit_symmetry", "qiskit_exact", "classical")


def _hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fit(train, labels, test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler().fit(train)
    head = LogisticRegression(C=1., max_iter=2000, random_state=seed)
    head.fit(scale.transform(train), labels)
    return head.predict_proba(scale.transform(test))[:, 1]


def _grid_permutations(side):
    indices = np.arange(side * side).reshape(side, side)
    return np.asarray([np.rot90(view, k).ravel()
                       for view in (indices, indices[:, ::-1]) for k in range(4)])


def _grid_edges(side):
    return [(r * side + c, r * side + c + 1)
            for r in range(side) for c in range(side - 1)] + [
                (r * side + c, (r + 1) * side + c)
                for r in range(side - 1) for c in range(side)]


def _exact_groups(rows, method):
    """Two equally informed implementations; no rounded numerical cache keys."""
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if method == "group_memory":
        table = {}
        for row in rows:
            key = row.tobytes()
            if key not in table:
                table[key] = row.copy()
        ordered = sorted(table)
        position = {key: index for index, key in enumerate(ordered)}
        return np.asarray([table[key] for key in ordered]), np.asarray(
            [position[row.tobytes()] for row in rows])
    packed = rows.view(np.dtype((np.void, rows.shape[1] * 8))).ravel()
    _, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
    return rows[first], inverse


def _canonical(rows, permutations):
    representatives = []
    for row in rows:
        orbit = np.ascontiguousarray(row[permutations])
        best = min(range(len(orbit)), key=lambda i: orbit[i].tobytes())
        representatives.append(orbit[best])
    return np.asarray(representatives)


def _simulate(source, parameters, bindings, measured_sites):
    # Imports and generic native compilation belong to the caller's timed run.
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    native = transpile(source, basis_gates=["rz", "sx", "x", "cx"],
                       optimization_level=3, approximation_degree=1.,
                       seed_transpiler=6147, num_processes=1)
    if native.layout is not None:
        raise AssertionError("This no-routing oracle requires unchanged wire order")
    indices = np.arange(2 ** native.num_qubits, dtype=np.uint64)
    signs = np.asarray([1. - 2. * ((indices >> q) & 1) for q in measured_sites])
    values = []
    for row in bindings:
        circuit = native.assign_parameters(dict(zip(parameters, row)))
        values.append(signs @ Statevector.from_instruction(circuit).probabilities())
    return np.asarray(values), int(native.count_ops().get("cx", 0))


def _patch_source(side, symmetry):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    n = side * side
    x = ParameterVector("x", n)
    q = QuantumCircuit(n)
    gains = np.ones(n) if symmetry else np.linspace(.65, 1.35, n)
    for j in range(n):
        q.ry(x[j] + .31, j)
    for a, b in _grid_edges(side):
        q.rzz(.53, a, b)
    for j in range(n):
        q.rx(-.41 * gains[j], j)
        q.ry(.47, j)
    return q, tuple(x)


def patch_features(patches, method, side=2, symmetry=True):
    """One scalar mean-local-Z filter response for each patch."""
    patches = np.asarray(patches, dtype=float)
    if side not in (2, 3) or patches.ndim != 2 or patches.shape[1] != side * side:
        raise ValueError("Expected flattened 2x2 or 3x3 patches")
    if method not in METHODS[:3] or not np.isfinite(patches).all():
        raise ValueError("Invalid quantum method or patch values")
    enabled = bool(symmetry and method != "qiskit_exact")
    rows = _canonical(patches, _grid_permutations(side)) if enabled else patches
    unique, inverse = _exact_groups(rows, method)
    source, parameters = _patch_source(side, symmetry)
    local, cx = _simulate(source, parameters, unique, range(side * side))
    return local.mean(axis=1)[inverse], {
        "requested_patches": len(patches), "evaluations": len(unique),
        "cx_per_evaluation": cx, "total_cx": cx * len(unique),
        "group_enabled": enabled, "symmetry": "D4 grid rotations/reflections",
    }


def _texture_data(side, seed, train_pairs=4, test_pairs=3):
    image_side = 4 * side
    rr, cc = np.indices((image_side, image_side))
    splits = []
    seen = set()
    for namespace, pairs in ((1103, train_pairs), (2207, test_pairs)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, side, namespace]))
        images, labels, ids = [], [], []
        for pair in range(pairs):
            # Balanced pixel histograms: spatial arrangement supplies the label.
            stripes = (cc % 2).astype(float)
            checker = ((rr + cc) % 2).astype(float)
            for label, base in enumerate((stripes, checker)):
                # Swap opposite-valued pixels so both classes retain the same
                # histogram. Reject whole-image orbits shared with any base.
                for _ in range(1000):
                    image = base.copy()
                    flat = image.ravel()
                    swaps = int(rng.integers(1, 3))
                    zero = rng.choice(np.flatnonzero(flat == 0), swaps, replace=False)
                    one = rng.choice(np.flatnonzero(flat == 1), swaps, replace=False)
                    flat[zero], flat[one] = 1., 0.
                    key = min(np.rot90(view, k).copy().tobytes()
                              for view in (image, image[:, ::-1]) for k in range(4))
                    if key not in seen:
                        seen.add(key)
                        break
                else:
                    raise RuntimeError("Unable to sample a new image orbit")
                for rotation in range(4):
                    images.append(np.rot90(image, rotation).copy())
                    labels.append(label)
                    ids.append(f"{namespace}-pair{pair}-class{label}")
        splits.append((np.asarray(images), np.asarray(labels), ids))
    return splits


def _patches(images, side):
    count, height, width = images.shape
    if height % side or width % side:
        raise ValueError("Images must be divisible into non-overlapping patches")
    return images.reshape(count, height // side, side, width // side, side).transpose(
        0, 1, 3, 2, 4).reshape(count, -1, side * side)


def _pool(maps):
    maps = np.asarray(maps)
    return np.stack([maps.mean(axis=1), maps.std(axis=1),
                     np.mean(maps * np.roll(maps, 1, axis=1), axis=1),
                     np.mean(maps * np.roll(maps, 2, axis=1), axis=1)], axis=1)


def _classical_texture(images):
    signed = 2 * images - 1
    return np.stack([
        signed.mean(axis=(1, 2)),
        .5 * ((signed[:, 1:] * signed[:, :-1]).mean(axis=(1, 2))
              + (signed[:, :, 1:] * signed[:, :, :-1]).mean(axis=(1, 2))),
        (signed[:, 1:, 1:] * signed[:, :-1, :-1]).mean(axis=(1, 2)),
    ], axis=1)


def _image_run(seed, method, config):
    side = int(config.get("patch_side", 2))
    if side not in (2, 3):
        raise ValueError("patch_side must be 2 or 3")
    symmetry = bool(config.get("symmetry", True))
    train, test = _texture_data(side, seed)
    images = np.concatenate([train[0], test[0]])
    diagnostics = {
        "architecture": "RY(data)-shared grid RZZ-RX-RY; mean local Z filter",
        "source": "Synthetic noisy binary stripes versus checkerboards",
        "learning": "Only a train-fitted classical scaler and logistic head",
        "independent_train_bases": len(set(train[2])),
        "independent_test_bases": len(set(test[2])),
        "base_ids_disjoint": set(train[2]).isdisjoint(test[2]),
        "whole_image_orbits_disjoint": True,
        "patch_side": side, "patch_qubits": side * side,
        "data_fingerprint": _hash(images, train[1], test[1]),
        "cache_scope": "Fixed unlabeled filter; exact patch values may recur across splits",
        "classical_control": "Train-only logistic head on local image correlations",
    }
    ntrain = len(train[0])
    if method == "classical":
        features = _classical_texture(images)
        evaluations = total_cx = 0
        vector = features.ravel()
    else:
        patches = _patches(images, side)
        values, receipt = patch_features(
            1.4 * patches.reshape(-1, side * side) - .7, method, side, symmetry)
        # Mean/dispersion are invariant to the order of local image patches.
        maps = values.reshape(len(images), -1)
        features = np.stack([maps.mean(axis=1), maps.std(axis=1),
                             np.mean(maps**2, axis=1)], axis=1)
        vector = maps.ravel()
        evaluations, total_cx = receipt["evaluations"], receipt["total_cx"]
        diagnostics.update(receipt)
    probabilities = _fit(features[:ntrain], train[1], features[ntrain:], seed)
    accuracy = float(np.mean((probabilities >= .5) == test[1]))
    diagnostics.update(test_probabilities=probabilities.tolist(),
                       test_labels=test[1].tolist(), chance_accuracy=.5)
    return _result("4A", seed, method, accuracy, side * side, evaluations,
                   total_cx, np.concatenate([vector, probabilities]), diagnostics)


def _cone_source():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    p = ParameterVector("p", 7)  # left, center, right, betaL, betaR, alpha, delta
    q = QuantumCircuit(3)
    for j in range(3):
        q.ry(p[j] + .31, j)
    q.rzz(p[3], 0, 1)
    q.rzz(p[4], 1, 2)
    q.rx(p[5], 1)
    q.ry(p[6], 1)
    return q, tuple(p)


def _ring_context(n, symmetry):
    beta = np.full(n, .53) if symmetry else np.linspace(.35, .73, n)
    alpha = np.full(n, -.41) if symmetry else np.linspace(-.6, -.22, n)
    return beta, alpha, .47


def _cone_rows(X, beta, alpha, delta):
    count, n = X.shape
    return np.stack([
        np.roll(X, 1, axis=1), X, np.roll(X, -1, axis=1),
        np.broadcast_to(np.roll(beta, 1), (count, n)),
        np.broadcast_to(beta, (count, n)),
        np.broadcast_to(alpha, (count, n)),
        np.full((count, n), delta),
    ], axis=-1).reshape(-1, 7)


def ring_formula(X, beta, alpha, delta=.47):
    """Exact local observable from product input and a commuting ZZ layer.

    This does not assume the post-entangler state factorizes. Gates outside
    the measured site's cone cancel in the Heisenberg picture.
    """
    X = np.asarray(X, dtype=float)
    a = X + .31
    left, right = np.roll(a, 1, axis=1), np.roll(a, -1, axis=1)
    beta_left = np.roll(beta, 1)
    product = (
        np.cos(beta_left) + 1j * np.sin(beta_left) * np.cos(left)
    ) * (np.cos(beta) + 1j * np.sin(beta) * np.cos(right))
    return (np.cos(delta) * np.cos(alpha) * np.cos(a)
            + np.cos(delta) * np.sin(alpha) * np.sin(a) * product.imag
            - np.sin(delta) * np.sin(a) * product.real)


def ring_features(X, method, symmetry=True):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 4 or not np.isfinite(X).all():
        raise ValueError("Expected finite input maps with at least four sites")
    if method not in METHODS:
        raise ValueError("Unknown method")
    n = X.shape[1]
    beta, alpha, delta = _ring_context(n, symmetry)
    if method == "classical":
        return ring_formula(X, beta, alpha, delta), {
            "evaluations": 0, "total_cx": 0, "group_enabled": False,
            "classical_formula_site_evaluations": int(X.size),
            "max_simulated_qubits": 0,
        }
    rows = _cone_rows(X, beta, alpha, delta)
    enabled = bool(symmetry and method != "qiskit_exact")
    if enabled:
        # Reflection exchanges both endpoint data and the associated couplings.
        rows = _canonical(rows, np.asarray([np.arange(7), [2, 1, 0, 4, 3, 5, 6]]))
    unique, inverse = _exact_groups(rows, method)
    source, parameters = _cone_source()
    local, cx = _simulate(source, parameters, unique, [1])
    values = local[:, 0][inverse].reshape(X.shape)
    return values, {
        "requested_local_observables": int(X.size), "evaluations": len(unique),
        "cx_per_evaluation": cx, "total_cx": cx * len(unique),
        "group_enabled": enabled, "max_simulated_qubits": 3,
        "causal_cone_sites": 3, "global_circuit_connected": True,
    }


def _ring_data(n, seed, periodic=True):
    motif = 8 if periodic else n
    splits = []
    for namespace in (3319, 4421):
        rng = np.random.default_rng(np.random.SeedSequence([seed, namespace]))
        inputs, labels, ids = [], [], []
        for pair in range(3):
            positive = .8 + rng.uniform(-.13, .13, motif // 2)
            negative = -.8 + rng.uniform(-.13, .13, motif // 2)
            contiguous = np.concatenate([positive, negative])
            alternating = np.empty(motif)
            alternating[::2], alternating[1::2] = positive, negative
            for label, block in enumerate((contiguous, alternating)):
                base = np.tile(block, n // motif)
                for view in (base, np.roll(base, 1), base[::-1], np.roll(base[::-1], 2)):
                    inputs.append(view.copy())
                    labels.append(label)
                    ids.append(f"{namespace}-pair{pair}-class{label}")
        splits.append((np.asarray(inputs), np.asarray(labels), ids))
    return splits


def _ring_run(seed, method, config):
    n = int(config.get("n_global", 256))
    if n < 8 or n > 1024 or n % 8:
        raise ValueError("n_global must be a multiple of eight in [8,1024]")
    periodic = bool(config.get("periodic", True))
    symmetry = bool(config.get("symmetry", True))
    train, test = _ring_data(n, seed, periodic)
    X = np.concatenate([train[0], test[0]])
    maps, diagnostics = ring_features(X, method, symmetry)
    beta, alpha, delta = _ring_context(n, symmetry)
    reference = ring_formula(X, beta, alpha, delta)
    # All methods pay the same independent readout check.
    error = float(np.max(np.abs(maps - reference)))
    if error > 1e-10:
        raise AssertionError(f"Local cone/formula mismatch: {error}")
    features = _pool(maps)
    probabilities = _fit(features[:len(train[0])], train[1], features[len(train[0]):], seed)
    diagnostics.update(
        architecture="Connected ring RY(data)-commuting RZZ-RX-RY, local Z outputs",
        global_qubits=n, full_global_state_simulated=False, QPU_executed=False,
        data_fingerprint=_hash(X, train[1], test[1]),
        input_motif_length=8 if periodic else n,
        source="Synthetic contiguous versus alternating histogram-matched ring motifs",
        independent_train_bases=len(set(train[2])),
        independent_test_bases=len(set(test[2])),
        base_ids_disjoint=set(train[2]).isdisjoint(test[2]),
        learning="Train-only scaler/logistic readout; quantum weights fixed",
        classical_control="Exact local analytic formula with identical features and readout",
        max_absolute_formula_error=error,
        test_probabilities=probabilities.tolist(), test_labels=test[1].tolist(),
        chance_accuracy=.5,
    )
    # Pooled features suffice for matching each head input without a huge report.
    return _result("4B", seed, method,
                   float(np.mean((probabilities >= .5) == test[1])),
                   n, diagnostics["evaluations"], diagnostics["total_cx"],
                   np.concatenate([features.ravel(), probabilities]), diagnostics)


def _result(experiment, seed, method, accuracy, n, evaluations, cx, vector, diagnostics):
    return {
        "experiment": experiment, "seed": int(seed), "method": method,
        "metric_name": "heldout_accuracy", "metric_value": accuracy,
        "higher_is_better": True, "n_qubits": n,
        "evaluations": int(evaluations), "total_cx": int(cx),
        "validation_vector": np.asarray(vector).tolist(),
        "validation_fingerprint": _hash(np.round(vector, 8)),
        "diagnostics": diagnostics,
    }


def run(experiment: str, seed: int, method: str, config: dict | None = None):
    if method not in METHODS:
        raise ValueError("Unknown method")
    config = {} if config is None else dict(config)
    if experiment in ("4A", "exp4A"):
        return _image_run(seed, method, config)
    if experiment in ("4B", "exp4B"):
        return _ring_run(seed, method, config)
    raise ValueError("Supported experiments: 4A, 4B")
