"""Input-orbit classification and vector-valued denoising, experiments 1A/1B."""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

METHOD_MAP = {"group_memory": "group_memory",
              "qiskit_symmetry": "qiskit_same_symmetry",
              "qiskit_exact": "qiskit_exact_cache"}


def permutations(n, views):
    if not 1 <= views <= 2*n:
        raise ValueError("views must be between 1 and 2*n")
    # Alternating rotations/reflections, with no duplicate permutation.
    idx = np.arange(n)
    return np.array([p for j in range(n)
                     for p in (np.roll(idx, j), np.roll(idx[::-1], j))][:views])


def _key(x):
    return min(np.ascontiguousarray(p).tobytes()
               for j in range(len(x)) for p in (np.roll(x, j), np.roll(x[::-1], j)))


def dataset(experiment, seed, n=6, views=4, bases=8):
    if n < 6 or n > 14 or n % 2 or bases < 4 or bases % 2:
        raise ValueError("even n in 6..14 and even bases >=4 required")
    perms = permutations(n, views)
    rng = np.random.default_rng(seed)
    splits = []
    orbit_sets = []
    for split in range(2):
        xx, yy, keys = [], [], set()
        for pair in range(bases//2):
            if experiment == "1A":
                a = .85+rng.uniform(-.16, .16, n//2)
                b = -.85+rng.uniform(-.16, .16, n//2)
                alt = np.empty(n); alt[::2], alt[1::2] = a, b
                examples = [(np.r_[a, b], 0), (alt, 1)]
            else:
                examples = []
                for _ in range(2):
                    t = np.arange(n)*2*np.pi/n
                    clean = (rng.uniform(.3, .65)*np.sin(t+rng.uniform(0, 2*np.pi))
                             + rng.uniform(.1, .3)*np.cos(2*t+rng.uniform(0, 2*np.pi)))
                    # Noise is drawn before creating transformed views.
                    noisy = clean+rng.normal(0, .22, n)
                    examples.append((noisy, clean))
            for x, y in examples:
                key = _key(x)
                if key in keys:
                    raise AssertionError("Unexpected repeated base orbit")
                keys.add(key)
                for p in perms:
                    xx.append(x[p]); yy.append(y if experiment == "1A" else y[p])
        splits.append((np.array(xx), np.array(yy)))
        orbit_sets.append(keys)
    assert not orbit_sets[0].intersection(orbit_sets[1])
    digest = hashlib.sha256()
    for x, y in splits:
        digest.update(x.astype("<f8").tobytes()); digest.update(y.astype("<f8").tobytes())
    return (*splits[0], *splits[1]), {
        "dataset_sha256": digest.hexdigest(), "bases_per_split": bases,
        "views_per_base": views, "orbit_disjoint": True,
        "test_independent_units": bases//2 if experiment == "1A" else bases,
        "test_views_are_not_independent": True,
    }


def _features(x):
    n = x.shape[1]
    return np.c_[np.abs(np.fft.rfft(x, axis=1))**2/n,
                 np.stack([np.mean(x*np.roll(x, j, axis=1), axis=1)
                           for j in range(1, n//2+1)], axis=1)]


def _fit(x, y, seed):
    scale = StandardScaler().fit(x)
    model = LogisticRegression(C=1, max_iter=1000, random_state=seed).fit(scale.transform(x), y)
    return scale, model


def _local_features(x):
    return np.stack([x]+[(np.roll(x, j, axis=1)+np.roll(x, -j, axis=1))/2
                         for j in range(1, x.shape[1]//2+1)], axis=-1)


def run(experiment, seed, method, config=None):
    if experiment not in ("1A", "1B") or method not in (*METHOD_MAP, "classical"):
        raise ValueError("Unknown experiment or method")
    cfg = {"n_qubits": 6, "views": 4, "bases": 8, "updates": 5, "symmetry": True}
    cfg.update(config or {})
    (x, y, xt, yt), metadata = dataset(experiment, seed, cfg["n_qubits"], cfg["views"], cfg["bases"])
    n = cfg["n_qubits"]
    engine = None
    trace = []
    if method != "classical":
        # Do not force the classical baseline to import quantum libraries.
        from ..engine import QMLReuseEngine
        engine = QMLReuseEngine(n, METHOD_MAP[method], cfg["symmetry"], gradients=experiment == "1A")
    if experiment == "1A":
        if engine is None:
            scale, head = _fit(_features(x), y, seed)
            probabilities = head.predict_proba(scale.transform(_features(xt)))[:, 1]
        else:
            weights = np.array([.53, -.41])
            for step in range(cfg["updates"]):
                z, jac = engine.evaluate(x, weights, gradient=True)
                f = z.mean(axis=1, keepdims=True)
                scale, head = _fit(f, y, seed)
                p = head.predict_proba(scale.transform(f))[:, 1]
                grad = np.mean((p-y)[:, None]*jac, axis=0)*head.coef_[0, 0]/scale.scale_[0]
                grad /= max(1., np.linalg.norm(grad))
                loss = -np.mean(y*np.log(np.clip(p, 1e-12, 1))+(1-y)*np.log(np.clip(1-p, 1e-12, 1)))
                trace.extend([float(loss), *weights.tolist(), *grad.tolist()])
                weights = weights-.05*grad
            z, _ = engine.evaluate(x, weights)
            zt, _ = engine.evaluate(xt, weights)
            scale, head = _fit(z.mean(axis=1, keepdims=True), y, seed)
            probabilities = head.predict_proba(scale.transform(zt.mean(axis=1, keepdims=True)))[:, 1]
            trace.extend(weights.tolist())
        predicted = probabilities >= .5
        metric, name, higher = float(np.mean(predicted == yt)), "heldout_accuracy", True
        vector = probabilities.tolist()+trace
        extra = {"chance_accuracy": .5, "predictions": predicted.astype(int).tolist(),
                 "labels": yt.tolist(), "training_trajectory": trace}
    else:
        if engine is None:
            f, ft = _local_features(x), _local_features(xt)
        else:
            z, _ = engine.evaluate(x, np.array([.53, -.41]))
            zt, _ = engine.evaluate(xt, np.array([.53, -.41]))
            # Frozen quantum features plus raw readings; one shared linear head.
            f = np.concatenate([x[..., None], _local_features(z)], axis=-1)
            ft = np.concatenate([xt[..., None], _local_features(zt)], axis=-1)
        head = Ridge(alpha=.1).fit(f.reshape(-1, f.shape[-1]), y.ravel())
        predicted = head.predict(ft.reshape(-1, ft.shape[-1])).reshape(yt.shape)
        metric, name, higher = float(np.mean((predicted-yt)**2)), "heldout_mse", False
        vector = predicted.ravel().tolist()
        extra = {"raw_noisy_input_mse": float(np.mean((xt-yt)**2)),
                 "zero_predictor_mse": float(np.mean(yt**2)),
                 "quantum_weights_frozen": True, "shared_ridge_head_trained": True}
    evaluations = engine.forward_count if engine else 0
    cx = engine.gates.get("cx", 0) if engine else 0
    return {
        "experiment": experiment, "seed": seed, "method": method, "config": cfg,
        "metric_name": name, "metric_value": metric, "higher_is_better": higher,
        "n_qubits": n, "evaluations": evaluations, "total_cx": evaluations*cx,
        "validation_vector": vector,
        "diagnostics": {**metadata, **extra,
                        "gradient_instances": engine.gradient_count if engine else 0,
                        "cache_invalidations": engine.invalidations if engine else 0,
                        "group_enabled": engine.group_enabled if engine else False,
                        "cx_scope": "Forward simulation only; reverse gradients are classical",
                        "compute": "Ideal CPU statevectors; no QPU or finite shots"},
    }
