"""Multi-parameter, matrix-free QML training with matched full-overhead controls.

These are new architectures, not a continuation of the shallow-model timings.
All same-model methods use the same data, head, initialization and updates.
"""
from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np

METHODS = (
    "group_spectral", "group_grid", "qiskit_response", "qiskit_shift",
    "classical_adjoint", "classical",
)


def _configuration(experiment, config):
    if experiment not in ("multi_sensor", "multi_shared"):
        raise ValueError("Unsupported experiment")
    cfg = dict(n_qubits=8, n_parameters=2,
               occurrences=4, updates=12, bases=8, views=2,
               n_train=16, n_test=32)
    if set(config) - set(cfg):
        raise ValueError("Unknown configuration field")
    cfg.update(config)
    for key, low, high in (("n_qubits", 4, 16), ("n_parameters", 1, 8),
                           ("occurrences", 1, 8),
                           ("updates", 1, 40), ("bases", 4, 32),
                           ("views", 1, 16), ("n_train", 4, 64),
                           ("n_test", 4, 64)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"Invalid {key}")
    if cfg["n_qubits"] % 2 or any(cfg[k] % 2 for k in ("bases", "n_train", "n_test")):
        raise ValueError("Qubit and base/sample counts must be even")
    if cfg["views"] > 2*cfg["n_qubits"]:
        raise ValueError("Too many views")
    return cfg


def run(experiment, seed, method, config=None):
    if method not in METHODS or type(seed) is not int or seed < 0:
        raise ValueError("Invalid method or seed")
    cfg = _configuration(experiment, dict(config or {}))
    tick = perf_counter()
    from .experiments import sensors, derivatives

    if experiment == "multi_sensor":
        (x, y, test, target), metadata = sensors.dataset(
            "1A", seed, cfg["n_qubits"], cfg["views"], cfg["bases"])
        classical_features = sensors._features
    else:
        raw, y, raw_test, target = derivatives.make_data(
            "2A", seed, cfg["n_train"], cfg["n_test"])
        x = np.repeat(raw, cfg["n_qubits"], axis=1)
        test = np.repeat(raw_test, cfg["n_qubits"], axis=1)
        metadata = dict(test_independent_units=len(test), orbit_disjoint=True,
                        test_views_are_not_independent=False)
        classical_features = lambda v: np.c_[np.sin(v[:, 0]), np.cos(v[:, 0])]
    digest = hashlib.sha256()
    for array in (x, y, test, target):
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    metadata["dataset_sha256"] = digest.hexdigest()
    data_s = perf_counter()-tick
    trace, engine = [], None
    parameter_training = None
    setup_s = evaluate_s = fit_s = 0.
    if method == "classical":
        tick = perf_counter()
        scale, head = sensors._fit(classical_features(x), y, seed)
        probabilities = head.predict_proba(
            scale.transform(classical_features(test)))[:, 1]
        fit_s = perf_counter()-tick
    else:
        tick = perf_counter()
        from .multi_response import MultiResponseEngine
        engine = MultiResponseEngine(cfg["n_qubits"], cfg["n_parameters"], cfg["occurrences"], method=method)
        setup_s = perf_counter()-tick
        theta = np.linspace(.37, .69, cfg["n_parameters"])
        initial_theta = theta.copy()
        for step in range(cfg["updates"]):
            tick = perf_counter()
            values, jacobian = engine.evaluate(x, theta, gradient=True)
            evaluate_s += perf_counter()-tick
            tick = perf_counter()
            feature = values[:, None]
            scale, head = sensors._fit(feature, y, seed)
            p = head.predict_proba(scale.transform(feature))[:, 1]
            gradient = np.mean((p-y)[:, None]*jacobian, axis=0)*head.coef_[0, 0]/scale.scale_[0]
            clipped = np.clip(gradient, -1., 1.)
            loss = float(-np.mean(y*np.log(np.clip(p, 1e-12, 1)) +
                                 (1-y)*np.log(np.clip(1-p, 1e-12, 1))))
            # Keep values and gradients as well as the optimizer trajectory:
            # accuracy alone cannot establish equivalence.
            trace.extend([*theta.tolist(), *gradient.tolist(), loss, *values.tolist(), *jacobian.ravel().tolist()])
            theta -= .05*clipped
            fit_s += perf_counter()-tick
        tick = perf_counter()
        final, _ = engine.evaluate(x, theta)
        heldout, _ = engine.evaluate(test, theta)
        evaluate_s += perf_counter()-tick
        tick = perf_counter()
        scale, head = sensors._fit(final[:, None], y, seed)
        probabilities = head.predict_proba(scale.transform(heldout[:, None]))[:, 1]
        trace.extend([*theta.tolist(), *final.tolist(), *heldout.tolist()])
        parameter_training = dict(
            independent_parameters=cfg["n_parameters"],
            updates=cfg["updates"], initial=initial_theta.tolist(),
            final=theta.tolist(), displacement=(theta-initial_theta).tolist(),
        )
        fit_s += perf_counter()-tick
    predicted = probabilities >= .5
    diagnostics = engine.diagnostics if engine is not None else {}
    if callable(diagnostics):
        diagnostics = diagnostics()
    return dict(
        experiment=experiment, seed=seed, method=method, config=cfg,
        n_qubits=cfg["n_qubits"], n_parameters=cfg["n_parameters"], metric_name="heldout_accuracy",
        metric_value=float(np.mean(predicted == target)), higher_is_better=True,
        evaluations=int(diagnostics.get("total_circuit_evaluations", 0)),
        total_cx=int(diagnostics.get("native_cx_work", 0)),
        validation_vector=[*probabilities.tolist(), *trace],
        diagnostics={
            **metadata, "response": diagnostics,
            "chance_accuracy": .5, "predictions": predicted.astype(int).tolist(),
            "labels": target.tolist(), "training_trajectory": trace,
            "parameter_training": parameter_training,
            "stages": dict(data_s=data_s, setup_s=setup_s,
                           evaluate_s=evaluate_s, fit_predict_s=fit_s),
            "compute": "Ideal CPU simulation, no QPU or finite shots",
            "symmetry_scope": "Exact multi-parameter Fourier structure; no spatial-orbit assumption",
            "cx_scope": "Native-L3 verification calls only; diagonal, coefficient and local-gate arithmetic have separate costs, not QPU savings",
        })
