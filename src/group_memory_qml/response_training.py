"""Matched training loops for persistent parameter-response memory.

The historical engines remain unchanged. New engines learn an exact supported
trigonometric response from Qiskit probes, including an independent off-grid
check, and reuse that response across parameter updates. The Qiskit response
control receives precisely the same response representation.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

METHODS = (
    "group_memory_response", "qiskit_response", "group_memory_previous",
    "qiskit_symmetry", "classical_same_model", "classical",
)


def _receipt(engine):
    receipt = engine.diagnostics
    return receipt() if callable(receipt) else receipt


def _sensor(seed, method, config):
    from .experiments import sensors

    if method in ("group_memory_previous", "qiskit_symmetry", "classical"):
        old = "group_memory" if method == "group_memory_previous" else method
        result = sensors.run("1A", seed, old, config)
        result["method"] = method
        result["diagnostics"]["implementation"] = "unchanged eight-experiment implementation"
        return result

    from .sensor_response import SensorResponseEngine, AnalyticSensorEngine

    cfg = {"n_qubits": 6, "views": 4, "bases": 8, "updates": 5, "symmetry": True}
    cfg.update(config)
    started = time.perf_counter()
    (x, y, xt, yt), metadata = sensors.dataset(
        "1A", seed, cfg["n_qubits"], cfg["views"], cfg["bases"]
    )
    data_s = time.perf_counter() - started
    tick = time.perf_counter()
    if method == "classical_same_model":
        engine = AnalyticSensorEngine(cfg["n_qubits"], symmetry=cfg["symmetry"])
    else:
        engine = SensorResponseEngine(
            cfg["n_qubits"], method=method, symmetry=cfg["symmetry"]
        )
    setup_s = time.perf_counter() - tick
    weights = np.array([.53, -.41])
    trace = []
    evaluation_s = fit_s = 0.0
    for step in range(cfg["updates"]):
        tick = time.perf_counter()
        z, jac = engine.evaluate(x, weights, gradient=True)
        evaluation_s += time.perf_counter() - tick
        tick = time.perf_counter()
        f = z.mean(axis=1, keepdims=True)
        scale, head = sensors._fit(f, y, seed)
        p = head.predict_proba(scale.transform(f))[:, 1]
        grad = np.mean((p-y)[:, None]*jac, axis=0)*head.coef_[0, 0]/scale.scale_[0]
        grad /= max(1., np.linalg.norm(grad))
        loss = -np.mean(
            y*np.log(np.clip(p, 1e-12, 1))+(1-y)*np.log(np.clip(1-p, 1e-12, 1))
        )
        trace.extend([float(loss), *weights.tolist(), *grad.tolist()])
        weights = weights - .05*grad
        fit_s += time.perf_counter() - tick
    tick = time.perf_counter()
    z, _ = engine.evaluate(x, weights)
    zt, _ = engine.evaluate(xt, weights)
    evaluation_s += time.perf_counter() - tick
    tick = time.perf_counter()
    scale, head = sensors._fit(z.mean(axis=1, keepdims=True), y, seed)
    probabilities = head.predict_proba(scale.transform(zt.mean(axis=1, keepdims=True)))[:, 1]
    fit_s += time.perf_counter() - tick
    trace.extend(weights.tolist())
    predicted = probabilities >= .5
    evaluations = engine.forward_count
    cx = engine.gates.get("cx", 0)
    return {
        "experiment": "1A", "seed": seed, "method": method, "config": cfg,
        "metric_name": "heldout_accuracy",
        "metric_value": float(np.mean(predicted == yt)), "higher_is_better": True,
        "n_qubits": cfg["n_qubits"], "evaluations": evaluations,
        "total_cx": evaluations*cx,
        "validation_vector": probabilities.tolist()+trace,
        "diagnostics": {
            **metadata, "chance_accuracy": .5,
            "predictions": predicted.astype(int).tolist(), "labels": yt.tolist(),
            "training_trajectory": trace, "response": _receipt(engine),
            "gradient_instances": engine.gradient_count,
            "cache_invalidations": engine.invalidations,
            "group_enabled": engine.group_enabled,
            "stages": {"data_s": data_s, "setup_s": setup_s,
                       "evaluate_s": evaluation_s, "fit_predict_s": fit_s},
            "cx_scope": "Actual native probe/direct forward circuits; reverse gradients are classical",
            "compute": "Ideal CPU calculation; no QPU or finite shots",
        },
    }


def _spin(seed, method, config):
    from .experiments import derivatives

    if method in ("group_memory_previous", "qiskit_symmetry", "classical"):
        old = "group_memory" if method == "group_memory_previous" else method
        result = derivatives.run("2A", seed, old, config)
        result["method"] = method
        result["diagnostics"]["implementation"] = "unchanged eight-experiment implementation"
        return result

    from .spin_response import SpinResponseEngine, AnalyticSpinEngine

    cfg = {"steps": 3, "n_train": 12, "n_test": 12, "broken_symmetry": False}
    cfg.update(config)
    started = time.perf_counter()
    X, y, test, target = derivatives.make_data(
        "2A", seed, cfg["n_train"], cfg["n_test"]
    )
    data_s = time.perf_counter() - started
    digest = hashlib.sha256(
        X.tobytes()+y.tobytes()+test.tobytes()+target.tobytes()
    ).hexdigest()
    tick = time.perf_counter()
    if method == "classical_same_model":
        engine = AnalyticSpinEngine(broken_symmetry=cfg["broken_symmetry"])
    else:
        engine = SpinResponseEngine(method=method, broken_symmetry=cfg["broken_symmetry"])
    setup_s = time.perf_counter() - tick
    shared, bias = 1.1, 0.0
    history = []
    evaluation_s = training_s = 0.0
    for step in range(cfg["steps"]):
        tick = time.perf_counter()
        values, occurrence_jacobian, _ = engine.derivatives(X, np.full(8, shared))
        evaluation_s += time.perf_counter() - tick
        tick = time.perf_counter()
        probabilities, loss = derivatives._classification_loss(values, y, bias)
        tied_jacobian = occurrence_jacobian.sum(axis=1)
        gradient = float(np.mean(4*(probabilities-y)*tied_jacobian))
        bias_gradient = float(np.mean(probabilities-y))
        history.append({
            "step": step, "loss": loss, "parameters": [shared, bias],
            "gradient": [gradient, bias_gradient],
        })
        shared -= .3*gradient
        bias -= .3*bias_gradient
        training_s += time.perf_counter() - tick
    theta = np.full(8, shared)
    tick = time.perf_counter()
    train_values = engine.evaluate(X, theta)
    _, final_loss = derivatives._classification_loss(train_values, y, bias)
    predictions, _ = derivatives._classification_loss(
        engine.evaluate(test, theta), target, bias
    )
    evaluation_s += time.perf_counter() - tick
    validation = predictions.tolist()+[shared, bias]
    for point in history:
        validation += [point["loss"]]+point["parameters"]+point["gradient"]
    return {
        "experiment": "2A", "seed": seed, "method": method, "config": cfg,
        "metric_name": "heldout_accuracy",
        "metric_value": float(np.mean((predictions >= .5) == target)),
        "higher_is_better": True, "n_qubits": 8,
        "evaluations": engine.evaluations,
        "cx_per_evaluation": engine.cx_per_evaluation,
        "total_cx": engine.evaluations*engine.cx_per_evaluation,
        "history": history, "predictions": predictions.tolist(),
        "targets": target.tolist(), "final_parameters": [shared, bias],
        "validation_vector": validation,
        "diagnostics": {
            "data_sha256": digest, "independent_train_bases": len(X),
            "independent_test_bases": len(test), "synthetic_task": True,
            "compute": "Ideal CPU calculation; no QPU or finite shots",
            "response": _receipt(engine), "requested_queries": engine.requests,
            "weight_invalidations": engine.invalidations,
            "final_train_loss": final_loss, "initial_train_loss": history[0]["loss"],
            "stages": {"data_s": data_s, "setup_s": setup_s,
                       "evaluate_s": evaluation_s, "training_s": training_s},
            "gradient_rule": "Exact supported parameter response, occurrence derivatives summed by chain rule",
        },
    }


def run(experiment, seed, method, config=None):
    """Return the same task, trajectory and validation vector as the old suite."""
    if experiment not in ("1A", "2A") or method not in METHODS:
        raise ValueError("Unsupported experiment or method")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cfg = dict(config or {})
    if experiment == "1A":
        allowed = {"n_qubits", "views", "bases", "updates", "symmetry"}
        updates = cfg.get("updates", 5)
        if type(updates) is not int or not 1 <= updates <= 20:
            raise ValueError("updates must be an integer in 1..20")
    else:
        allowed = {"steps", "n_train", "n_test", "broken_symmetry"}
        steps = cfg.get("steps", 3)
        if type(steps) is not int or not 1 <= steps <= 8:
            raise ValueError("steps must be an integer in 1..8")
        for name in ("n_train", "n_test"):
            count = cfg.get(name, 12)
            if type(count) is not int or count % 2 or not 4 <= count <= 64:
                raise ValueError(f"{name} must be even in 4..64")
    if set(cfg)-allowed:
        raise ValueError("Unknown configuration field")
    return _sensor(seed, method, cfg) if experiment == "1A" else _spin(seed, method, cfg)
