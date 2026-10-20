"""Reproduce one installed-Aer read-only-unitary-buffer correctness fixture.

Run from the repository with:
    PYTHONPATH=src python examples/aer_readonly_block_regression.py --output receipt.json

The fixture takes about one second on a laptop. It executes no remote or QPU
work. It records whether this exact read-only-buffer issue reproduces; it does
not assume every Aer version exhibits the issue. The writable-copy path must
pass the unchanged absolute 1e-10 output checks and 1e-10 fidelity tolerance.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, state_fidelity
from threadpoolctl import threadpool_limits

from group_memory_qml.deep_response import DeepResponseEngine


def run():
    started = time.perf_counter()
    n, layers = 4, 3
    X = np.random.default_rng(9).normal(size=(2, n))
    angles = np.array([.317, -.583])
    engine = DeepResponseEngine(n, layers, "qiskit_blocks")
    bindings = {p: X[:, j].tolist() for j, p in enumerate(engine.inputs)}
    bindings[engine.weight] = angles.tolist()
    native = engine._native_values(X, angles)
    reference_states = []
    for row, angle in zip(X, angles):
        mapping = dict(zip(engine.inputs, row))
        mapping[engine.weight] = angle
        reference_states.append(Statevector.from_instruction(engine.source.assign_parameters(mapping)))

    rows = []
    for readonly in (True, False):
        matrices = list(engine.blocks)
        for matrix in matrices:
            matrix.flags.writeable = not readonly
        gates = [UnitaryGate(matrix, check_input=False) for matrix in matrices]
        circuit = QuantumCircuit(n)
        for j, parameter in enumerate(engine.inputs):
            circuit.ry(parameter, j)
        for layer in range(layers):
            circuit.ry(engine.weight, layer % n)
            circuit.append(gates[layer % 2], range(n), copy=False)
        circuit.save_expectation_value(engine._observable, range(n), label="z")
        circuit.save_statevector(label="state")
        for runtime_binding in (False, True):
            engine._backend.set_options(runtime_parameter_bind_enable=runtime_binding)
            result = engine._backend.run(
                circuit, parameter_binds=[bindings], shots=None, seed_simulator=6147
            ).result()
            values = np.array([result.data(i)["z"] for i in range(len(X))])
            states = [Statevector(result.data(i)["state"]) for i in range(len(X))]
            fidelity = [float(state_fidelity(ref, actual))
                        for ref, actual in zip(reference_states, states)]
            tv = [float(.5*np.sum(np.abs(ref.probabilities()-actual.probabilities())))
                  for ref, actual in zip(reference_states, states)]
            error = float(np.max(np.abs(values-native)))
            passed = error <= 1e-10 and min(fidelity) >= 1-1e-10 and max(tv) <= 1e-10
            if not readonly and not passed:
                raise AssertionError("Writable-copy control failed the unchanged correctness checks")
            rows.append({
                "readonly_matrix": readonly, "runtime_parameter_bind_enable": runtime_binding,
                "mean_z": values.tolist(), "max_mean_z_error": error,
                "state_fidelity": fidelity, "probability_tv": tv, "passed": passed,
            })
    return {
        "scope": "Exact local CPU fixture; not a universal Aer or parameter-binding claim",
        "n_qubits": n, "layers": layers, "input_seed": 9, "inputs": X.tolist(),
        "theta": angles.tolist(), "native_mean_z": native.tolist(),
        "source_sha256": engine.diagnostics["source_sha256"],
        "native_sha256": engine.diagnostics["native_sha256"],
        "fixed_block_sha256": engine.diagnostics["fixed_block_sha256"],
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "qiskit", "qiskit-aer"]},
        "output_atol": 1e-10, "state_fidelity_minimum": 1-1e-10,
        "readonly_failure_reproduced": any(r["readonly_matrix"] and not r["passed"] for r in rows),
        "writable_copy_passes": all(r["passed"] for r in rows if not r["readonly_matrix"]),
        "results": rows, "elapsed_seconds": time.perf_counter()-started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        receipt = run()
    encoded = json.dumps(receipt, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded+"\n")
    print(encoded)
