"""Bounded, reproducible QML use-case experiments; no QPU execution."""

EXPERIMENTS = {
    "1A": "sensors",
    "1B": "sensors",
    "2A": "derivatives",
    "2B": "derivatives",
    "3A": "kernels",
    "3B": "kernels",
    "4A": "local",
    "4B": "local",
}
METHODS = ("group_memory", "qiskit_symmetry", "qiskit_exact", "classical")
