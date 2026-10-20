# Archived sensor reference

These are the unchanged 10-qubit symmetric sensor rows extracted from the original local CPU experiment on 2026-10-19. `trials.json` contains 3 data seeds × 2 timing repeats × 3 quantum methods; `classical.json` contains 3 separately timed classical runs. They support the historical README table and are separate from the freshly executed package runs in ../release-validation/.

The original experiment used the same fixed ring architecture and supplied dihedral symmetry. This release refactors its code, adds defensive copies, and avoids recomputing an already cached forward result when a subsequent request adds a gradient. The historical values are not measurements of this refactored package.

Only two quantum parameters were trained, for three updates. Every test seed has four independent paired histograms, eight bases and32relatedviews. Forward-CNOT-work savings count avoided circuits, not per-circuit gate reductions. Reverse gradients are classical computations. Same-symmetry Qiskit gets the same reduction; the classical control is much faster.
