# Real smartphone sensor classification

This experiment asks whether exact parameter-response reuse helps on real
classification data once training, data preparation and competing classical
models are included. It is CPU simulation, not QPU execution.

## Data and split

The [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
contains accelerometer and gyroscope measurements from 30 participants performing
six activities. We use the released 561 time/frequency features. The original
feature-extraction cost is unavailable; the scope starts from this public
feature dataset, not from a live sensor stream. Dataset DOI: 10.24432/C54S4K,
Reyes-Ortiz, Anguita, Ghio, Oneto and Parra (2013). The current UCI page lists
CC BY 4.0; the bundled older README contains a noncommercial restriction. Both
statements are preserved in provenance. Raw data is not redistributed here.

Official test subjects stay outside training and validation. Validation subjects
come from the official training population. Subsets contain all six classes and
are balanced across classes and, where possible, subjects. Train-only scaling
and PCA produce 8 or 16 angle features; the fitted transformations are applied
unchanged to validation and test. Full 561-feature classical controls are also
included. Five seeds vary subject allocation, sampling, PCA and initialization;
they are repeated views of one dataset, not five independent datasets.

## Structured model and controls

The quantum-inspired model consists of 4 or 8 independent two-qubit blocks,
with two independent trainable angles per block, each occurring twice, separated
by fixed noncommuting rotations and CZ gates. It has 8 or 16 independent quantum
parameters and8 or 16 total qubits, but entanglement is confined to two qubits.
The measured mean-Z of each block feeds a jointly trained six-class softmax head.
Every quantum angle is updated independently. Increasing the parameter count
also increases the PCA input width and classical head width; this is a complete
model-size comparison, not an isolated causal effect of parameter count.

Exact factorization avoids an exponentially large joint response in all 16
parameters. Each block has a 25-point response grid. GroupMemory constructs its
Fourier coefficients directly and reuses them across updates. Equally informed
Qiskit uses the same factorization, coefficient support, compiled architecture,
input identities, cached diagonals and response lifetime. Both perform the same
native off-grid checks. When training uses at least 25 updates, both response
methods precompute the validation response once and charge that setup cost. An exact classical adjoint differentiates the identical
model directly, without response reconstruction.

Three separate task-specific classical controls use logistic regression, an
RBF SVM and a small MLP. Each fits both the same angle features and all 561
standardized public features. The feature view is selected using validation
log loss, and both fits count toward its runtime. Hyperparameters are fixed;
test labels never select a model or checkpoint. All candidate validation scores
and selected views are retained.

## Frozen plan

- 128 training examples,8 independently trainable quantum parameters.
- 512 training examples,8 independently trainable quantum parameters.
- 512 training examples,16 independently trainable quantum parameters.
- 128 validation and 256 test examples per seed.
- Five seeds, two timing repeats, six methods: 180 attempted runs when complete.
- Sixty full-batch Adam updates for the quantum model and its exact controls;
  best validation loss selects the final checkpoint. The two response engines also precompute validation responses when the update
  count reaches the 25-point response size; this setup is fully charged. Classical learners use
  their own appropriate solvers; the comparison is task completion, not equal
  optimizer-step counts.

Accuracy, balanced accuracy, cross-entropy, predictions, training trajectories,
parameter motion, resource use, compilation checks and process timings are saved.
Corresponding quantum methods must agree on the complete saved trajectory and
predictions within 1e-10. A speed result is only described when all five paired
seed medians are faster and median paired saving is at least 5%. A useful win
against a classical control additionally requires no lower balanced accuracy
on every paired seed. These are descriptive criteria, not significance tests.
Missing, failed and timed-out attempts remain visible.

## Timing and reproduction

Each attempt uses a fresh interpreter and cache, one computational thread, and
serial execution. The parent clock includes startup, imports, dataset reading,
train-only preprocessing, construction, compilation, calibration and checks,
cache bookkeeping, all training and validation, final prediction, result
serialization and shutdown. Download and raw-text-to-NPZ preparation are common
one-time costs reported separately. Parent orchestration, tests, pilots and
reporting are also separately recorded. Package installation and implementation
are excluded. No remote or paid compute is required.

Prepare the dataset with `python -m group_memory_qml.real_sensor_data --data-dir data/uci_har`. Run a small
checkpoint/resume smoke before the full benchmark:

```bash
python -m group_memory_qml.real_sensor_suite run \
  --conditions benchmarks/real_sensor_quick.json \
  --data-dir data/uci_har --output results/my-real-smoke \
  --seeds 955001 --repeats 1 --limit 2 --budget-seconds 600
# Repeat the same command without --limit to resume the same immutable plan.
python -m group_memory_qml.real_sensor_suite run \
  --conditions benchmarks/real_sensor.json \
  --data-dir data/uci_har --output results/my-real-run \
  --budget-seconds 1800
```

The controller keeps the last two checkpoints plus the most complete snapshot,
checks every completed journal record, and preserves its original deadline on
resume. An unfinished short attempt is rerun from scratch; completed runs are
never silently overwritten. Each attempt is capped at 240 seconds. This is a
benchmark-stage checkpoint, not a promise to resume Adam halfway through an
individual training attempt.
