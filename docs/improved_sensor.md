# Longer sensor training with a validation-selected representation

This follow-up keeps `results/real-sensor` frozen. It asks whether better input
encoding and longer training close the accuracy gap to ordinary classical
sensor classifiers, and whether GroupMemory saves complete application time.

## Model and selection

The same eight independent two-qubit patches have sixteen independently trained
rotation angles. Each angle occurs twice. A six-class softmax head is trained
jointly with the quantum angles. There is no entanglement between patches; the
patches could be evaluated sequentially on two qubits. Repeating input coordinates
does not tie the trainable angles or create new input information.

A separate two-seed training/validation-only search compared original PCA,
compressed/shifted PCA, tiled five-dimensional LDA, and LDA plus PCA. The recipe
with smallest mean validation log loss was selected before final tests:

1. Fit standardization of all 561 sensor features on selected training rows.
2. Fit five-component, shrinkage-auto LDA on those rows and their training labels.
3. Standardize the LDA coordinates using training rows only.
4. Repeat the five coordinates across sixteen patch inputs and encode each as
   `pi/2 + 0.5 * atan(z)`.
5. Run 1,000 full-batch joint Adam updates (final repair: learning rate .003,
   Adam epsilon 1e-4, head L2 1e-4).
6. Select the lowest validation log loss, including initialization, before one
   final test evaluation. Do not select the highest test accuracy.

LDA is **classical supervised learning**. This is a hybrid model improvement,
not evidence that GroupMemory improves the mathematical predictions of the
same model. The physics and exact response functions are unchanged.

## Controls

- **GroupMemory:** directly constructs exact local parameter responses.
- **Equally informed Qiskit:** receives identical patch factorization, parameter
  response support, cache keys/lifetime and compilation reuse.
- **Classical adjoint:** computes the same model using direct derivatives.
- **Frozen quantum angles:** computes the fixed train/validation features once
  and trains only the same output head; final test once after selection.
- **Logistic, RBF and MLP:** each fits the encoded input view, all 561 standardized
  features, and the five unique uncompressed LDA axes. Validation chooses the
  view within each family; all fits and predictions are charged. MLP has a
  1,000-iteration limit, with warnings retained.

All four conditions use the same repaired optimizer. "Original PCA" names the
input encoding; it does not reproduce the earlier study's different learning
rate. Three GroupMemory-only quality ablations use that PCA encoding for 60
and 1,000 updates and shifted PCA for 1,000 updates. Their runtime is not used as
an equally informed Qiskit comparison. The full main plan has five matched
training/split seeds, two timing repetitions and seven methods; the ablations
add thirty attempts, for one hundred planned attempts.

## Numerical repair and test reuse

The initial 100-attempt recipe used learning rate .03 and Adam epsilon 1e-8.
Although every process completed, the strict full-trajectory comparison failed:
maximum absolute disagreement was 0.0340823. Equal classification accuracies did
not rescue that failure. Those timings are retained as rejected evidence, never
pooled with the repaired comparison. The original source is frozen in commit
`a0f8cb5fc67845c0be30ea08cc51dac782b8ac96`.

A bounded validation-only numerical pilot tested common optimizer settings.
The repair uses learning rate .003 and epsilon 1e-4 identically for all quantum
engines and the fixed-angle control. No values are rounded to force agreement;
the complete trajectory threshold remains absolute 1e-10. Final performance
claims require the separate saved-data audit to pass.

The initial receipt also showed a 48.499 ms disagreement between UTC and
monotonic elapsed time. The cause is unknown. The new controller records
monotonic interval endpoints for elapsed-time checks and retains UTC separately
for deadlines, without rewriting the old receipt.

The same final rows have already been evaluated by the initial recipe. The
repaired experiment is a **same-holdout correctness-recovery retest**, not a new
untouched holdout. Neither representation nor repair selection uses test scores.

## Original fresh-row holdout construction

The source-pinned final holdout has 256 balanced official-test rows. It excludes
all 1,248 unique rows evaluated in the earlier quick/pilot/main runs and their
immediate same-subject neighbors (2,376 excluded rows in total). The same nine
official-test subjects appeared in the previous study. This is fresh-row
follow-up evaluation within UCI HAR, **not a new independent cohort**. Adjacent
index exclusion reduces obvious reuse but does not certify temporal independence.
All methods and training seeds share this holdout. Seeds are not independent
datasets. Subject-disjoint train/validation splits come only from official
training subjects. Search opens no test arrays.

## Reproduce

```bash
python -m pip install -e '.[dev]'
python -m group_memory_qml.real_sensor_data --data-dir data/uci_har
python -m pytest -q
python -m group_memory_qml.improved_sensor_suite run \
  --conditions benchmarks/improved_sensor_stable_quick.json --data-dir data/uci_har \
  --output results/my-improved-smoke --seeds 956001 --repeats 1 \
  --budget-seconds 600 --limit 2
# Resume the exact same short run without --limit; do not change its source/data.
python -m group_memory_qml.improved_sensor_suite run \
  --conditions benchmarks/improved_sensor_stable_quick.json --data-dir data/uci_har \
  --output results/my-improved-smoke --seeds 956001 --repeats 1 --budget-seconds 600
python -m group_memory_qml.improved_sensor_suite run \
  --conditions benchmarks/improved_sensor_stable.json --data-dir data/uci_har \
  --output results/my-improved-run --seeds 956101 956102 956103 956104 956105 \
  --repeats 2 --budget-seconds 1800
```

The controller uses fresh serial subprocesses, one computational thread, a
fixed deadline and interruption-safe journal checkpoints. Resume checks source,
data, plan, deadline and all recorded results. A completed attempt is the resume
unit; an unfinished short fit restarts. A per-method timeout is 240 seconds.
The resume smoke validates a graceful stage-boundary stop. There is a narrow
hard-interruption window between committing a checkpoint and writing its timing
sidecar: recovery then fails closed on the missing receipt and requires manual
inspection. It does not invent missing timing evidence.
Expired budgets are not silently extended. Errors and missing coverage remain
visible. These commands use local CPU simulation, no paid QPU or remote instance.

## Timing and claims

The primary per-attempt timer includes the worker and parent result/journal/
checkpoint/status writes. The worker component encloses interpreter startup, imports, dataset read/hash,
training-only preprocessing, construction/compilation/response calibration,
training, every validation evaluation, selection, final test, serialization and
process exit. Both clocks are reported. Sidecar writes, controller setup and
remaining campaign overhead are covered by the enclosing campaign clock;
development search, preflight and report production are separate study costs. Common acquisition components
from the original dataset preparation are separate and are not a directly
measured complete acquisition wall time. Installation, human/AI development and
the data owner's sensor collection/feature extraction are unmeasured.

Equivalent quantum pipelines must match saved full optimizer trajectories and
outputs within absolute `1e-10`; the tolerance is not relaxed for longer training.
A descriptive runtime win requires five complete seed pairs, two matching
repetitions each, every seed median faster, and median paired saving at least 5%.
A useful observed win also requires no lower point accuracy in any paired seed.
These gates are not significance or statistical equivalence tests. A faster
simulation is not a QPU or quantum learning advantage.

Source: [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
DOI 10.24432/C54S4K. Dataset provenance and historical license statements are
preserved by the loader; raw dataset files are not included in this repository.
