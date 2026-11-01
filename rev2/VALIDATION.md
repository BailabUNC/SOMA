# Integration validation

Checks completed on 2026-11-01 using Python 3.12, PyTorch 2.6.0+cpu,
NumPy 2.3.5 and scikit-learn 1.8.0. Private input tensors/checkpoints were
used locally and are not part of this source distribution.

## Saved checkpoint inference

| Condition | Correct counts by fold | Mean accuracy |
|---|---|---:|
| Main | 160/199, 138/198, 144/198 | 74.2754% |
| AE64-only | 135/199, 111/198, 127/198 | 62.6804% |
| AE64→SimSiam | 150/199, 131/198, 130/198 | 69.0650% |
| SimSiam from scratch | 108/199, 126/198, 116/198 | 58.8312% |
| Supervised CNN from scratch | 139/199, 82/198, 106/198 | 54.9329% |
| C2, three main checkpoints | 98/204, 65/204, 102/204 | 43.3007% |

All 12 supplemental-ablation fold prediction vectors match the supplied JSON
vectors exactly; their ground-truth vectors also match. The main counts and
metrics match the supplied main summary. C2 reproduces the previously checked
counts and metrics. Held-out checkpoint indices were checked against the
configured data layout and split.

## Reconstruction

The adapted recovered evaluator was run with Phase I weights embedded in the
supplied final main checkpoints. Across all three folds and all train,
validation and held-out windows, the maximum absolute difference from the
supplied per-window MSE/nRMSE/Pearson values was 2.384185791015625e-7. All
held-out summary values agree with SI Table S6 at its displayed precision.

## Training and interfaces

Five generated-signal CPU tests pass:

- Outer folds and inner training/validation sets are disjoint and complete.
- Extra metadata is accepted; invalid cohort IDs and non-finite features fail.
- Phase III preserves the Phase II encoder; Phase II preserves the first three
  Phase I encoder blocks.
- The supervised from-scratch branch updates the encoder and produces valid
  classifier outputs.
- Scratch SSL and seeded reconstruction run on CPU with valid tensor shapes.

All Python entrypoints compile and their `--help` commands succeed. Actual
one-epoch CPU runs (60/100/220 schedules overridden only in temporary validation
configs) completed through all three folds for `train.py`, `raw_train.py`,
`components.py` (all four variants), `ae64_only.py`, `ae64_simsiam.py`,
`train_q9.py`, and `ablation.py` (all six variants). Prediction and history-plot
commands were also exercised. These short runs check execution, not manuscript
accuracy or convergence.

The supplied 3×(60/100/220) epoch histories are complete. Applying their recorded
validation rule selects epochs 147/102/89, matching the supplied main checkpoint
selection metadata. The AE64 pair has identical saved starting encoder weights,
identical fold indices and changes only the last two encoder blocks during SSL.

## Limits

Full-duration model training was not repeated during integration. GPU/CUDA
execution was not available for integration tests. The raw-projection baseline
Table S5 summary is marked as reported, since no original raw-baseline weights
were supplied for inference comparison. The history package lacks its rerun
weights, so matching epoch-selection metadata is not asserted to prove complete
parameter identity. Participant/day identities follow the collaborator's
interpretation and are not inferable from tensor values or filenames alone.

Only code, configuration, aggregate results and aggregate epoch histories are
included in the new directory. No second-revision input tensors, checkpoint
weights, per-window prediction files or manuscript documents are staged.
