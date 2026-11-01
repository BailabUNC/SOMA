# SOMA second revision implementation

This directory integrates the collaborator-provided October 2026 implementation
and supplementary experiments for the second manuscript revision. It preserves
the model operations, seeds, checkpoint-selection rules and reported experiment
definitions while giving the scripts consistent local input/output paths.
The earlier repository workflows, acquisition code and hardware files remain
available outside this directory.

## Scope and private assets

This is a source-and-summary distribution. The second-revision datasets, model
weights, per-window labels/predictions and manuscript working documents are not
included. Running the scientific experiments requires private inputs supplied
locally. The aggregate values in [`results/`](results/) are provided results,
not a claim that full-duration training was repeated during integration.

According to the collaborator's data interpretation adopted for this revision,
`S1.pt`, `S2.pt` and `S3.pt` represent three participants; `C2.pt` is an additional
recording of Participant 2 on another day. This interpretation must accompany the
private files. File names or numeric IDs alone cannot establish provenance.
The historical `../data/loso/` documentation and resplitting script describe a
different pseudo-session workflow. Do not use that generator to manufacture
participant cohorts for these experiments.

## Environment

Use Python 3.12 and a virtual environment. Install the appropriate PyTorch 2.6
CPU or CUDA build for your system, then install the remaining requirements:

```bash
python -m pip install -r rev2/requirements.txt
```

The supplied component-result environment records PyTorch 2.6.0+cu126, CUDA 12.6,
cuDNN 90501 and an NVIDIA GeForce RTX 4060. This is the recorded package
environment; it does not establish the hardware used for every manuscript
analysis. Integration checks used PyTorch 2.6.0+cpu, NumPy 2.3.5 and
scikit-learn 1.8.0. CPU execution is supported, but exact retraining numbers can
depend on backend and dependency versions. All full runs use the schedules in
`config.json`; CPU smoke tests use short schedules and generated signals.

Run the commands below from the repository root. Paths are resolved from the
script directory by default, so execution also works from another directory.
Replace `/private/soma/DATA` with your actual local input folder.

## Input contract and split

Each `S{i}.pt` is a tensor dictionary with the following required fields.
Additional metadata is accepted and is not used as model input.

| Field | Shape and meaning |
|---|---|
| `features` | finite tensor `[N,512,48]` |
| `labels` | integer vector `[N]`, using the label IDs in `config.json` |
| `subject_ids` | integer vector `[N]`, all equal to `i` in `S{i}.pt` |

Configured cohort sizes are 199, 198 and 198 (595 windows total, 24 classes).
`session_sizes` and `session_seed` are retained legacy configuration names.
The seeded permutation restores the supplied files' global checkpoint index
layout; it does **not** pool and repartition participants. Each outer fold holds
out the corresponding entire input cohort. Within the other two cohorts,
`train_test_split(test_size=0.2, random_state=42, stratify=labels)` creates the
training/validation split before corruption or augmentation. This is a 20%
window-level stratified split, not a trial-grouped split. The tensors do not
supply original trial IDs, so no trial-grouped independence claim is made.

Phase I and II see only inner-training windows. Phase III standardizes the
frozen encoder features with inner-training mean and population SD. Input
normalization is separately computed per window and per channel over time.
For pretraining, the normalized targets are rounded to a 1/32 grid. Phase I
uses Gaussian corruption with SD 0.1 and Smooth L1 loss. Phase I and II retain
the last scheduled checkpoint (60 and 100 epochs). The 220-epoch classifier
retains the highest validation accuracy, breaking ties by lower validation
cross-entropy. Test labels do not select checkpoints.

## Main model and raw-input baseline

```bash
python rev2/train.py --data-dir /private/soma/DATA --device cuda
python rev2/raw_train.py --data-dir /private/soma/DATA --device cuda
```

Defaults are `rev2/outputs/main/` and `rev2/outputs/raw/`. Every training entry
accepts `--data-dir`, `--config`, `--out` and `--device`; scripts using the
separate classifier settings also accept `--head-config`.

The main Phase I encoder has 128/64/32/16/32 filters. Phase II copies its first
four blocks into an encoder ending in 64 filters and global average pooling;
the fifth block is new. It updates only the last two blocks with SimSiam.
Phase III freezes the entire encoder and trains LayerNorm(64), Linear(64,256),
ReLU, Dropout(0.05), Linear(256,24), with 22,936 trainable parameters.
The raw-input baseline uses a frozen random projection and the same classifier
head. It is distinct from the end-to-end supervised CNN below.

## Component experiments

```bash
python rev2/components.py --data-dir /private/soma/DATA --device cuda
# Run just the two supplemental from-scratch conditions:
python rev2/components.py --data-dir /private/soma/DATA --device cuda --variants scratch_simsiam supervised_scratch
```

Default output: `rev2/outputs/components/`. Conditions are `full`, `ae_only`,
`scratch_simsiam`, `supervised_scratch`. The `full` and `ae_only` conditions
train their Phase I reference within each fold. The two from-scratch conditions
do not load pretrained weights. The supervised condition optimizes encoder and
classifier together; feature mean/SD are computed once from the initially
random encoder on the training windows and held fixed during joint training,
matching the supplied implementation.

## Matched AE64 experiment

```bash
python rev2/ae64_only.py --data-dir /private/soma/DATA --device cuda
python rev2/ae64_simsiam.py --data-dir /private/soma/DATA --device cuda
```

Default output folders: `rev2/outputs/ae64/` and `rev2/outputs/ae64_simsiam/`.
The second command reads the first folder; override it with `--ae-dir` when
needed. Here the fifth block has 64 filters **during AE pretraining**. Both
conditions start from the same fully pretrained encoder; the SimSiam condition
then updates its last two blocks. This pair isolates SimSiam's contribution
within that configuration and is distinct from the main model and its
`ae_only` component condition.

| Condition | Mean accuracy ± population SD across 3 folds |
|---|---:|
| Main model | 74.28% ± 4.51 percentage points |
| Component `ae_only` (AE32 first four blocks, new fifth block) | 71.76% ± 0.78 percentage points |
| Matched AE64-only | 62.68% ± 4.92 percentage points |
| Matched AE64 → SimSiam | 69.07% ± 4.47 percentage points |
| SimSiam from scratch | 58.83% ± 3.83 percentage points |
| Supervised CNN from scratch | 54.93% ± 11.65 percentage points |

The older six-condition augmentation study remains available through
`ablation.py --models-dir ... --variants ...`. Its variants are `saved_ssl`,
`mild_ssl`, `s4_ssl`, `lastblock_ssl`, `ae_only`, `raw`; its separate settings
are in `ablation_config.json`. The latter's SSL learning rate and quantization
settings differ from the main/component configuration. Do not relabel those
conditions as the matched AE64 experiment or combine their scores silently.

## Learning histories

```bash
python rev2/train_q9.py --data-dir /private/soma/DATA --device cuda
python rev2/plot_q9_curves.py --history-dir rev2/outputs/history
# Plot the supplied aggregate histories without private study inputs:
python rev2/plot_q9_curves.py --history-dir rev2/results/history --out rev2/outputs/supplied_histories.svg
```

`train_q9.py` records train/validation loss for every epoch of all three stages
and train/validation accuracy for the classifier. It writes checkpoints and
histories under `rev2/outputs/history/`. AE/SimSiam selection remains fixed at
the final epoch, irrespective of the diagnostic validation losses. The supplied
histories select classifier epochs 147, 102 and 89; these match the selection
metadata in the supplied main checkpoints. The history package did not include
its rerun checkpoints, so per-parameter equality with the earlier main weights
has not been asserted.

## Evaluation and prediction

```bash
python rev2/evaluate.py --data-dir /private/soma/DATA --models-dir rev2/outputs/main --out rev2/outputs/main_evaluation
python rev2/evaluate.py --external /private/soma/DATA/C2.pt --models-dir rev2/outputs/main --out rev2/outputs/c2_evaluation
python rev2/evaluate.py --data-dir /private/soma/DATA --models-dir rev2/outputs/components --checkpoint-pattern 'fold{fold}/supervised_scratch.pth' --out rev2/outputs/supervised_evaluation
python rev2/predict.py --fold 1 --data-dir /private/soma/DATA --models-dir rev2/outputs/main
```

The evaluator validates saved indices against the configured held-out split
and reports accuracy, macro-F1, balanced accuracy and confusion matrices.
External inputs require `features` and `labels` using the same shape and IDs;
they never contribute to training or selection. `predict.py` retains a
prediction-only interface. Generated per-window outputs are private and ignored
by Git. Use only trusted local checkpoint files.

C2 accuracies for the main checkpoints are 48.04%, 31.86% and 50.00%.
Under the adopted provenance, Folds 1 and 3 have seen Participant 2's earlier
recording; Fold 2 has not seen that participant. Fold 2 therefore combines
participant and day changes; the three scores are not three independent
external participants or a pure estimate of a day effect.

## Reconstruction metrics

```bash
python rev2/evaluate_reconstruction.py --data-dir /private/soma/DATA --models-dir rev2/outputs/main
# For Phase I weights stored inside final model files:
python rev2/evaluate_reconstruction.py --data-dir /private/soma/DATA --models-dir /private/checkpoints --checkpoint-pattern 'fold{fold}.pth'
```

The evaluation follows the supplied original evaluator: normalize and quantize
each target, add seeded Gaussian noise (SD 0.1), then reconstruct. Per-window
MSE is averaged over all channels/time points; nRMSE divides RMSE by the clean
target's full-window range; Pearson r flattens channel/time jointly. Summary
SD uses `ddof=0` across windows within each split. These numbers describe the
specified synthetic-corruption denoising task, not measured physiological ground
truth or an uncorrupted-input reconstruction test. Supplied held-out nRMSE
means are 0.1255/0.1211/0.1274 and Pearson r means are 0.9045/0.9120/0.8998.

## Validation and release boundary

```bash
python -m unittest discover -s rev2/tests -v
```

Integration validation includes generated-signal CPU training, disjoint split
checks, encoder-freezing checks, both from-scratch branches, actual main/C2
checkpoint inference and all four supplementary ablation conditions. Details
and limits are recorded in `VALIDATION.md`. The second-revision code does not
replace earlier hardware firmware or supply new hardware characterization
scripts. This branch prepares the code integration; no release tag or published
GitHub Release is created by the commit itself.
