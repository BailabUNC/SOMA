# Experiments and result definitions

Commands assume an installed source checkout and private input files as defined
in [the data guide](data.md). Checkpoints remain compatible: model module names
changed, but state-dictionary keys and tensor operations did not.

## Main model and raw-input baseline


```bash
python -m soma train --data-dir /private/soma/DATA --device cuda
python -m soma baseline --data-dir /private/soma/DATA --device cuda
```

Defaults are `outputs/main/` and `outputs/raw/`. Every training entry
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
python -m soma components --data-dir /private/soma/DATA --device cuda
# Run just the two supplemental from-scratch conditions:
python -m soma components --data-dir /private/soma/DATA --device cuda --variants scratch_simsiam supervised_scratch
```

Default output: `outputs/components/`. Conditions are `full`, `ae_only`,
`scratch_simsiam`, `supervised_scratch`. The `full` and `ae_only` conditions
train their Phase I reference within each fold. The two from-scratch conditions
do not load pretrained weights. The supervised condition optimizes encoder and
classifier together; feature mean/SD are computed once from the initially
random encoder on the training windows and held fixed during joint training,
matching the supplied implementation.


## Matched AE64 experiment


```bash
python -m soma ae64 --data-dir /private/soma/DATA --device cuda
python -m soma ae64-simsiam --data-dir /private/soma/DATA --device cuda
```

Default output folders: `outputs/ae64/` and `outputs/ae64_simsiam/`.
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

The separate six-condition augmentation study remains available through
`python -m soma augmentations --models-dir ... --variants ...`. Its variants are `saved_ssl`,
`mild_ssl`, `s4_ssl`, `lastblock_ssl`, `ae_only`, `raw`; its separate settings
are in `src/soma/configs/augmentation.json`. The latter's SSL learning rate and quantization
settings differ from the main/component configuration. Do not relabel those
conditions as the matched AE64 experiment or combine their scores silently.


## Learning histories


```bash
python -m soma train-history --data-dir /private/soma/DATA --device cuda
python -m soma plot-history --history-dir outputs/history
# Plot the supplied aggregate histories without private study inputs:
python -m soma plot-history --history-dir results/rev2/history --out outputs/supplied_histories.svg
```

`soma train-history` records train/validation loss for every epoch of all three stages
and train/validation accuracy for the classifier. It writes checkpoints and
histories under `outputs/history/`. AE/SimSiam selection remains fixed at
the final epoch, irrespective of the diagnostic validation losses. The supplied
histories select classifier epochs 147, 102 and 89; these match the selection
metadata in the supplied main checkpoints. The history package did not include
its rerun checkpoints, so per-parameter equality with the earlier main weights
has not been asserted.


## Reconstruction metrics


```bash
python -m soma reconstruct --data-dir /private/soma/DATA --models-dir outputs/main
# For Phase I weights stored inside final model files:
python -m soma reconstruct --data-dir /private/soma/DATA --models-dir /private/checkpoints --checkpoint-pattern 'fold{fold}.pth'
```

The evaluation follows the supplied original evaluator: normalize and quantize
each target, add seeded Gaussian noise (SD 0.1), then reconstruct. Per-window
MSE is averaged over all channels/time points; nRMSE divides RMSE by the clean
target's full-window range; Pearson r flattens channel/time jointly. Summary
SD uses `ddof=0` across windows within each split. These numbers describe the
specified synthetic-corruption denoising task, not measured physiological ground
truth or an uncorrupted-input reconstruction test. Supplied held-out nRMSE
means are 0.1255/0.1211/0.1274 and Pearson r means are 0.9045/0.9120/0.8998.



Aggregate records and their provenance are indexed in [results/rev2](../results/rev2/README.md).
