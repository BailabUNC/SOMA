# Quick start

Use Python 3.12 and a suitable PyTorch 2.6 CPU/CUDA build. From the repository
root, install the source checkout and check the command menu:

```bash
python -m pip install -e .
python -m soma --help
```

The `soma` console command is equivalent to `python -m soma`. Each subcommand
accepts `--help`. Core dependencies are declared in `pyproject.toml`; use
`python -m pip install -e '.[acquisition,preparation]'` only when you need the
serial acquisition and CSV preparation tools. Choose your PyTorch build before
installing this project if your machine requires a specific CPU/CUDA index.

## Train and evaluate

Supply private `S1.pt`, `S2.pt`, `S3.pt` via `--data-dir`, or place them in
ignored `data/private/`. See [Data protocol](data.md) before starting a study run.

```bash
python -m soma train --data-dir /private/soma/DATA --device cuda
python -m soma evaluate --data-dir /private/soma/DATA --models-dir outputs/main
python -m soma evaluate --external /private/soma/DATA/C2.pt --models-dir outputs/main --out outputs/c2
python -m soma reconstruct --data-dir /private/soma/DATA --models-dir outputs/main
```

Default generated files live under `outputs/`. In an editable source checkout,
this is the repository's output folder regardless of your current directory.
Set `SOMA_WORKSPACE` to use a different workspace, or pass explicit `--out`,
`--data-dir` and `--models-dir` paths. For a regular wheel installation without
an editable checkout, the current working directory is the default workspace.
Packaged defaults are `src/soma/configs/main.json` and `augmentation.json`;
external copies can be supplied with `--config` and `--head-config`.

## Command index

| Command | Purpose | Default output folder |
|---|---|---|
| `train` | Main AE32→SimSiam→frozen-feature classifier | `outputs/main` |
| `baseline` | Frozen random-projection classifier baseline | `outputs/raw` |
| `components` | Main, AE-only, scratch SSL and scratch supervised CNN | `outputs/components` |
| `ae64` | Matched AE64-only | `outputs/ae64` |
| `ae64-simsiam` | Refine AE64; reads the previous output via `--ae-dir` | `outputs/ae64_simsiam` |
| `augmentations` | Six separate augmentation/freeze conditions | `outputs/augmentation` |
| `train-history` | Main training with per-epoch recording | `outputs/history` |
| `evaluate` | Held-out classification or external data evaluation | `outputs/evaluation` |
| `reconstruct` | Seeded Phase I denoising evaluation | `outputs/reconstruction` |
| `predict` | Prediction-only interface | `outputs` |
| `plot-history` | Standalone SVG from aggregate histories | Input history directory |

Each full training command uses the configured schedule. For experiment
comparability and the distinction between the two AE-only results, read
[Experiment definitions](experiments.md).

## Inspect results without private inputs

```bash
python -m soma plot-history --history-dir results/rev2/history --out outputs/supplied_histories.svg
python -m unittest discover -s tests -v
```

The public result files contain aggregate metrics and epoch histories. Complete
scientific retraining requires the private study inputs; short CPU tests use
generated signals and do not establish the reported accuracy.
