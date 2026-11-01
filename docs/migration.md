# Migration from the earlier layout

## Project name

The installable distribution is now named `SOMA`; its import and command remain
`soma`. Continue installing from this checkout with `python -m pip install -e .`.
If you previously installed this checkout as `soma-traill`, first remove that
editable distribution with `python -m pip uninstall soma-traill`, then reinstall
from the checkout. No checkpoint keys or private asset paths change because of
the name update. Existing acquisition/preparation module names are retained.

Renaming the GitHub repository is a separate repository-administration action;
changing this Python metadata does not rename the hosted repository.

This is a directory/import/documentation reorganization of the `rev2` branch,
not a new scientific experiment. The old `revision` branch and the preceding
`rev2` commit remain available in Git history. No private input folders are
migrated or inferred automatically; continue to use explicit local paths.

## Current commands

| Previous entry | Current command |
|---|---|
| `python rev2/train.py` | `python -m soma train` |
| `python rev2/raw_train.py` | `python -m soma baseline` |
| `python rev2/components.py` | `python -m soma components` |
| `python rev2/ae64_only.py` | `python -m soma ae64` |
| `python rev2/ae64_simsiam.py` | `python -m soma ae64-simsiam` |
| `python rev2/ablation.py` | `python -m soma augmentations` |
| `python rev2/train_q9.py` | `python -m soma train-history` |
| `python rev2/evaluate.py` | `python -m soma evaluate` |
| `python rev2/evaluate_reconstruction.py` | `python -m soma reconstruct` |
| `python rev2/predict.py` | `python -m soma predict` |
| `python rev2/plot_q9_curves.py` | `python -m soma plot-history` |

Command arguments and checkpoint tensor keys retain their meaning. Default
outputs move from `rev2/outputs/` to `outputs/`; existing checkpoints can be read
in place with `--models-dir`, `--checkpoint` or `--ae-dir`. Training configuration
files move into `src/soma/configs/`; private input defaults move from `rev2/DATA/`
to `data/private/`. Provide `--data-dir` to continue using any existing location.

## Other code and records

| Previous location | New location |
|---|---|
| Root `train_phase*.py`, `path_config.py` | `legacy/` |
| `experiment/`, `tools/` | `legacy/experiment/`, `legacy/tools/` |
| `traill_daq/` | `src/traill_daq/`; run `python -m traill_daq.run` |
| `traill_data_preparation/` | `src/traill_data_preparation/`; run modules with `python -m` |
| `CAD/`, `firmware/` | `hardware/cad/`, `hardware/firmware/` |
| `rev2/results/` | `results/rev2/` |
| `rev2/tests/` | `tests/` |
| Long `rev2/README.md` | Focused guides under `docs/` |
| Old dependency-free `uv.lock` | Removed; package dependencies are in `pyproject.toml` |

The machine-readable source path map is [path-mapping.json](path-mapping.json).
It also records the original-to-current Python module locations. The old
requirements snapshot is retained at `legacy/requirements-original.txt`.
Use the new root install rather than the former CUDA-specific requirements file.

For legacy runs, invoke `python legacy/...` from the repository root and see
[Legacy usage](../legacy/README.md). Shared legacy data/checkpoint/output roots
remain at the repository root; Python imports inside that archive remain local.
