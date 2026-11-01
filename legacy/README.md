# Earlier training and experiment workflows

This directory preserves the implementations that preceded the collaborator's
second-revision pipeline. Use the maintained `python -m soma` commands for new
second-revision runs. Model definitions, checkpoints and experiment protocols
in this archive are different and should not be silently substituted.

Run archived scripts from the repository root:

```bash
python -m pip install -e '.[legacy]'
python legacy/train_phase1_autoencoder.py --dry-run
python legacy/train_phase2_simsiam.py --help
python legacy/train_phase3_end_task.py --dry-run
python legacy/experiment/loso/run_loso_phase1.py --help
```

`path_config.ROOT` continues to point at the repository root for historical
`data/`, `checkpoints/` and `outputs/` folders. Internal import names remain
local to `legacy/`; for interactive use, add `legacy/` to `PYTHONPATH` separately.
The old `tools/` and `experiment/` folders retain their relative layout here.
`requirements-original.txt` is an archived reference,
not current installation instructions. All archived Python files are syntax
checked; full numerical revalidation of earlier experiments is not claimed.
