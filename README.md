# SOMA optical myography research code

SOMA combines multichannel optical myography acquisition with a three-stage
representation-learning and gesture-classification pipeline. This `rev2` branch
contains the maintained second-revision implementation as the installable
`soma` Python package. Private second-revision data, weights and per-window
predictions are not included.

The project and installable distribution are named **SOMA** (formerly TRAiLL).
The Python import and command remain `soma`. Historical `traill_daq` and
`traill_data_preparation` module names remain available for compatibility;
they are implementation names rather than separate projects.

## Start here

```bash
python -m pip install -e .
python -m soma --help
python -m soma train --data-dir /private/soma/DATA --device cuda
```

Use Python 3.12 and the PyTorch 2.6 build appropriate for your machine. Commands
also accept `--device cpu`. See [Quick start](docs/getting-started.md) for the
full train/evaluate workflow, optional dependencies and available commands.

## Repository guide

| Location | Purpose |
|---|---|
| [src/soma](src/soma/) | Current training, experiments, evaluation, plotting and packaged configs |
| [src/traill_daq](src/traill_daq/README.md) | Serial acquisition and live visualization |
| [src/traill_data_preparation](src/traill_data_preparation/README.md) | Recording preparation and exploratory analysis |
| [hardware](hardware/README.md) | Pico/LED firmware, CAD asset and desktop CAD helpers |
| [results/rev2](results/rev2/README.md) | Supplied aggregate metrics and training histories |
| [data](data/README.md) | Historical public assets; private local inputs go in ignored `data/private/` |
| [tests](tests/) | CPU regression, import and repository-structure tests |
| [legacy](legacy/README.md) | Earlier training/evaluation implementations retained for reference |

## Documentation

- [Quick start and commands](docs/getting-started.md)
- [Data contract and evaluation protocol](docs/data.md)
- [Experiment definitions and reported results](docs/experiments.md)
- [Repository architecture](docs/repository.md)
- [Migration from the earlier layout](docs/migration.md)
- [Validation and limitations](docs/validation.md)
- [Change history](docs/changelog.md)

Run `python -m unittest discover -s tests -v` after installing the package.
Start new experiments with the `soma` commands; earlier root-level scripts and
`experiment/` tools now live under `legacy/` and have different checkpoint and
experimental definitions. Data identities follow the collaborator's description;
the legacy pseudo-session generator is not a source of independent participants.
