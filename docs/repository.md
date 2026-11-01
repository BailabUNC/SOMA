# Repository architecture

The maintained pipeline has one package, one command index and one set of
packaged defaults. Applications that need serial hardware or recorded-CSV
preparation are separate packages with optional dependencies.

| Path | Responsibility |
|---|---|
| `src/soma/cli.py` | Dispatch commands, importing only the requested workflow |
| `src/soma/models.py` | Model classes and signal augmentations |
| `src/soma/data.py` | Private tensor validation and nested fold indices |
| `src/soma/common.py` | Seeds, state export, feature extraction and metrics |
| `src/soma/paths.py` | Config resource paths and workspace defaults |
| `src/soma/configs/` | Main schedule and separate augmentation/head settings |
| `src/soma/training/` | Main, raw baseline, history recording and shared classifier fitting |
| `src/soma/experiments/` | Component, matched AE64 and augmentation studies |
| `src/soma/evaluation/` | Labeled evaluation, prediction-only and denoising metrics |
| `src/soma/plotting/` | Aggregate history SVG plotting |
| `src/traill_daq/` | Serial acquisition application and profile resources |
| `src/traill_data_preparation/` | Recording preparation and spatial analysis utilities |
| `hardware/` | Firmware and desktop CAD artifacts/helpers |
| `results/rev2/` | Supplied aggregate experiment records, not generated output |
| `legacy/` | Earlier implementations with their internal import topology preserved |

All current imports use explicit package paths. The classifier-fitting helper
is shared by experiments and baselines rather than imported from an experiment
script. Plotting is import-safe and runs only when its command is invoked.
Models remain state-dictionary compatible with the supplied checkpoints.

Only current packages under `src/` are installed. Legacy dependencies are
optional and archived scripts are not imported by the maintained pipeline.
Public raw assets retain their existing `data/` paths. Default private tensors
live under ignored `data/private/`, generated outputs under ignored `outputs/`,
and checkpoints are ignored at any path.
