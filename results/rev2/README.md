# Second revision aggregate records

These are supplied or previously verified aggregate records, not outputs from
a full retraining during the layout change. Dataset tensors, checkpoint weights
and per-window predictions are deliberately absent.

| Record | Meaning |
|---|---|
| `components.json` | Main, original AE-only, scratch SSL and scratch supervised CNN |
| `ae64_only.json`, `ae64_simsiam.json` | Separate matched-initialization AE64 experiment |
| `raw_baseline_reported.json` | SI Table S5 baseline summary, marked as reported |
| `c2.json` | Three main checkpoints evaluated on the one C2 recording |
| `reconstruction.json` | Supplied held-out seeded-denoising summary |
| `history/` | Supplied aggregate per-epoch train/validation histories and selections |
| `source_provenance.json` | Original source hashes and first integration hashes |

The provenance hashes refer to the first integration snapshot. Current module
locations are mapped in [path-mapping.json](../../docs/path-mapping.json), and the
new source hashes are recorded in `layout_provenance.json`. Scientific result
JSON records and history values are unchanged by the reorganization.
