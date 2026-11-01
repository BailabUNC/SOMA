# Recording preparation and exploratory analysis

Install optional dependencies with `python -m pip install -e '.[preparation]'`.
Run these modules from the repository root to preserve historical relative
`data/` defaults, for example `python -m traill_data_preparation.traill_dataset --help`.

| Module | Purpose |
|---|---|
| `traill_dataset` | Segment/preprocess recorded CSV signals |
| `traill_dataset_concat` | Combine prepared tensors |
| `traill_dataset_augmentation` | Export augmentations for the historical workflow |
| `channel_reduction` | Exploratory LDA/channel reduction |
| `ring_visualization` | Inspect spatial signals |
| `traill_loso_session_resplit` | Historical pooled-window pseudo-session resplitting |

The resplit utility does not establish independent participants. It must not be
used to manufacture the private second-revision S1/S2/S3 participant cohorts.
Use [the data contract](../../docs/data.md) for the maintained pipeline.
