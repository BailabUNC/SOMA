# Data contract and evaluation protocol


This is a source-and-summary distribution. The second-revision datasets, model
weights, per-window labels/predictions and manuscript working documents are not
included. Running the scientific experiments requires private inputs supplied
locally. The aggregate values in [`results/rev2/`](../results/rev2/) are provided results,
not a claim that full-duration training was repeated during integration.

According to the collaborator's data interpretation adopted for this revision,
`S1.pt`, `S2.pt` and `S3.pt` represent three participants; `C2.pt` is an additional
recording of Participant 2 on another day. This interpretation must accompany the
private files. File names or numeric IDs alone cannot establish provenance.
The historical `../data/loso/` documentation and resplitting script describe a
different pseudo-session workflow. Do not use that generator to manufacture
participant cohorts for these experiments.


## Required tensors and split


Each `S{i}.pt` is a tensor dictionary with the following required fields.
Additional metadata is accepted and is not used as model input.

| Field | Shape and meaning |
|---|---|
| `features` | finite tensor `[N,512,48]` |
| `labels` | integer vector `[N]`, using the label IDs in `src/soma/configs/main.json` |
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


## External C2 evaluation

C2 contains 204 labeled windows. Under the adopted collaborator interpretation,
it is a recording of Participant 2 on a different day. Folds 1 and 3 have seen
that participant's first recording; Fold 2 has not. Fold 2 therefore combines
participant and day changes. These are three checkpoints applied to one external
recording, not three independent external participants.

Use `python -m soma evaluate --external /private/soma/DATA/C2.pt` for evaluation.
External labels never enter training or checkpoint selection. Per-window outputs
are written only to the local ignored `outputs/` tree.

Historical public recordings and baseline arrays retain their existing paths
under `data/`; moving the code has not changed those files. The historical mixed
pseudo-session definition is in `data/loso/README.md`. Keep its interpretation
separate from the collaborator-supplied second-revision tensors.
