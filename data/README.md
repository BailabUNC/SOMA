# Data locations

The participant-named CSV recordings and `baseline_arrays/` are historical public
assets inherited from the earlier repository. CSV content is stored through
Git LFS; a normal checkout without LFS may contain pointers. Their paths and
contents are unchanged by the `rev2` code reorganization.

Second-revision `S1.pt`, `S2.pt`, `S3.pt`, `C2.pt` and weights are private and are
not added to this repository. Put local copies in ignored `data/private/`, or
supply an explicit `--data-dir` to the maintained `soma` commands. The authoritative
shape, label, normalization and split definitions are in [the data guide](../docs/data.md).

`loso/README.md` documents the earlier mixed pseudo-session workflow. It is not
a statement that the private second-revision S files were generated that way.
The current package does not regenerate independent participants from pooled
windows. Recording preparation tools are documented [here](../src/traill_data_preparation/README.md).
