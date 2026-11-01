# Change history

## SOMA project name

- Use SOMA for distribution metadata and current command-line descriptions.
- Keep `soma` imports/commands and historical acquisition/preparation modules
  compatible. Checkpoint and dataset formats are unchanged.
- Repository-hosting renaming remains separate from Python project metadata.

## Package and repository organization

- Replace the flat `rev2/` scripts with an installable `src/soma/` package and
  a unified `python -m soma` command menu.
- Separate shared data/metrics/classifier helpers from workflow entrypoints.
- Move acquisition and recording preparation into installable source packages;
  fix their existing bare/missing imports.
- Group firmware and CAD under `hardware/`, aggregate records under
  `results/rev2/`, and earlier implementations under `legacy/`.
- Split documentation by task; add directory and command migration tables.
- Keep current model operations, configured schedules, checkpoint tensor keys,
  historical raw assets, firmware content and scientific result records intact.

## First second-revision integration


Based on `revision` commit `0ca0f06f4d4f855cd9a512a8e839196392372bdc`.

- Add the collaborator's three-stage model and raw-input baseline in `rev2/`.
- Add component experiments including SSL-only and end-to-end supervised CNN
  from random initialization using the supplemental `components.py` source.
- Add the matched-initialization AE64-only and AE64→SimSiam pair.
- Add complete learning-history recording and the supplied aggregate histories.
- Add held-out/external classification and seeded denoising evaluation commands.
- Standardize local data/config/checkpoint/output arguments; allow extra tensor
  metadata; support CPU execution without changing schedules or seeds.
- Keep experiment definitions and separate results explicit: main AE32→SSL,
  component AE-only and matched AE64-only are different conditions.
- Include aggregate results and source provenance while excluding second-revision
  tensors, weights, per-window outputs and manuscript working files.
- Preserve the pre-existing acquisition, hardware and earlier experiment tree.
