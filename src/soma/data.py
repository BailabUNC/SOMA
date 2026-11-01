"""Private tensor input contract and fixed nested split."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def fold_indices(y, sessions, fold):
    te = np.asarray(sessions[fold - 1])
    tr = np.concatenate([s for i, s in enumerate(sessions) if i != fold - 1])
    ia, iv = train_test_split(
        np.arange(len(tr)), test_size=0.2, random_state=42, stratify=y[tr].numpy()
    )
    return (tr[ia], tr[iv], te)


def load_data(folder, config):
    """Load collaborator-defined cohorts; preserve the checkpoint index layout.

    The permutation restores global indices and does not form new cohorts.
    Additional metadata is accepted but never used as model input.
    """
    folder = Path(folder)
    sizes = config["session_sizes"]
    if len(sizes) != 3 or any(n <= 0 for n in sizes):
        raise ValueError("session_sizes must contain three positive cohort sizes")
    sessions = np.split(
        np.random.default_rng(config["session_seed"]).permutation(sum(sizes)),
        np.cumsum(sizes)[:-1],
    )
    x = torch.empty(sum(sizes), 512, 48)
    labels = torch.empty(sum(sizes), dtype=torch.long)
    required = {"features", "labels", "subject_ids"}
    for i, (ix, n) in enumerate(zip(sessions, sizes), 1):
        path = folder / f"S{i}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path}: supply private S1.pt, S2.pt and S3.pt via --data-dir"
            )
        d = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(d, dict) or not required.issubset(d):
            raise ValueError(f"{path}: required keys are {sorted(required)}")
        if (
            d["features"].shape != (n, 512, 48)
            or d["labels"].shape != (n,)
            or d["subject_ids"].shape != (n,)
        ):
            raise ValueError(
                f"{path}: expected features [{n},512,48] and labels/subject_ids [{n}]"
            )
        if not d["subject_ids"].eq(i).all():
            raise ValueError(
                f"{path}: expected subject_ids={i}; IDs must reflect the supplied provenance"
            )
        if (
            not torch.isfinite(d["features"]).all()
            or not torch.isfinite(d["labels"]).all()
        ):
            raise ValueError(f"{path}: non-finite features or labels")
        if not torch.equal(d["labels"], d["labels"].long()):
            raise ValueError(f"{path}: labels must be integers")
        x[ix] = d["features"]
        labels[ix] = d["labels"]
    if sorted(labels.unique().tolist()) != config["labels"]:
        raise ValueError("Observed labels do not match config labels")
    y = torch.tensor([config["labels"].index(int(v)) for v in labels])
    return x, y, sessions
