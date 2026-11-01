"""Shared deterministic training, metrics and serialization helpers."""

import json
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def metric(y, p):
    return dict(
        correct=int((y == p).sum()),
        n=len(y),
        accuracy=float(accuracy_score(y, p)),
        balanced_accuracy=float(balanced_accuracy_score(y, p)),
        macro_f1=float(
            f1_score(y, p, labels=np.arange(24), average="macro", zero_division=0)
        ),
    )


def summarize(rows):
    r = {"correct": [d["correct"] for d in rows]}
    for k in ("accuracy", "macro_f1", "balanced_accuracy"):
        a = np.array([d[k] for d in rows])
        r[k] = {"mean": float(a.mean()), "sd_population": float(a.std())}
    return r


def cpu_state(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def augment_gpu(x):
    n, c, length = x.shape
    dev = x.device
    cut = torch.randint(int(0.8 * length), length + 1, (n, 1), device=dev)
    start = (torch.rand(n, 1, device=dev) * (length - cut + 1)).floor()
    shift = torch.randint(-length // 10, length // 10 + 1, (n, 1), device=dev)
    t = (torch.arange(length, device=dev)[None] - shift) % length
    flip = torch.rand(n, 1, device=dev) < 0.5
    t = torch.where(flip, length - 1 - t, t)
    pos = ((t + 0.5) * cut / length - 0.5).clamp(min=0)
    pos = torch.minimum(pos, cut - 1) + start
    lo = pos.long()
    hi = (lo + 1).clamp(max=length - 1)
    weight = (pos - lo)[:, None]
    result = (
        x.gather(2, lo[:, None].expand(-1, c, -1)) * (1 - weight)
        + x.gather(2, hi[:, None].expand(-1, c, -1)) * weight
    )
    result = result + 0.01 * torch.randn_like(result)
    drop = torch.rand(n, c, device=dev).argsort(1)[:, : max(1, c // 10)]
    mask = torch.ones(n, c, device=dev)
    mask.scatter_(1, drop, 0)
    mask = torch.where(torch.rand(n, 1, device=dev) < 0.2, mask, torch.ones_like(mask))
    width = torch.randint(0, length // 5 + 1, (n, 1), device=dev)
    begin = (torch.rand(n, 1, device=dev) * (length - width + 1)).floor()
    timeline = torch.arange(length, device=dev)[None]
    keep = (timeline < begin) | (timeline >= begin + width)
    return (result * mask[:, :, None] * keep[:, None]).clamp(0, 1)


@torch.no_grad()
def extract(en, x):
    return torch.cat([en(b) for b in x.split(64)])
