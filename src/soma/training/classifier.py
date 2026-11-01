"""Frozen-feature classifier fitting and validation checkpoint selection."""

import torch

from soma.common import cpu_state, seed_all
from soma.models import Head


def fit_head(z, y, ia, iv, fold, config, device):
    mu = z[ia].mean(0)
    sd = z[ia].std(0, unbiased=False).clamp(min=1e-6)
    z = (z - mu) / sd
    yt = y.to(device)
    seed_all(5011 + (fold - 1) * 113)
    head = Head().to(device)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=config["head_lr"],
        weight_decay=config["head_weight_decay"],
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["head_epochs"])
    gen = torch.Generator().manual_seed(3000 + fold - 1)
    best = (-1, float("inf"))
    state = None
    epoch = 0
    for ep in range(config["head_epochs"]):
        head.train()
        for jj in torch.randperm(len(ia), generator=gen).split(config["batch_size"]):
            ix = ia[jj]
            loss = torch.nn.functional.cross_entropy(head(z[ix]), yt[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            logits = head(z[iv])
            acc = int((logits.argmax(1) == yt[iv]).sum())
            ce = float(torch.nn.functional.cross_entropy(logits, yt[iv]))
        if acc > best[0] or (acc == best[0] and ce < best[1]):
            best = (acc, ce)
            state = cpu_state(head)
            epoch = ep + 1
    head.load_state_dict(state)
    head.eval()
    return head, mu, sd, dict(val_correct=best[0], val_ce=best[1], epoch=epoch)
