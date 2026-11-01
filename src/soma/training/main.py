"""Main AE32 pretraining, partial SimSiam refinement and frozen-feature classification."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from soma.common import (
    augment_gpu,
    cpu_state,
    extract,
    metric,
    save_json,
    seed_all,
    summarize,
)
from soma.data import fold_indices, load_data
from soma.models import AE32, Encoder64, Head, Predictor, Projector, ae_norm
from soma.paths import DEFAULT_CONFIG, DEFAULT_DATA_DIR, ROOT


def train_fold(
    raw,
    y,
    sessions,
    fold,
    out,
    device,
    ae_epochs=60,
    ssl_epochs=100,
    head_epochs=220,
    ae_lr=0.001,
    ssl_lr=0.001,
    head_lr=0.001,
    pretrain_batch=32,
    pretrain_quantization=0.0,
    full_precision=False,
):
    path = out / f"fold{fold}.pth"
    ia, iv, te = fold_indices(y, sessions, fold)
    seed_all(59 + fold * 101)
    torch.backends.cudnn.allow_tf32 = not full_precision
    torch.backends.cuda.matmul.allow_tf32 = False
    x = ae_norm(raw).to(device)
    yt = y.to(device)
    pretrain_x = (
        (x / pretrain_quantization).round() * pretrain_quantization
        if pretrain_quantization > 0
        else x
    )
    ae = AE32().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=ae_lr)
    gen = torch.Generator().manual_seed(1000 + fold)
    for ep in range(ae_epochs):
        ae.train()
        for jj in torch.randperm(len(ia), generator=gen).split(pretrain_batch):
            xb = pretrain_x[ia[jj]]
            pred = ae(xb + 0.1 * torch.randn_like(xb))
            loss = torch.nn.functional.smooth_l1_loss(pred, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            print(f"Fold{fold} AE {ep + 1}: {loss.item():.6f}", flush=True)
    torch.save(
        dict(ae=cpu_state(ae), train=ia, val=iv, test=te),
        out / f"fold{fold}_phase1.pth",
    )
    seed_all(59 + fold * 101)
    en = Encoder64(ae, "new").to(device)
    for j, b in enumerate(en.blocks):
        for p in b.parameters():
            p.requires_grad = j >= 3
    pr = Projector().to(device)
    pd = Predictor().to(device)
    opt = torch.optim.Adam(
        [p for p in en.parameters() if p.requires_grad]
        + list(pr.parameters())
        + list(pd.parameters()),
        lr=ssl_lr,
    )
    gen = torch.Generator().manual_seed(20000 + fold)
    for ep in range(ssl_epochs):
        en.train()
        pr.train()
        pd.train()
        for jj in torch.randperm(len(ia), generator=gen).split(pretrain_batch):
            xb = pretrain_x[ia[jj]]
            z1 = pr(en(augment_gpu(xb)))
            z2 = pr(en(augment_gpu(xb)))
            p1 = pd(z1)
            p2 = pd(z2)
            loss = -0.5 * (
                torch.nn.functional.cosine_similarity(p1, z2.detach()).mean()
                + torch.nn.functional.cosine_similarity(p2, z1.detach()).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 25 == 0:
            print(f"Fold{fold} SSL {ep + 1}: {loss.item():.6f}", flush=True)
    torch.save(
        dict(
            encoder=cpu_state(en),
            projector=cpu_state(pr),
            predictor=cpu_state(pd),
            train=ia,
            val=iv,
            test=te,
        ),
        out / f"fold{fold}_phase2.pth",
    )
    en.eval()
    for p in en.parameters():
        p.requires_grad = False
    z = extract(en, x)
    mu = z[ia].mean(0)
    sd = z[ia].std(0, unbiased=False).clamp(min=1e-06)
    torch.save(
        dict(
            encoder=cpu_state(en),
            mu=mu.cpu(),
            sd=sd.cpu(),
            train=ia,
            val=iv,
            test=te,
            frozen=True,
        ),
        out / f"fold{fold}_frozen_encoder.pth",
    )
    z = (z - mu) / sd
    seed_all(5011 + (fold - 1) * 113)
    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=head_lr, weight_decay=0.0001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=head_epochs)
    gen = torch.Generator().manual_seed(3000 + fold - 1)
    best = (-1, float("inf"))
    state = None
    selected = 0
    for ep in range(head_epochs):
        head.train()
        for jj in torch.randperm(len(ia), generator=gen).split(32):
            ix = ia[jj]
            loss = torch.nn.functional.cross_entropy(head(z[ix]), yt[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            v = head(z[iv])
            acc = int((v.argmax(1) == yt[iv]).sum())
            ce = float(torch.nn.functional.cross_entropy(v, yt[iv]))
        if acc > best[0] or (acc == best[0] and ce < best[1]):
            best = (acc, ce)
            state = cpu_state(head)
            selected = ep + 1
    ck = dict(
        encoder=cpu_state(en),
        classifier=state,
        ae=cpu_state(ae),
        projector=cpu_state(pr),
        predictor=cpu_state(pd),
        mu=mu.cpu(),
        sd=sd.cpu(),
        train=ia,
        val=iv,
        test=te,
        selected_epoch=selected,
        val_correct=best[0],
        config=dict(
            ae_epochs=ae_epochs,
            ssl_epochs=ssl_epochs,
            head_epochs=head_epochs,
            encoder_frozen_phase3=True,
            ae_noise=0.1,
            ssl_last_blocks=2,
            ae_lr=ae_lr,
            ssl_lr=ssl_lr,
            head_lr=head_lr,
            pretrain_batch=pretrain_batch,
            pretrain_quantization=pretrain_quantization,
            full_precision=full_precision,
            head_batch=32,
            weight_decay=0.0001,
            seed=59 + fold * 101,
        ),
    )
    torch.save(ck, path)
    print(f"Fold{fold} saved; selected epoch {selected}", flush=True)
    return ck


def evaluate(raw, y, sessions, folder, device):
    rows = []
    all_pred = np.zeros(len(y), dtype=int)
    for fold in (1, 2, 3):
        ck = torch.load(
            folder / f"fold{fold}.pth", weights_only=False, map_location="cpu"
        )
        torch.backends.cudnn.allow_tf32 = not ck["config"].get("full_precision", False)
        torch.backends.cuda.matmul.allow_tf32 = False
        en = Encoder64().to(device)
        en.load_state_dict(ck["encoder"])
        en.eval()
        h = Head().to(device)
        h.load_state_dict(ck["classifier"])
        h.eval()
        te = ck["test"]
        with torch.no_grad():
            z = extract(en, ae_norm(raw[te]).to(device))
            logits = h((z - ck["mu"].to(device)) / ck["sd"].to(device))
            p = logits.argmax(1).cpu().numpy()
        yy = y[te].numpy()
        r = metric(yy, p)
        rows.append(r)
        all_pred[te] = p
        save_json(
            folder / f"fold{fold}_predictions.json",
            dict(
                true=yy.tolist(),
                pred=p.tolist(),
                logits=logits.cpu().tolist(),
                original_indices=te.tolist(),
                confusion=confusion_matrix(yy, p, labels=np.arange(24)).tolist(),
                metrics=r,
            ),
        )
    result = dict(folds=rows, summary=summarize(rows))
    save_json(folder / "results.json", result)
    print(json.dumps(result["summary"]), flush=True)
    return (result, all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "main")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    x, y, sessions = load_data(args.data_dir, config)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config)
    for f in (1, 2, 3):
        train_fold(x, y, sessions, f, out, args.device, **config["training"])
    evaluate(x, y, sessions, out, args.device)


if __name__ == "__main__":
    main()
