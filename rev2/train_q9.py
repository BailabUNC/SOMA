"""Re-run the project LOSO pipeline while recording fold-specific train/validation curves.

Outputs are written to outputs_q9/ so the original results remain untouched.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import AE32, Encoder64, Head, Predictor, Projector, ae_norm
from train import (
    augment_gpu,
    cpu_state,
    extract,
    fold_indices,
    load_data,
    metric,
    save_json,
    seed_all,
)


ROOT = Path(__file__).resolve().parent


def batches(indices, batch_size):
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def scalar(x):
    return float(x.detach().cpu())


@torch.no_grad()
def ae_eval(model, clean, noisy, batch_size=32):
    model.eval()
    total_l1 = total_mse = total_n = 0.0
    for start in range(0, len(clean), batch_size):
        target = clean[start:start + batch_size]
        pred = model(noisy[start:start + batch_size])
        n = target.numel()
        total_l1 += scalar(F.smooth_l1_loss(pred, target, reduction="sum"))
        total_mse += scalar(F.mse_loss(pred, target, reduction="sum"))
        total_n += n
    mse = total_mse / total_n
    return {"smooth_l1": total_l1 / total_n, "mse": mse, "rmse_0_1": math.sqrt(mse)}


def phase1(raw_x, ia, iv, fold, cfg, device, out):
    tr = cfg["training"]
    seed_all(59 + fold * 101)
    x = ae_norm(raw_x).to(device)
    q = tr["pretrain_quantization"]
    xq = (x / q).round() * q if q > 0 else x
    model = AE32().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr["ae_lr"])
    order_gen = torch.Generator().manual_seed(1000 + fold)
    noise_gen = torch.Generator().manual_seed(70000 + fold)
    val_noise = torch.randn(xq[iv].shape, generator=noise_gen, dtype=xq.dtype).to(device) * 0.1
    val_noisy = xq[iv] + val_noise
    history = []

    for ep in range(tr["ae_epochs"]):
        model.train()
        perm = torch.randperm(len(ia), generator=order_gen)
        train_sum = train_n = 0
        for pos in perm.split(tr["pretrain_batch"]):
            ix = ia[pos]
            target = xq[ix]
            pred = model(target + 0.1 * torch.randn_like(target))
            loss = F.smooth_l1_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_sum += scalar(loss) * len(ix)
            train_n += len(ix)
        train_metric = train_sum / train_n
        val_metric = ae_eval(model, xq[iv], val_noisy, tr["pretrain_batch"])
        history.append({"epoch": ep + 1, "train_smooth_l1": train_metric,
                        "val_smooth_l1": val_metric["smooth_l1"],
                        "val_mse": val_metric["mse"], "val_rmse_0_1": val_metric["rmse_0_1"]})
        if (ep + 1) % 20 == 0:
            print(f"fold{fold} AE {ep + 1}/{tr['ae_epochs']} train={train_metric:.6f} val={val_metric['smooth_l1']:.6f}", flush=True)

    torch.save({"ae": cpu_state(model), "train": ia, "val": iv}, out / f"fold{fold}_phase1.pth")
    save_json(out / f"fold{fold}_phase1_history.json", history)
    return model, x, xq


def ssl_objective(encoder, projector, predictor, x, batch_size=32, seed=0):
    devices = [torch.cuda.current_device()] if x.is_cuda else []
    total = total_n = 0.0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if x.is_cuda:
            torch.cuda.manual_seed_all(seed)
        for xb in x.split(batch_size):
            v1, v2 = augment_gpu(xb), augment_gpu(xb)
            z1, z2 = projector(encoder(v1)), projector(encoder(v2))
            p1, p2 = predictor(z1), predictor(z2)
            loss = -0.5 * (F.cosine_similarity(p1, z2.detach()).mean() +
                           F.cosine_similarity(p2, z1.detach()).mean())
            total += scalar(loss) * len(xb)
            total_n += len(xb)
    return total / total_n


def phase2(ae, xq, ia, iv, fold, cfg, device, out):
    tr = cfg["training"]
    seed_all(59 + fold * 101)
    encoder = Encoder64(ae, "new").to(device)
    for j, block in enumerate(encoder.blocks):
        for p in block.parameters():
            p.requires_grad = j >= 3
    projector, predictor = Projector().to(device), Predictor().to(device)
    opt = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad] +
                           list(projector.parameters()) + list(predictor.parameters()),
                           lr=tr["ssl_lr"])
    order_gen = torch.Generator().manual_seed(20000 + fold)
    history = []

    for ep in range(tr["ssl_epochs"]):
        encoder.train(); projector.train(); predictor.train()
        perm = torch.randperm(len(ia), generator=order_gen)
        train_sum = train_n = 0
        for pos in perm.split(tr["pretrain_batch"]):
            xb = xq[ia[pos]]
            z1, z2 = projector(encoder(augment_gpu(xb))), projector(encoder(augment_gpu(xb)))
            p1, p2 = predictor(z1), predictor(z2)
            loss = -0.5 * (F.cosine_similarity(p1, z2.detach()).mean() +
                           F.cosine_similarity(p2, z1.detach()).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_sum += scalar(loss) * len(xb)
            train_n += len(xb)
        train_loss = train_sum / train_n
        encoder.eval(); projector.eval(); predictor.eval()
        val_loss = ssl_objective(encoder, projector, predictor, xq[iv],
                                 tr["pretrain_batch"], seed=90000 + fold * 1000 + ep)
        history.append({"epoch": ep + 1, "train_simsiam_loss": train_loss,
                        "val_simsiam_loss": val_loss})
        if (ep + 1) % 25 == 0:
            print(f"fold{fold} SSL {ep + 1}/{tr['ssl_epochs']} train={train_loss:.6f} val={val_loss:.6f}", flush=True)

    torch.save({"encoder": cpu_state(encoder), "projector": cpu_state(projector),
                "predictor": cpu_state(predictor), "train": ia, "val": iv},
               out / f"fold{fold}_phase2.pth")
    save_json(out / f"fold{fold}_phase2_history.json", history)
    return encoder


def phase3(encoder, x, y, ia, iv, te, fold, cfg, device, out):
    tr = cfg["training"]
    encoder.eval().requires_grad_(False)
    z = extract(encoder, x)
    mu = z[ia].mean(0)
    sd = z[ia].std(0, unbiased=False).clamp(min=1e-6)
    z = (z - mu) / sd
    yt = y.to(device)
    seed_all(5011 + (fold - 1) * 113)
    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=tr["head_lr"], weight_decay=0.0001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["head_epochs"])
    order_gen = torch.Generator().manual_seed(3000 + fold - 1)
    best = (-1, float("inf")); best_state = None; selected_epoch = 0
    history = []

    for ep in range(tr["head_epochs"]):
        head.train()
        perm = torch.randperm(len(ia), generator=order_gen)
        for pos in perm.split(32):
            ix = ia[pos]
            loss = F.cross_entropy(head(z[ix]), yt[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            train_logits, val_logits = head(z[ia]), head(z[iv])
            train_ce = scalar(F.cross_entropy(train_logits, yt[ia]))
            val_ce = scalar(F.cross_entropy(val_logits, yt[iv]))
            train_acc = float((train_logits.argmax(1) == yt[ia]).float().mean())
            val_acc = float((val_logits.argmax(1) == yt[iv]).float().mean())
            val_correct = int((val_logits.argmax(1) == yt[iv]).sum())
        history.append({"epoch": ep + 1, "train_ce": train_ce, "val_ce": val_ce,
                        "train_accuracy": train_acc, "val_accuracy": val_acc})
        if val_correct > best[0] or (val_correct == best[0] and val_ce < best[1]):
            best = (val_correct, val_ce)
            best_state = cpu_state(head)
            selected_epoch = ep + 1
        if (ep + 1) % 25 == 0:
            print(f"fold{fold} MLP {ep + 1}/{tr['head_epochs']} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best_epoch={selected_epoch}", flush=True)

    checkpoint = {"encoder": cpu_state(encoder), "classifier": best_state, "mu": mu.cpu(),
                  "sd": sd.cpu(), "train": ia, "val": iv, "test": te,
                  "selected_epoch": selected_epoch, "val_correct": best[0],
                  "selected_val_ce": best[1],
                  "config": {**tr, "encoder_frozen_phase3": True,
                             "checkpoint_selection": "highest validation accuracy; tie-break by lowest validation cross-entropy"}}
    torch.save(checkpoint, out / f"fold{fold}.pth")
    save_json(out / f"fold{fold}_phase3_history.json", history)
    save_json(out / f"fold{fold}_selection.json", {
        "selected_epoch": selected_epoch, "val_correct": best[0],
        "val_n": len(iv), "val_accuracy": best[0] / len(iv), "val_ce": best[1],
        "criterion": "highest validation accuracy; tie-break by lowest validation cross-entropy"})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "history")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "DATA")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cudnn.allow_tf32 = not cfg["training"].get("full_precision", False)
    torch.backends.cuda.matmul.allow_tf32 = False
    x, y, sessions = load_data(args.data_dir, cfg)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", cfg)
    for fold in (1, 2, 3):
        ia, iv, te = fold_indices(y, sessions, fold)
        print(f"Starting fold {fold}: train={len(ia)}, validation={len(iv)}, test={len(te)}", flush=True)
        ae, xnorm, xq = phase1(x, ia, iv, fold, cfg, args.device, out)
        encoder = phase2(ae, xq, ia, iv, fold, cfg, args.device, out)
        phase3(encoder, xnorm, y, ia, iv, te, fold, cfg, args.device, out)
    print(f"Completed. Fold histories and checkpoints are in {out}", flush=True)


if __name__ == "__main__":
    main()
