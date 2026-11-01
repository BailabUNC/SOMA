"""Main, AE-only, scratch SimSiam and end-to-end supervised CNN experiments."""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path

import torch

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
from soma.paths import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_HEAD_CONFIG, ROOT
from soma.training.classifier import fit_head
from soma.training.main import train_fold


def scratch_ssl(x, ia, fold, cfg, device):
    seed_all(59 + fold * 101)
    en = Encoder64().to(device)
    pr = Projector().to(device)
    pd = Predictor().to(device)
    opt = torch.optim.Adam(
        list(en.parameters()) + list(pr.parameters()) + list(pd.parameters()),
        lr=cfg["ssl_lr"],
    )
    q = cfg["pretrain_quantization"]
    xx = (x / q).round() * q if q else x
    gen = torch.Generator().manual_seed(20000 + fold)
    for ep in range(cfg["ssl_epochs"]):
        en.train()
        pr.train()
        pd.train()
        for jj in torch.randperm(len(ia), generator=gen).split(cfg["pretrain_batch"]):
            b = xx[ia[jj]]
            z1 = pr(en(augment_gpu(b)))
            z2 = pr(en(augment_gpu(b)))
            loss = -0.5 * (
                torch.nn.functional.cosine_similarity(pd(z1), z2.detach()).mean()
                + torch.nn.functional.cosine_similarity(pd(z2), z1.detach()).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 25 == 0:
            print(f"fold{fold} scratch SSL {ep + 1}", flush=True)
    return en, dict(projector=cpu_state(pr), predictor=cpu_state(pd))


def supervised(x, y, ia, iv, fold, cfg, device):
    seed_all(59 + fold * 101)
    en = Encoder64().to(device)
    en.eval()
    with torch.no_grad():
        z = extract(en, x[ia])
        mu = z.mean(0)
        sd = z.std(0, unbiased=False).clamp(min=1e-6)
    seed_all(5011 + (fold - 1) * 113)
    head = Head().to(device)
    opt = torch.optim.AdamW(
        list(en.parameters()) + list(head.parameters()),
        lr=cfg["head_lr"],
        weight_decay=0.0001,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["head_epochs"])
    gen = torch.Generator().manual_seed(3000 + fold - 1)
    best = (-1, float("inf"))
    saved = None
    for ep in range(cfg["head_epochs"]):
        en.train()
        head.train()
        for jj in torch.randperm(len(ia), generator=gen).split(32):
            ix = ia[jj]
            loss = torch.nn.functional.cross_entropy(head((en(x[ix]) - mu) / sd), y[ix])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        en.eval()
        head.eval()
        with torch.no_grad():
            v = head((extract(en, x[iv]) - mu) / sd)
            acc = int((v.argmax(1) == y[iv]).sum())
            ce = float(torch.nn.functional.cross_entropy(v, y[iv]))
        if acc > best[0] or (acc == best[0] and ce < best[1]):
            best = (acc, ce)
            saved = (cpu_state(en), cpu_state(head))
            selected = ep + 1
    en.load_state_dict(saved[0])
    head.load_state_dict(saved[1])
    return en, head, mu, sd, dict(val_correct=best[0], val_ce=best[1], epoch=selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "components")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["full", "ae_only", "scratch_simsiam", "supervised_scratch"],
        default=["full", "ae_only", "scratch_simsiam", "supervised_scratch"],
    )
    args = p.parse_args()
    device = args.device
    cfg = json.loads(args.config.read_text())
    hc = json.loads(args.head_config.read_text())
    tc = cfg["training"]
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    raw, y, sessions = load_data(args.data_dir, cfg)
    x = ae_norm(raw).to(device)
    yt = y.to(device)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"folds": {}, "summary": {}}
    kinds = args.variants
    for fold in (1, 2, 3):
        directory = out / f"fold{fold}"
        directory.mkdir(exist_ok=True)
        ia, iv, te = fold_indices(y, sessions, fold)
        reference = (
            train_fold(raw, y, sessions, fold, directory, device, **tc)
            if set(kinds) & {"full", "ae_only"}
            else None
        )
        report["folds"][str(fold)] = {}
        for kind in kinds:
            print(f"fold{fold} {kind}", flush=True)
            extra = {}
            if kind == "full":
                en = Encoder64().to(device)
                en.load_state_dict(reference["encoder"])
                head = Head().to(device)
                head.load_state_dict(reference["classifier"])
                mu = reference["mu"].to(device)
                sd = reference["sd"].to(device)
                selection = dict(
                    epoch=reference["selected_epoch"],
                    val_correct=reference["val_correct"],
                )
            elif kind == "supervised_scratch":
                en, head, mu, sd, selection = supervised(
                    x, yt, ia, iv, fold, tc, device
                )
            else:
                if kind == "ae_only":
                    ae = AE32().to(device)
                    ae.load_state_dict(reference["ae"])
                    seed_all(59 + fold * 101)
                    en = Encoder64(ae, "new").to(device)
                else:
                    en, extra = scratch_ssl(x, ia, fold, tc, device)
                en.eval()
                en.requires_grad_(False)
                with torch.no_grad():
                    z = extract(en, x)
                head, mu, sd, selection = fit_head(z, y, ia, iv, fold, hc, device)
            en.eval()
            head.eval()
            torch.save(
                dict(
                    encoder=cpu_state(en),
                    classifier=cpu_state(head),
                    mu=mu.cpu(),
                    sd=sd.cpu(),
                    train=ia,
                    val=iv,
                    test=te,
                    variant=kind,
                    val_selection=selection,
                    **extra,
                ),
                directory / f"{kind}.pth",
            )
            report["folds"][str(fold)][kind] = {"validation": selection}
    for fold in (1, 2, 3):
        for kind in kinds:
            directory = out / f"fold{fold}"
            ck = torch.load(
                directory / f"{kind}.pth", weights_only=False, map_location="cpu"
            )
            te = ck["test"]
            en = Encoder64().to(device)
            en.load_state_dict(ck["encoder"])
            head = Head().to(device)
            head.load_state_dict(ck["classifier"])
            en.eval()
            head.eval()
            with torch.no_grad():
                v = head(
                    (extract(en, x[te]) - ck["mu"].to(device)) / ck["sd"].to(device)
                )
                pred = v.argmax(1).cpu().numpy()
            row = metric(y[te].numpy(), pred)
            report["folds"][str(fold)][kind]["test"] = row
            save_json(
                directory / f"{kind}_predictions.json",
                dict(true=y[te].tolist(), pred=pred.tolist(), metrics=row),
            )
    for kind in kinds:
        report["summary"][kind] = summarize(
            [report["folds"][str(f)][kind]["test"] for f in (1, 2, 3)]
        )
    save_json(out / "results.json", report)
    save_json(out / "config.json", cfg)
    save_json(
        out / "environment.json",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name()
            if str(device).startswith("cuda")
            else None,
        ),
    )
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
