"""Refine the matched pretrained AE64 encoder with SimSiam."""

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
from soma.models import Encoder64, Head, Predictor, Projector, ae_norm
from soma.paths import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DEFAULT_HEAD_CONFIG, ROOT
from soma.training.classifier import fit_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--head-config", type=Path, default=DEFAULT_HEAD_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "ae64_simsiam")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--ae-dir", type=Path, default=ROOT / "outputs" / "ae64")
    args = parser.parse_args()
    device = args.device
    cfg = json.loads(args.config.read_text())
    tc = cfg["training"]
    hc = json.loads(args.head_config.read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    raw, y, sessions = load_data(args.data_dir, cfg)
    x = ae_norm(raw).to(device)
    q = tc["pretrain_quantization"]
    xx = (x / q).round() * q if q else x
    out = args.out.resolve()
    out.mkdir(exist_ok=True, parents=True)
    selections = []
    for fold in (1, 2, 3):
        ia, iv, te = fold_indices(y, sessions, fold)
        source = torch.load(
            args.ae_dir / f"fold{fold}.pth", weights_only=False, map_location="cpu"
        )
        for key, value in [("train", ia), ("val", iv), ("test", te)]:
            assert (source[key] == value).all()
        seed_all(59 + fold * 101)
        en = Encoder64().to(device)
        for j in range(5):
            en.blocks[j].load_state_dict(
                {
                    k.removeprefix(f"enc{j + 1}."): v
                    for k, v in source["ae"].items()
                    if k.startswith(f"enc{j + 1}.")
                }
            )
        initial = cpu_state(en)
        for j, b in enumerate(en.blocks):
            for p in b.parameters():
                p.requires_grad = j >= 3
        pr = Projector().to(device)
        pd = Predictor().to(device)
        opt = torch.optim.Adam(
            [p for p in en.parameters() if p.requires_grad]
            + list(pr.parameters())
            + list(pd.parameters()),
            lr=tc["ssl_lr"],
        )
        gen = torch.Generator().manual_seed(20000 + fold)
        for ep in range(tc["ssl_epochs"]):
            en.train()
            pr.train()
            pd.train()
            for jj in torch.randperm(len(ia), generator=gen).split(
                tc["pretrain_batch"]
            ):
                b = xx[ia[jj]]
                z1 = pr(en(augment_gpu(b)))
                z2 = pr(en(augment_gpu(b)))
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
                print(
                    f"fold{fold} AE64 SimSiam {ep + 1}: {loss.item():.6f}", flush=True
                )
        en.eval()
        en.requires_grad_(False)
        with torch.no_grad():
            z = extract(en, x)
        head, mu, sd, selection = fit_head(z, y, ia, iv, fold, hc, device)
        selections.append(selection)
        assert all(torch.equal(initial[k], source["encoder"][k]) for k in initial)
        torch.save(
            dict(
                encoder=cpu_state(en),
                classifier=cpu_state(head),
                mu=mu.cpu(),
                sd=sd.cpu(),
                projector=cpu_state(pr),
                predictor=cpu_state(pd),
                initial_encoder=initial,
                train=ia,
                val=iv,
                test=te,
                val_selection=selection,
            ),
            out / f"fold{fold}.pth",
        )
    rows = []
    for fold in (1, 2, 3):
        ck = torch.load(out / f"fold{fold}.pth", weights_only=False, map_location="cpu")
        en = Encoder64().to(device)
        en.load_state_dict(ck["encoder"])
        en.eval()
        head = Head().to(device)
        head.load_state_dict(ck["classifier"])
        head.eval()
        te = ck["test"]
        with torch.no_grad():
            pred = (
                head((extract(en, x[te]) - ck["mu"].to(device)) / ck["sd"].to(device))
                .argmax(1)
                .cpu()
                .numpy()
            )
        row = metric(y[te].numpy(), pred)
        rows.append(row)
        save_json(
            out / f"fold{fold}_predictions.json",
            dict(true=y[te].tolist(), pred=pred.tolist(), metrics=row),
        )
    result = dict(
        folds=rows,
        summary=summarize(rows),
        validation=selections,
        ae64_initialization_identical=True,
    )
    save_json(out / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
