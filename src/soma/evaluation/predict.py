import argparse
import json
from pathlib import Path

import torch

from soma.models import Encoder64, Head, RawProject, ae_norm
from soma.paths import DEFAULT_CONFIG, DEFAULT_DATA_DIR, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "outputs" / "main")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    torch.set_num_threads(6)
    ck = torch.load(
        args.checkpoint or args.models_dir / f"fold{args.fold}.pth",
        weights_only=False,
        map_location="cpu",
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.backends.cudnn.allow_tf32 = not ck.get("config", config["training"]).get(
        "full_precision", False
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = (RawProject() if ck.get("encoder_kind") == "raw" else Encoder64()).to(
        args.device
    )
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()
    encoder.requires_grad_(False)
    head = Head().to(args.device)
    head.load_state_dict(ck["classifier"])
    head.eval()
    path = args.data or args.data_dir / f"S{args.fold}.pt"
    data = torch.load(path, weights_only=True, map_location="cpu")
    x = data["features"].float()
    with torch.no_grad():
        if ck.get("encoder_kind") == "raw":
            z = encoder(((x - ck["raw_mean"]) / ck["raw_std"]).to(args.device))
        else:
            z = torch.cat([encoder(b) for b in ae_norm(x).to(args.device).split(64)])
        logits = head((z - ck["mu"].to(args.device)) / ck["sd"].to(args.device))
    labels = json.loads(args.config.read_text(encoding="utf-8"))["labels"]
    indices = logits.argmax(1).cpu().tolist()
    result = {
        "predicted_labels": [labels[i] for i in indices],
        "logits": logits.cpu().tolist(),
    }
    out = args.out or ROOT / "outputs" / f"predictions_fold{args.fold}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
