"""Phase I denoising metrics with the collaborator's original noise seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import AE32, ae_norm
from train import fold_indices, load_data, save_json

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def sample_metrics(model, clean, indices, fold, split, device):
    target = clean[indices]
    seeds = {'train': 80000 + fold, 'validation': 70000 + fold, 'heldout': 90000 + fold}
    generator = torch.Generator().manual_seed(seeds[split])
    noisy = target + 0.1 * torch.randn(target.shape, generator=generator, dtype=target.dtype)
    # Preserve original full-split inference rather than changing numerical batch order.
    prediction = model(noisy.to(device)).cpu()
    mse = (prediction - target).square().mean((1, 2))
    dynamic_range = target.flatten(1).amax(1) - target.flatten(1).amin(1)
    nrmse = mse.sqrt() / dynamic_range.clamp_min(1e-12)
    a, b = prediction.flatten(1), target.flatten(1)
    ac, bc = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    correlation = (ac * bc).sum(1) / (ac.square().sum(1) * bc.square().sum(1)).sqrt().clamp_min(1e-12)
    return [{'sample_index': int(ix), 'split': split, 'mse': float(mse[j]),
             'nrmse': float(nrmse[j]), 'pearson_r': float(correlation[j])}
            for j, ix in enumerate(indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'DATA')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'outputs' / 'main')
    parser.add_argument('--checkpoint-pattern', default='fold{fold}_phase1.pth')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'reconstruction')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    raw, labels, sessions = load_data(args.data_dir, cfg)
    x = ae_norm(raw)
    q = cfg['training']['pretrain_quantization']
    x = (x / q).round() * q if q else x
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold in (1, 2, 3):
        ck = torch.load(args.models_dir / args.checkpoint_pattern.format(fold=fold), weights_only=False, map_location='cpu')
        model = AE32().to(args.device)
        model.load_state_dict(ck['ae'])
        model.eval()
        indices = fold_indices(labels, sessions, fold)
        for key, expected in zip(('train', 'val', 'test'), indices):
            if key in ck and not np.array_equal(np.asarray(ck[key]), expected):
                raise ValueError(f'Fold {fold}: checkpoint {key} indices differ from configured split')
        samples = []
        summary = []
        for split, ix in zip(('train', 'validation', 'heldout'), indices):
            rows = sample_metrics(model, x, ix, fold, split, args.device)
            samples.extend(rows)
            entry = {'fold': fold, 'split': split, 'n_samples': len(rows)}
            for metric in ('mse', 'nrmse', 'pearson_r'):
                values = np.array([row[metric] for row in rows])
                entry[metric + '_mean'] = float(values.mean())
                entry[metric + '_sd'] = float(values.std(ddof=0))
            summary.append(entry)
        summaries.extend(summary)
        save_json(args.out / f'fold{fold}_metrics.json', {'samples': samples, 'summary': summary})
    save_json(args.out / 'all_folds_summary.json', summaries)
    print(json.dumps([row for row in summaries if row['split'] == 'heldout'], indent=2))


if __name__ == '__main__':
    main()
