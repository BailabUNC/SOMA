"""Evaluate private checkpoints on held-out cohorts or an external labeled file."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from model import Encoder64, Head, RawProject, ae_norm
from train import extract, fold_indices, load_data, metric, save_json, summarize

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def predict_checkpoint(checkpoint, raw, device):
    encoder = (RawProject() if checkpoint.get('encoder_kind') == 'raw' else Encoder64()).to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()
    head = Head().to(device)
    head.load_state_dict(checkpoint['classifier'])
    head.eval()
    if checkpoint.get('encoder_kind') == 'raw':
        features = encoder(((raw - checkpoint['raw_mean']) / checkpoint['raw_std']).to(device))
    else:
        features = extract(encoder, ae_norm(raw).to(device))
    return head((features - checkpoint['mu'].to(device)) / checkpoint['sd'].to(device)).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'DATA')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'outputs' / 'main')
    parser.add_argument('--checkpoint-pattern', default='fold{fold}.pth')
    parser.add_argument('--external', type=Path, help='Labeled external data, e.g. private C2.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.external:
        data = torch.load(args.external, weights_only=True, map_location='cpu')
        raw = data['features'].float()
        if raw.ndim != 3 or tuple(raw.shape[1:]) != (512, 48) or not torch.isfinite(raw).all():
            raise ValueError('External features must be finite [N,512,48] tensors')
        values = data['labels']
        if tuple(values.shape) != (len(raw),) or not torch.equal(values, values.long()):
            raise ValueError('External labels must be an integer vector with one label per window')
        truth = torch.tensor([config['labels'].index(int(v)) for v in values])
        sessions = None
    else:
        raw, truth, sessions = load_data(args.data_dir, config)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in (1, 2, 3):
        path = args.models_dir / args.checkpoint_pattern.format(fold=fold)
        ck = torch.load(path, weights_only=False, map_location='cpu')
        if sessions is None:
            indices = np.arange(len(truth))
        else:
            expected = fold_indices(truth, sessions, fold)
            for key, values in zip(('train', 'val', 'test'), expected):
                if not np.array_equal(np.asarray(ck[key]), values):
                    raise ValueError(f'{path}: {key} indices differ from configured split')
            indices = expected[2]
        logits = predict_checkpoint(ck, raw[indices], args.device)
        pred = logits.argmax(1).numpy()
        y = truth[indices].numpy()
        row = metric(y, pred)
        rows.append(row)
        save_json(args.out / f'fold{fold}_predictions.json', {
            'true': y.tolist(), 'pred': pred.tolist(), 'logits': logits.tolist(),
            'original_indices': indices.tolist(), 'metrics': row,
            'confusion': confusion_matrix(y, pred, labels=np.arange(24)).tolist(),
        })
    result = {'evaluation': 'external' if args.external else 'heldout',
              'folds': rows, 'summary': summarize(rows)}
    save_json(args.out / 'results.json', result)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
