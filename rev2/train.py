import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from model import AE32, Encoder64, Head, Predictor, Projector, ae_norm
ROOT = Path(__file__).resolve().parent

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def fold_indices(y, sessions, fold):
    te = np.asarray(sessions[fold - 1])
    tr = np.concatenate([s for i, s in enumerate(sessions) if i != fold - 1])
    ia, iv = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42, stratify=y[tr].numpy())
    return (tr[ia], tr[iv], te)

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def metric(y, p):
    return dict(correct=int((y == p).sum()), n=len(y), accuracy=float(accuracy_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p)), macro_f1=float(f1_score(y, p, labels=np.arange(24), average='macro', zero_division=0)))

def summarize(rows):
    r = {'correct': [d['correct'] for d in rows]}
    for k in ('accuracy', 'macro_f1', 'balanced_accuracy'):
        a = np.array([d[k] for d in rows])
        r[k] = {'mean': float(a.mean()), 'sd_population': float(a.std())}
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
    result = x.gather(2, lo[:, None].expand(-1, c, -1)) * (1 - weight) + x.gather(2, hi[:, None].expand(-1, c, -1)) * weight
    result = result + 0.01 * torch.randn_like(result)
    drop = torch.rand(n, c, device=dev).argsort(1)[:, :max(1, c // 10)]
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

def train_fold(raw, y, sessions, fold, out, device, ae_epochs=60, ssl_epochs=100, head_epochs=220, ae_lr=0.001, ssl_lr=0.001, head_lr=0.001, pretrain_batch=32, pretrain_quantization=0.0, full_precision=False):
    path = out / f'fold{fold}.pth'
    ia, iv, te = fold_indices(y, sessions, fold)
    seed_all(59 + fold * 101)
    torch.backends.cudnn.allow_tf32 = not full_precision
    torch.backends.cuda.matmul.allow_tf32 = False
    x = ae_norm(raw).to(device)
    yt = y.to(device)
    pretrain_x = (x / pretrain_quantization).round() * pretrain_quantization if pretrain_quantization > 0 else x
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
            print(f'Fold{fold} AE {ep + 1}: {loss.item():.6f}', flush=True)
    torch.save(dict(ae=cpu_state(ae), train=ia, val=iv, test=te), out / f'fold{fold}_phase1.pth')
    seed_all(59 + fold * 101)
    en = Encoder64(ae, 'new').to(device)
    for j, b in enumerate(en.blocks):
        for p in b.parameters():
            p.requires_grad = j >= 3
    pr = Projector().to(device)
    pd = Predictor().to(device)
    opt = torch.optim.Adam([p for p in en.parameters() if p.requires_grad] + list(pr.parameters()) + list(pd.parameters()), lr=ssl_lr)
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
            loss = -0.5 * (torch.nn.functional.cosine_similarity(p1, z2.detach()).mean() + torch.nn.functional.cosine_similarity(p2, z1.detach()).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 25 == 0:
            print(f'Fold{fold} SSL {ep + 1}: {loss.item():.6f}', flush=True)
    torch.save(dict(encoder=cpu_state(en), projector=cpu_state(pr), predictor=cpu_state(pd), train=ia, val=iv, test=te), out / f'fold{fold}_phase2.pth')
    en.eval()
    for p in en.parameters():
        p.requires_grad = False
    z = extract(en, x)
    mu = z[ia].mean(0)
    sd = z[ia].std(0, unbiased=False).clamp(min=1e-06)
    torch.save(dict(encoder=cpu_state(en), mu=mu.cpu(), sd=sd.cpu(), train=ia, val=iv, test=te, frozen=True), out / f'fold{fold}_frozen_encoder.pth')
    z = (z - mu) / sd
    seed_all(5011 + (fold - 1) * 113)
    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=head_lr, weight_decay=0.0001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=head_epochs)
    gen = torch.Generator().manual_seed(3000 + fold - 1)
    best = (-1, float('inf'))
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
    ck = dict(encoder=cpu_state(en), classifier=state, ae=cpu_state(ae), projector=cpu_state(pr), predictor=cpu_state(pd), mu=mu.cpu(), sd=sd.cpu(), train=ia, val=iv, test=te, selected_epoch=selected, val_correct=best[0], config=dict(ae_epochs=ae_epochs, ssl_epochs=ssl_epochs, head_epochs=head_epochs, encoder_frozen_phase3=True, ae_noise=0.1, ssl_last_blocks=2, ae_lr=ae_lr, ssl_lr=ssl_lr, head_lr=head_lr, pretrain_batch=pretrain_batch, pretrain_quantization=pretrain_quantization, full_precision=full_precision, head_batch=32, weight_decay=0.0001, seed=59 + fold * 101))
    torch.save(ck, path)
    print(f'Fold{fold} saved; selected epoch {selected}', flush=True)
    return ck

def evaluate(raw, y, sessions, folder, device):
    rows = []
    all_pred = np.zeros(len(y), dtype=int)
    for fold in (1, 2, 3):
        ck = torch.load(folder / f'fold{fold}.pth', weights_only=False, map_location='cpu')
        torch.backends.cudnn.allow_tf32 = not ck['config'].get('full_precision', False)
        torch.backends.cuda.matmul.allow_tf32 = False
        en = Encoder64().to(device)
        en.load_state_dict(ck['encoder'])
        en.eval()
        h = Head().to(device)
        h.load_state_dict(ck['classifier'])
        h.eval()
        te = ck['test']
        with torch.no_grad():
            z = extract(en, ae_norm(raw[te]).to(device))
            logits = h((z - ck['mu'].to(device)) / ck['sd'].to(device))
            p = logits.argmax(1).cpu().numpy()
        yy = y[te].numpy()
        r = metric(yy, p)
        rows.append(r)
        all_pred[te] = p
        save_json(folder / f'fold{fold}_predictions.json', dict(true=yy.tolist(), pred=p.tolist(), logits=logits.cpu().tolist(), original_indices=te.tolist(), confusion=confusion_matrix(yy, p, labels=np.arange(24)).tolist(), metrics=r))
    result = dict(folds=rows, summary=summarize(rows))
    save_json(folder / 'results.json', result)
    print(json.dumps(result['summary']), flush=True)
    return (result, all_pred)

def load_data(folder, config):
    """Load collaborator-defined cohorts; preserve the checkpoint index layout.

    The permutation restores global indices and does not form new cohorts.
    Additional metadata is accepted but never used as model input.
    """
    folder = Path(folder)
    sizes = config['session_sizes']
    if len(sizes) != 3 or any(n <= 0 for n in sizes):
        raise ValueError('session_sizes must contain three positive cohort sizes')
    sessions = np.split(np.random.default_rng(config['session_seed']).permutation(sum(sizes)), np.cumsum(sizes)[:-1])
    x = torch.empty(sum(sizes), 512, 48)
    labels = torch.empty(sum(sizes), dtype=torch.long)
    required = {'features', 'labels', 'subject_ids'}
    for i, (ix, n) in enumerate(zip(sessions, sizes), 1):
        path = folder / f'S{i}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'{path}: supply private S1.pt, S2.pt and S3.pt via --data-dir')
        d = torch.load(path, weights_only=True, map_location='cpu')
        if not isinstance(d, dict) or not required.issubset(d):
            raise ValueError(f'{path}: required keys are {sorted(required)}')
        if d['features'].shape != (n, 512, 48) or d['labels'].shape != (n,) or d['subject_ids'].shape != (n,):
            raise ValueError(f'{path}: expected features [{n},512,48] and labels/subject_ids [{n}]')
        if not d['subject_ids'].eq(i).all():
            raise ValueError(f'{path}: expected subject_ids={i}; IDs must reflect the supplied provenance')
        if not torch.isfinite(d['features']).all() or not torch.isfinite(d['labels']).all():
            raise ValueError(f'{path}: non-finite features or labels')
        if not torch.equal(d['labels'], d['labels'].long()):
            raise ValueError(f'{path}: labels must be integers')
        x[ix] = d['features']
        labels[ix] = d['labels']
    if sorted(labels.unique().tolist()) != config['labels']:
        raise ValueError('Observed labels do not match config labels')
    y = torch.tensor([config['labels'].index(int(v)) for v in labels])
    return x, y, sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'main')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'DATA')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    torch.set_num_threads(config['cpu_threads'])
    x, y, sessions = load_data(args.data_dir, config)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'config.json', config)
    for f in (1, 2, 3):
        train_fold(x, y, sessions, f, out, args.device, **config['training'])
    evaluate(x, y, sessions, out, args.device)
if __name__ == '__main__':
    main()
