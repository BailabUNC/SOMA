"""Small CPU checks using generated signals, never the private study dataset."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from components import scratch_ssl, supervised
from evaluate import predict_checkpoint
from model import AE32, Encoder64, Head, ae_norm
from train import cpu_state, fold_indices, load_data, train_fold
from evaluate_reconstruction import sample_metrics


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / 'config.json').read_text())
        cls.config['session_sizes'] = [72, 72, 72]
        cls.training = {**cls.config['training'], 'ae_epochs': 1, 'ssl_epochs': 1, 'head_epochs': 1}
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        generator = torch.Generator().manual_seed(20261101)
        for i in (1, 2, 3):
            torch.save({'features': torch.rand(72, 512, 48, generator=generator),
                        'labels': torch.tensor(cls.config['labels'] * 3),
                        'subject_ids': torch.full((72,), i),
                        'recording_note': 'generated test signal, not study data'}, cls.folder / f'S{i}.pt')
        cls.raw, cls.y, cls.sessions = load_data(cls.folder, cls.config)
        cls.x = ae_norm(cls.raw)
        cls.ia, cls.iv, cls.te = fold_indices(cls.y, cls.sessions, 1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_outer_and_inner_splits_are_disjoint_and_complete(self):
        heldout = []
        for fold in (1, 2, 3):
            ia, iv, te = fold_indices(self.y, self.sessions, fold)
            self.assertFalse(set(ia) & set(iv) or set(ia) & set(te) or set(iv) & set(te))
            self.assertEqual(set(ia) | set(iv) | set(te), set(range(len(self.y))))
            self.assertEqual(set(te), set(self.sessions[fold - 1]))
            self.assertEqual(len(iv), 29)
            heldout.extend(te)
        self.assertEqual(sorted(heldout), list(range(len(self.y))))

    def test_loader_rejects_invalid_identity_and_nonfinite_inputs(self):
        path = self.folder / 'S1.pt'
        original = torch.load(path, weights_only=True)
        try:
            invalid = {**original, 'subject_ids': torch.zeros(72, dtype=torch.long)}
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, 'subject_ids'):
                load_data(self.folder, self.config)
            invalid = {**original, 'features': original['features'].clone()}
            invalid['features'][0, 0, 0] = float('nan')
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                load_data(self.folder, self.config)
        finally:
            torch.save(original, path)

    def test_main_checkpoint_is_frozen_phase2_encoder(self):
        out = self.folder / 'main'
        out.mkdir()
        ck = train_fold(self.raw, self.y, self.sessions, 1, out, 'cpu', **self.training)
        phase2 = torch.load(out / 'fold1_phase2.pth', weights_only=False)
        phase1 = torch.load(out / 'fold1_phase1.pth', weights_only=False)
        for key, value in phase2['encoder'].items():
            self.assertTrue(torch.equal(value, ck['encoder'][key]))
        for block in range(3):
            for suffix in ('weight', 'bias'):
                self.assertTrue(torch.equal(ck['encoder'][f'blocks.{block}.0.{suffix}'],
                                            phase1['ae'][f'enc{block+1}.0.{suffix}']))
        self.assertFalse(np.intersect1d(ck['train'], ck['test']).size)
        logits = predict_checkpoint(ck, self.raw[self.te], 'cpu')
        self.assertEqual(tuple(logits.shape), (72, 24))
        self.assertTrue(torch.isfinite(logits).all())

    def test_supervised_baseline_updates_encoder_and_head_on_cpu(self):
        torch.manual_seed(59 + 101)
        initial = cpu_state(Encoder64())
        encoder, head, mu, sd, selection = supervised(self.x, self.y, self.ia, self.iv, 1, self.training, 'cpu')
        self.assertTrue(any(not torch.equal(initial[key], value) for key, value in encoder.state_dict().items()))
        self.assertEqual(selection['epoch'], 1)
        ck = {'encoder': cpu_state(encoder), 'classifier': cpu_state(head), 'mu': mu, 'sd': sd}
        logits = predict_checkpoint(ck, self.raw[self.te], 'cpu')
        self.assertEqual(tuple(logits.shape), (72, 24))

    def test_scratch_ssl_and_reconstruction_cpu_paths(self):
        encoder, extra = scratch_ssl(self.x, self.ia, 1, self.training, 'cpu')
        self.assertIn('projector', extra)
        self.assertIn('predictor', extra)
        self.assertEqual(tuple(encoder(self.x[:2]).shape), (2, 64))
        model = AE32().eval()
        rows = sample_metrics(model, self.x, self.te[:3], 1, 'heldout', 'cpu')
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(np.isfinite(row[key]) for row in rows for key in ('mse', 'nrmse', 'pearson_r')))


if __name__ == '__main__':
    unittest.main()
