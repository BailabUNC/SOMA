"""Model definitions shared by the second-revision workflows."""

import random

import torch
from torch import nn


class AE32(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(48, 128, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(128, 64, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(64, 32, 3, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(32, 16, 3, 2, 1), nn.ReLU())
        self.enc5 = nn.Sequential(nn.Conv1d(16, 32, 3, 2, 1), nn.ReLU())
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 48, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.enc5(e4)
        z = self.dec5(z)
        z = self.dec4(torch.cat([z, e4], dim=1))
        z = self.dec3(torch.cat([z, e3], dim=1))
        z = self.dec2(torch.cat([z, e2], dim=1))
        return self.dec1(torch.cat([z, e1], dim=1))


class Encoder64(nn.Module):
    def __init__(self, ae=None, fifth_layer_init="new"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(48, 128, 3, 2, 1), nn.ReLU()),
                nn.Sequential(nn.Conv1d(128, 64, 3, 2, 1), nn.ReLU()),
                nn.Sequential(nn.Conv1d(64, 32, 3, 2, 1), nn.ReLU()),
                nn.Sequential(nn.Conv1d(32, 16, 3, 2, 1), nn.ReLU()),
                nn.Sequential(nn.Conv1d(16, 64, 3, 2, 1), nn.ReLU()),
            ]
        )
        if ae is not None:
            with torch.no_grad():
                for j in range(4):
                    self.blocks[j].load_state_dict(
                        getattr(ae, f"enc{j + 1}").state_dict()
                    )
                if fifth_layer_init == "partial_copy":
                    self.blocks[4][0].weight[:32].copy_(ae.enc5[0].weight)
                    self.blocks[4][0].bias[:32].copy_(ae.enc5[0].bias)
                elif fifth_layer_init != "new":
                    raise ValueError(fifth_layer_init)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x.mean(dim=-1)


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64, affine=True),
        )

    def forward(self, z):
        return self.net(z)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, z):
        return self.net(z)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 24),
        )
        assert sum((p.numel() for p in self.parameters())) == 22936

    def forward(self, x):
        return self.net(x)


def ae_norm(x):
    xc = x.permute(0, 2, 1).contiguous()
    lo = xc.amin(dim=-1, keepdim=True)
    hi = xc.amax(dim=-1, keepdim=True)
    return ((xc - lo) / (hi - lo + 1e-08)).float()


class RawProject(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(48 * 512, 64, bias=False)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="linear")
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x.flatten(1))


def augment(x, noise=0.1, channel_drop=0.04, scale=0.08):
    x = x + noise * torch.randn_like(x)
    x = x * (1 + scale * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))
    keep = (
        torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > channel_drop
    ).float()
    return x * keep


def augment_table_s4(x):
    import torch.nn.functional as F

    n, c, L = x.shape
    result = []
    for a in x:
        cut = random.randint(int(0.8 * L), L)
        lo = random.randint(0, L - cut)
        a = F.interpolate(
            a[:, lo : lo + cut].unsqueeze(0), size=L, mode="linear", align_corners=False
        )[0]
        if random.random() < 0.5:
            a = a.flip(-1)
        a = torch.roll(a, random.randint(-L // 10, L // 10), dims=-1)
        a = a + 0.01 * torch.randn_like(a)
        if random.random() < 0.2:
            drop = torch.randperm(c, device=a.device)[: max(1, c // 10)]
            a = a.clone()
            a[drop] = 0
        width = random.randint(0, L // 5)
        if width:
            a = a.clone()
            i = random.randint(0, L - width)
            a[:, i : i + width] = 0
        result.append(a.clamp(0, 1))
    return torch.stack(result, dim=0)


def load_frozen_encoder(path, device="cpu"):
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    encoder = Encoder64()
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(device), checkpoint["mu"].to(device), checkpoint["sd"].to(device)
