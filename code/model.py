"""Risk net + ensemble. Each member predicts P(poisonous); the ensemble's MEAN is the
pragmatic risk and its STD across members is the epistemic uncertainty (member
disagreement = where the model has no support, e.g. the odorless lookalikes)."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Cfg, CFG


class RiskNet(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)               # logit


def train_member(Xtr: np.ndarray, ytr: np.ndarray, d_in: int, seed: int, cfg: Cfg = CFG) -> RiskNet:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    if cfg.bootstrap:
        bidx = torch.randint(0, n, (n,), generator=g)
        Xtr, ytr = Xtr[bidx.numpy()], ytr[bidx.numpy()]
    X = torch.tensor(Xtr); y = torch.tensor(ytr, dtype=torch.float32)
    net = RiskNet(d_in)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):
        bi = torch.randint(0, n, (cfg.batch,), generator=g)
        loss = F.binary_cross_entropy_with_logits(net(X[bi]), y[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def train_ensemble(Xtr, ytr, d_in, cfg: Cfg = CFG) -> List[RiskNet]:
    return [train_member(Xtr, ytr, d_in, seed=s, cfg=cfg) for s in range(cfg.ensemble_k)]


@torch.no_grad()
def ensemble_risk(ensemble: List[RiskNet], X: np.ndarray):
    """Returns (p_mean [N], p_std [N]) over the ensemble. p = P(poisonous)."""
    Xt = torch.tensor(X)
    P = torch.stack([torch.sigmoid(net(Xt)) for net in ensemble])   # [K, N]
    return P.mean(0).numpy(), P.std(0).numpy()
