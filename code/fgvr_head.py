"""FGVR metric-learning head on FROZEN I-JEPA latents (the frozen-backbone lever).

Trains a triplet/contrastive projection that pulls same-species latents together and pushes the
toxic lookalike away -- on the FROZEN latents, no encoder fine-tuning. Honest question: does
metric learning on the same latents reduce the lethal false-eat (the confidently-wrong residual),
or is the ceiling in the ENCODER (so a head can't help)?

Protocol (no leakage): triplets trained on the TRAIN split only; the eat/discard/inspect decision
is evaluated on HELD-OUT Destroying Angels, raw-latent pipeline vs triplet-projected pipeline.
"""
from __future__ import annotations

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import fit, distance
from gate import auroc
import efe

CFG.ensemble_k, CFG.steps = 6, 250
SPMAP = {"edible_chanterelle": 0, "poison_jackolantern": 1,
         "edible_fieldmushroom": 0, "poison_destroyingangel": 1}
PAIRS = [("edible_chanterelle", "poison_jackolantern"),
         ("edible_fieldmushroom", "poison_destroyingangel")]
L = {s: np.load(f"data/fungi_real/{s}.npy") for s in SPMAP}

rng = np.random.default_rng(0)
tr, te = {}, {}
for s in SPMAP:
    i = rng.permutation(len(L[s])); c = int(0.7 * len(L[s]))
    tr[s], te[s] = L[s][i[:c]], L[s][i[c:]]


class Proj(nn.Module):
    def __init__(self, din=1280, d=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, d))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


def train_triplet(steps=900, margin=0.3):
    g = Proj(); opt = torch.optim.Adam(g.parameters(), 1e-3)
    T = {s: torch.tensor(tr[s]) for s in SPMAP}
    for _ in range(steps):
        loss = 0.0
        for e, p in PAIRS:
            for anc, neg in [(e, p), (p, e)]:
                A, N = T[anc], T[neg]
                ia, ip, inn = (torch.randint(0, len(A), (128,)), torch.randint(0, len(A), (128,)),
                               torch.randint(0, len(N), (128,)))
                za, zp, zn = g(A[ia]), g(A[ip]), g(N[inn])
                da = ((za - zp) ** 2).sum(-1); dn = ((za - zn) ** 2).sum(-1)
                loss = loss + torch.clamp(da - dn + margin, min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    g.eval()
    return lambda X: g(torch.tensor(X)).detach().numpy()


def pca(Xfit, k):
    mu = Xfit.mean(0); _, _, Vt = np.linalg.svd(Xfit - mu, full_matrices=False)
    return (lambda X: (X - mu) @ Vt[:k].T)


def sep(Xe, Xp):
    ec, pc = Xe.mean(0), Xp.mean(0)
    sc = lambda Z: np.linalg.norm(Z - ec, axis=1) - np.linalg.norm(Z - pc, axis=1)
    return auroc(np.r_[sc(Xe), sc(Xp)], np.r_[np.zeros(len(Xe)), np.ones(len(Xp))])


def false_eat(feat, d):
    Xtr = np.vstack([feat(tr[s]) for s in SPMAP])
    ytr = np.concatenate([np.full(len(tr[s]), SPMAP[s]) for s in SPMAP]).astype("f4")
    ens = train_ensemble(Xtr, ytr, d, CFG)
    Zda = feat(te["poison_destroyingangel"])
    r, s = ensemble_risk(ens, Zda)
    muE, invE = fit(Xtr[ytr == 0])
    nov = distance(Zda, muE, invE) > np.percentile(distance(Xtr[ytr == 0], muE, invE), 99)
    d_ = efe.decide(r, s, CFG, kappa=3.0, novel=nov)
    return int((d_ == efe.EAT).sum()), len(Zda)


def main():
    raw = pca(np.vstack([tr[s] for s in SPMAP]), 64)        # raw frozen latents (PCA-64)
    print("training FGVR triplet head on frozen latents ...")
    proj = train_triplet()
    fmte, date = te["edible_fieldmushroom"], te["poison_destroyingangel"]
    print(f"\nlethal-pair separability (held-out):  raw {sep(raw(fmte), raw(date)):.2f}  ->  "
          f"FGVR-triplet {sep(proj(fmte), proj(date)):.2f}")
    er, n = false_eat(raw, 64)
    ep, _ = false_eat(proj, 128)
    print(f"held-out Destroying Angels EATEN @kappa=3:  raw {er}/{n}  ->  FGVR-triplet {ep}/{n}")
    verdict = ("FGVR head REDUCES the lethal residual on frozen latents (the lever works)"
               if ep < er else
               "FGVR head does NOT reduce it -> the ceiling is in the ENCODER, not fixable by a head "
               "(need encoder fine-tune / higher-res / crop-to-volva, or an independent channel)")
    print("verdict:", verdict)
    print("\nSAFETY: research vehicle only; even if reduced, false-eat>0 on a lethal call is unacceptable.")


if __name__ == "__main__":
    main()
