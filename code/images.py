"""Fungi-image latents for the image track.

REAL: expects an image tree
    data/fungi/train/edible/*.jpg, train/poisonous/*.jpg, ood/edible/*.jpg, ood/poisonous/*.jpg
encodes every image once with I-JEPA, caches the latents.

SYNTHETIC (smoke): plants the lookalike phenomenon directly in latent space so the three-regime
mechanism is testable with no weights/data:
  * dim 0 is the discriminative axis (edible negative, poisonous positive) the classifier learns;
  * a few NOVELTY dims are ~0 in all training data;
  * OOD lookalikes LOOK edible on dim 0 (so a point classifier eats them, and the ensemble
    AGREES = epistemic is blind) but have nonzero novelty dims (off-manifold = observation-surprise
    catches them). This mirrors the tabular odorless-poisonous result in continuous space.
"""
from __future__ import annotations

import glob
import os
from typing import Dict

import numpy as np

from config import Cfg, CFG


def synthetic(cfg: Cfg = CFG, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    D, nv = cfg.syn_latent_dim, cfg.syn_novel_dims

    def block(n, disc_mean, novel_mean):
        x = rng.normal(0, 1, (n, D)).astype(np.float32)
        x[:, 0] = rng.normal(disc_mean, 0.5, n)        # discriminative axis
        x[:, D - nv:] = novel_mean + rng.normal(0, 0.05, (n, nv))  # novelty dims (~0 in train)
        return x

    # in-distribution (training manifold): novelty dims ~ 0
    Xe = block(1500, -2.0, 0.0); ye = np.zeros(1500, np.int64)   # edible
    Xp = block(1500, +2.0, 0.0); yp = np.ones(1500, np.int64)    # poisonous
    Xin = np.vstack([Xe, Xp]); yin = np.concatenate([ye, yp])
    idx = rng.permutation(len(Xin)); nval = len(idx) // 5
    vi, ti = idx[:nval], idx[nval:]

    # OOD lookalikes: novelty dims ~ 3 (off-manifold). Most poisonous LOOK edible (disc ~ -2)
    # = confidently-wrong consensus; a few sit near the boundary (disc ~ 0) so the ensemble
    # catches some -> shows ensemble partially helps but doesn't reach zero.
    Xo_e = block(400, -2.0, 3.0); yo_e = np.zeros(400, np.int64)               # OOD edible
    Xo_p1 = block(100, -2.0, 3.0)                                              # look-edible poison
    Xo_p2 = block(20, 0.0, 3.0)                                                # boundary poison
    Xo_p = np.vstack([Xo_p1, Xo_p2]); yo_p = np.ones(120, np.int64)
    Xood = np.vstack([Xo_e, Xo_p]); yood = np.concatenate([yo_e, yo_p])

    return {"train": (Xin[ti], yin[ti]), "val": (Xin[vi], yin[vi]),
            "ood": (Xood, yood), "D": D}


def _encode_dir(enc, root, label_dirs):
    from PIL import Image
    X, y = [], []
    for lab, sub in label_dirs.items():
        paths = sorted(glob.glob(os.path.join(root, sub, "*")))
        if not paths:
            continue
        imgs = [Image.open(p).convert("RGB") for p in paths]
        X.append(enc.encode(imgs)); y += [lab] * len(imgs)
    return np.vstack(X), np.array(y, np.int64)


def real(cfg: Cfg = CFG) -> Dict:
    from ijepa_iface import IJepaEncoder
    enc = IJepaEncoder(cfg, synthetic=False)
    Xtr, ytr = _encode_dir(enc, os.path.join(cfg.image_dir, "train"),
                           {0: "edible", 1: "poisonous"})
    Xood, yood = _encode_dir(enc, os.path.join(cfg.image_dir, "ood"),
                             {0: "edible", 1: "poisonous"})
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(Xtr)); nval = len(idx) // 5
    vi, ti = idx[:nval], idx[nval:]
    return {"train": (Xtr[ti], ytr[ti]), "val": (Xtr[vi], ytr[vi]),
            "ood": (Xood, yood), "D": Xtr.shape[1]}
