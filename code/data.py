"""UCI Mushroom loader + the lookalike (odor='none') OOD split.

One-hot encodes the 22 categorical features over a FIXED vocabulary built on the full set,
so the odor='none' column exists but is never active in training -> at OOD time it activates
an untrained input, which (by design) drives ensemble disagreement = the epistemic signal.
"""
from __future__ import annotations

import csv
from typing import List, Tuple

import numpy as np

from config import Cfg, CFG


def _load_rows(cfg: Cfg) -> Tuple[List[List[str]], np.ndarray]:
    rows = [r for r in csv.reader(open(cfg.csv)) if len(r) == 23]
    y = np.array([1 if r[0] == "p" else 0 for r in rows], dtype=np.int64)  # 1 = poisonous
    feats = [r[1:] for r in rows]                                          # 22 categorical features
    return feats, y


def load(cfg: Cfg = CFG):
    feats, y = _load_rows(cfg)
    cols = list(zip(*feats))                                # 22 columns of values
    vocabs = [sorted(set(c)) for c in cols]                 # fixed vocab per feature
    offsets, D = [], 0
    for v in vocabs:
        offsets.append(D)
        D += len(v)
    X = np.zeros((len(feats), D), dtype=np.float32)
    val2idx = [{val: k for k, val in enumerate(v)} for v in vocabs]
    for i, r in enumerate(feats):
        for j, val in enumerate(r):
            X[i, offsets[j] + val2idx[j][val]] = 1.0
    odor = np.array([r[cfg.odor_feature_index] for r in feats])
    return X, y, odor, D


def splits(cfg: Cfg = CFG):
    """train/val on strong-odor mushrooms; OOD = odor='none' (the lookalike trap)."""
    X, y, odor, D = load(cfg)
    ood = odor == cfg.ood_odor_value
    Xin, yin = X[~ood], y[~ood]
    Xood, yood = X[ood], y[ood]
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(Xin))
    nval = len(idx) // 5
    vi, ti = idx[:nval], idx[nval:]
    return {
        "train": (Xin[ti], yin[ti]),
        "val": (Xin[vi], yin[vi]),
        "ood": (Xood, yood),          # ~3528: mostly edible + ~120 poisonous "smell-fine" lookalikes
        "D": D,
    }
