"""Observation-surprise / input-novelty signal -- complementary to ensemble disagreement.

Ensemble std catches predictive uncertainty but is BLIND to confidently-wrong consensus (all
members agree, all wrong). The fix: detect that the INPUT itself is off the training manifold.
For one-hot tabular data, the cleanest signal is: does the input activate any feature VALUE
that had ~zero support in training? An odorless mushroom activates odor='none', which the
strong-odor training set never saw -> the system knows it's looking at something it never
learned, and the safe act is to abstain. (This is the observation-surprise term of free energy.)
"""
from __future__ import annotations

import numpy as np


def train_support(Xtr: np.ndarray) -> np.ndarray:
    """Per-feature-column training frequency (fraction of rows where the value is active)."""
    return Xtr.mean(axis=0)


def novelty_flag(X: np.ndarray, support: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """True where the input uses a feature value with ~zero training support (off-manifold)."""
    active_unsupported = (X > 0) & (support[None, :] <= eps)
    return active_unsupported.any(axis=1)
