"""Continuous-latent observation-surprise: Mahalanobis distance to the training-latent
manifold. The image analog of the tabular 'unseen feature value' flag.

Ensemble disagreement is blind to confidently-wrong consensus; this catches inputs that sit
OFF the manifold the encoder mapped the training data onto (a lookalike whose I-JEPA latent
lands somewhere training never did). High distance -> the system is looking at something it
can't place -> abstain.
"""
from __future__ import annotations

import numpy as np


def fit(Xtr: np.ndarray, ridge: float = 1e-3):
    mu = Xtr.mean(axis=0)
    cov = np.cov(Xtr.T) + ridge * np.eye(Xtr.shape[1])
    inv = np.linalg.inv(cov)
    return mu, inv


def distance(X: np.ndarray, mu: np.ndarray, inv: np.ndarray) -> np.ndarray:
    d = X - mu
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, inv, d), 0.0))


def novelty_flag(X: np.ndarray, Xtr: np.ndarray, percentile: float = 99.0):
    """Returns (flag [N] bool, dist [N]). flag = Mahalanobis distance above the given
    percentile of the TRAINING distances (so ~1-(p/100) of in-dist data is flagged)."""
    mu, inv = fit(Xtr)
    thresh = np.percentile(distance(Xtr, mu, inv), percentile)
    dist = distance(X, mu, inv)
    return dist > thresh, dist


# --- geometry-bucket OOD (the keeper from the divergence-floor analysis) ---
# Mahalanobis assumes ONE Gaussian (first two moments) -> blind to multi-modal manifolds:
# a point in the GAP between two training modes has small Mahalanobis but is genuinely OOD.
# kNN-distance to the training set is the nonparametric / Wasserstein-family geometry signal
# that catches it. It SHARPENS the separable regime; it CANNOT beat the coincident floor
# (a lookalike sitting on the training manifold has small kNN distance too -- by theorem).
def knn_novelty(X: np.ndarray, Xtr: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest TRAINING points (geometry OOD; multi-modal aware)."""
    d = np.sqrt(np.maximum(
        ((X[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1), 0.0))   # [N, M]
    k = min(k, d.shape[1])
    return np.sort(d, axis=1)[:, :k].mean(axis=1)
