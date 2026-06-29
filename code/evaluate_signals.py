"""(a) corrected, NON-CIRCULAR directed kill-test  +  (b) the keeper (geometry OOD), both
judged by the standing symmetrization-null gate (gate.py).

Non-circularity fixes vs the earlier quick test:
  * the outcome y = false-eat is the ENSEMBLE classifier's decision (not the candidate signal);
  * the baseline ALREADY INCLUDES the classifier risk + symmetric overlap, so a candidate must
    beat the model's own signal, not a strawman;
  * the directed candidate is computed from latent geometry, independent of y;
  * everything is measured separately on the COINCIDENT (killer) regime.
"""
from __future__ import annotations

import numpy as np

from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import fit, distance, knn_novelty
from gate import gate, auroc

rng = np.random.default_rng(0)
D = 16


# ---------- (b) KEEPER: geometry OOD vs Mahalanobis on a MULTI-MODAL manifold ----------
def keeper_demo():
    n = 1000
    mode = rng.integers(0, 2, n) * 6 - 3                 # edible is BIMODAL on dim0: {-3, +3}
    Xtr = rng.normal(0, 1, (n, D)).astype("f4"); Xtr[:, 0] = mode + rng.normal(0, 0.4, n)
    def pts(n, d0):
        x = rng.normal(0, 1, (n, D)).astype("f4"); x[:, 0] = d0 + rng.normal(0, 0.4, n); return x
    indist = pts(300, +3.0)                               # on a training mode (in-distribution)
    gap_ood = pts(300, 0.0)                               # in the GAP between modes -> genuinely OOD
    coincident = pts(300, +3.0)                           # a lookalike sitting ON the manifold (floor)

    mu, inv = fit(Xtr)
    maha = lambda X: distance(X, mu, inv)
    knn = lambda X: knn_novelty(X, Xtr, k=10)

    def au(a, b, scorer):                                 # AUROC: detect b(=OOD,1) vs a(=in-dist,0)
        s = np.r_[scorer(a), scorer(b)]; y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        return auroc(s, y)
    print("(b) KEEPER -- geometry OOD vs Mahalanobis on a bimodal manifold (AUROC, higher=better):")
    print(f"      detect GAP-OOD (between modes): Mahalanobis {au(indist, gap_ood, maha):.2f}  "
          f"kNN-geometry {au(indist, gap_ood, knn):.2f}   <- kNN sees the gap a single Gaussian can't")
    print(f"      detect COINCIDENT lookalike  : Mahalanobis {au(indist, coincident, maha):.2f}  "
          f"kNN-geometry {au(indist, coincident, knn):.2f}   <- both ~0.5: the on-manifold floor, by theorem")


# ---------- (a) corrected, non-circular DIRECTED kill-test, via the gate ----------
def directed_contamination(X, Xtr, ytr, k=15):
    muE, muP = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    u = muE - muP; u = u / (np.linalg.norm(u) + 1e-9)    # poison->edible danger axis
    dd = np.sqrt(np.maximum(((X[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1), 0.0))
    idx = np.argsort(dd, axis=1)[:, :k]
    dc = np.zeros(len(X)); sym = np.zeros(len(X))
    for i in range(len(X)):
        nb = idx[i]; proj = (X[i] - Xtr[nb]) @ u; lab = ytr[nb]
        p, e = lab == 1, lab == 0
        dc[i] = np.maximum(proj[p], 0).sum() - np.maximum(-proj[e], 0).sum()   # net DIRECTED flux
        sym[i] = np.abs(proj[p]).sum() - np.abs(proj[e]).sum()                 # direction-blind
    return dc, sym


def directed_killtest():
    n = 1000
    Xe = rng.normal(0, 1, (n, D)).astype("f4")
    Xp = rng.normal(0, 1, (n, D)).astype("f4"); Xp[:, 0] = 2.5 + rng.normal(0, 0.5, n)
    Xtr = np.vstack([Xe, Xp]); ytr = np.r_[np.zeros(n), np.ones(n)].astype("f4")
    ens = train_ensemble(Xtr, ytr, D, CFG)

    # poison TEST points spanning coincident (dim0~0, look edible) -> separable (dim0~2.5)
    m = 500
    Xt = rng.normal(0, 1, (m, D)).astype("f4"); Xt[:, 0] = rng.uniform(-0.3, 2.5, m)
    risk, _ = ensemble_risk(ens, Xt)                      # classifier's own signal
    y_falseeat = (risk < 0.5).astype(float)              # poison predicted edible = would be EATEN
    mu, inv = fit(Xe)
    maha = distance(Xt, mu, inv)
    dc, sym = directed_contamination(Xt, Xtr, ytr)
    coincident = Xt[:, 0] < 0.4                           # the killer regime

    print(f"\n(a) corrected directed kill-test: {len(Xt)} poison test points, "
          f"{int(y_falseeat.sum())} would be false-eaten; {int(coincident.sum())} coincident.")
    print(f"    mean|directed flux| overall={np.abs(dc).mean():.3f}  on coincident={np.abs(dc[coincident]).mean():.3f}")
    v = gate("directed-contamination", dc,
             baseline_cols=[risk, maha, sym],            # baseline INCLUDES classifier risk + symmetric flux
             y=y_falseeat, sym_candidate=sym, coincident_mask=coincident)
    print("    gate verdict:", v)


if __name__ == "__main__":
    keeper_demo()
    directed_killtest()
    print("\nHONEST READ: kNN-geometry is a real keeper for the SEPARABLE/non-Gaussian regime and 0.5 on the")
    print("coincident floor; the directed-contamination candidate is judged by the same standing gate.")
