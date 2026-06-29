"""Would a GENERATIVE / diffusion-style model help the open-set problem (vs JEPA discriminative)?

Tests generative likelihood OOD detection (Gaussian + GMM = a fair stand-in for a diffusion density
model) against discriminative geometry (kNN) on the held-out death cap. The known pathology
(Nalisnick 2019, 'Do Deep Generative Models Know What They Don't Know?'): generative models can
assign HIGHER likelihood to OOD. If generative AUROC <= chance and OOD log-likelihood is not lower
than in-distribution, a diffusion density detector would be WORSE here, not better.
"""
from __future__ import annotations
import numpy as np
np.seterr(all="ignore")
import real_ijepa_test as R
from gate import auroc
from novelty_latent import fit, distance, knn_novelty

import os
lat = {s: np.load(f"data/fungi_real/{s}.npy") for s in R.SPECIES if os.path.exists(f"data/fungi_real/{s}.npy")}
tr_species = [s for s in R.SPECIES if R.SPECIES[s][1] in ("edible", "poison") and s in lat]
train = np.vstack([lat[s] for s in tr_species])
muT, VT = R.pca(train, 16)
nhold = max(4, len(train) // 6)
Xtr = R.proj(train[:-nhold], muT, VT)
ind = R.proj(train[-nhold:], muT, VT)
ood = R.proj(lat["heldout_deathcap"], muT, VT)
y = np.r_[np.zeros(len(ind)), np.ones(len(ood))]          # 1 = OOD (death cap), want high score

# (1) Gaussian generative likelihood  (OOD score = Mahalanobis distance = -log-likelihood)
mu, inv = fit(Xtr)
maha = np.r_[distance(ind, mu, inv), distance(ood, mu, inv)]
auc_gauss = auroc(maha, y)

# (2) GMM generative likelihood (richer generative density; sklearn if available, else numpy fallback)
auc_gmm, gmm_tag = None, ""
try:
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=4, covariance_type="full", reg_covar=1e-2, random_state=0).fit(Xtr)
    ll_ind, ll_ood = gm.score_samples(ind), gm.score_samples(ood)   # log-likelihood, high = in-dist
    auc_gmm = auroc(np.r_[-ll_ind, -ll_ood], y); gmm_tag = "(sklearn GMM k=4)"
except Exception:
    # numpy fallback: k-means into 4 comps, per-comp Gaussian, mixture log-likelihood
    rng = np.random.default_rng(0); C = Xtr[rng.choice(len(Xtr), 4, replace=False)]
    for _ in range(25):
        d = ((Xtr[:, None] - C[None]) ** 2).sum(-1); a = d.argmin(1)
        C = np.array([Xtr[a == k].mean(0) if (a == k).any() else C[k] for k in range(4)])
    comps = []
    for k in range(4):
        Xk = Xtr[a == k]
        if len(Xk) < 5: continue
        comps.append(fit(Xk) + (len(Xk) / len(Xtr),))
    def nll(Z):
        L = np.array([[ -0.5 * distance(z[None], m, iv)[0] ** 2 + np.log(w) for (m, iv, w) in comps] for z in Z])
        return -(np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1))  # -log mixture likelihood
    auc_gmm = auroc(np.r_[nll(ind), nll(ood)], y); gmm_tag = "(numpy GMM k=4)"

# (3) discriminative geometry (the JEPA keeper)
knn = np.r_[knn_novelty(ind, Xtr, 5), knn_novelty(ood, Xtr, 5)]
auc_knn = auroc(knn, y)

print("open-set: detect held-out DEATH CAP as unknown (higher AUROC = better; 0.5 = chance)")
print(f"  GENERATIVE  Gaussian likelihood : AUROC {auc_gauss:.2f}")
print(f"  GENERATIVE  GMM likelihood {gmm_tag:<16}: AUROC {auc_gmm:.2f}")
print(f"  DISCRIMIN.  kNN geometry (JEPA keeper): AUROC {auc_knn:.2f}")
# pathology check: is OOD log-likelihood actually LOWER (it should be, if the detector works)?
print(f"\n  pathology check (mean Mahalanobis dist; OOD SHOULD be larger):")
print(f"    in-distribution held-out: {distance(ind, mu, inv).mean():.2f}   death cap (OOD): {distance(ood, mu, inv).mean():.2f}")
print("  => if generative AUROC <= chance / OOD not clearly farther, a diffusion DENSITY detector")
print("     exhibits 'don't know what they don't know' -- WORSE than discriminative geometry here.")
print("SAFETY: research vehicle only; when in doubt, DON'T EAT.")
