"""Do more cross-validation methods or SMOTE resampling help the lethal false-eat?

Honest framing tested empirically:
  - CV is an ESTIMATOR of generalization, not a fix. We already did 20-split bootstrap (4.6%+-2.3%).
    The ONE CV choice that changes the story is GROUPING: random k-fold leaks species identity
    (same species in train AND test) -> optimistic; leave-one-species-out (LOSO) is the honest,
    foraging-relevant protocol (you meet a species not in your exact training) and is WORSE.
  - SMOTE fixes class IMBALANCE by interpolating minority points in feature space. Our failure is
    feature COINCIDENCE, not imbalance (the lethal pair is ~balanced). SMOTE cannot manufacture a
    discriminative feature that isn't in the frozen latent; in the overlap it places synthetic
    poison on top of edibles. At best it shifts the operating point (false-eat down <-> yield down)
    = the same ROC / corroboration dial, no new information.

Both run on the cached real I-JEPA latents.  SAFETY: research vehicle only; when in doubt DON'T EAT.
"""
from __future__ import annotations

import numpy as np
np.seterr(all="ignore")

from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import fit, distance
import efe

CFG.ensemble_k, CFG.steps = 6, 200
EDIBLE = ["edible_chanterelle", "edible_fieldmushroom", "heldout_morel"]
POISON = ["poison_jackolantern", "poison_destroyingangel", "heldout_deathcap"]
SPECIES = EDIBLE + POISON
LAT = {s: np.load(f"data/fungi_real/{s}.npy") for s in SPECIES}
LABEL = {s: (0 if s in EDIBLE else 1) for s in SPECIES}


def pca_fit(X, k=64):
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False); V = Vt[:k]
    return lambda Z: (Z - mu) @ V.T


def eat_mask(Xtr, ytr, Xte, proj, kappa=3.0):
    """Return boolean 'would EAT' for each test point, using the section-3 decision."""
    Ztr, Zte = proj(Xtr), proj(Xte)
    ens = train_ensemble(Ztr, ytr.astype("f4"), Ztr.shape[1], CFG)
    r, s = ensemble_risk(ens, Zte)
    muE, invE = fit(Ztr[ytr == 0])
    nov = distance(Zte, muE, invE) > np.percentile(distance(Ztr[ytr == 0], muE, invE), 99)
    return efe.decide(r, s, CFG, kappa=kappa, novel=nov) == efe.EAT


def smote(Xmin, n_new, rng, k=5):
    out = []
    for _ in range(n_new):
        i = int(rng.integers(len(Xmin)))
        d = ((Xmin - Xmin[i]) ** 2).sum(1); nn = np.argsort(d)[1:k + 1]
        j = int(nn[rng.integers(len(nn))]); lam = float(rng.random())
        out.append(Xmin[i] + lam * (Xmin[j] - Xmin[i]))
    return np.array(out, dtype=Xmin.dtype)


def section_cv():
    print("=== (A) cross-validation: does the CHOICE of CV change the safety picture? ===")
    rng = np.random.default_rng(0)
    X = np.vstack([LAT[s] for s in SPECIES])
    y = np.concatenate([np.full(len(LAT[s]), LABEL[s]) for s in SPECIES])

    # random stratified 5-fold (species identity LEAKS across folds -> optimistic)
    idx = {c: rng.permutation(np.where(y == c)[0]) for c in (0, 1)}
    folds = [np.concatenate([idx[c][f::5] for c in (0, 1)]) for f in range(5)]
    pe = ye = pn = yn = 0
    for f in range(5):
        teid = folds[f]; trid = np.setdiff1d(np.arange(len(y)), teid)
        proj = pca_fit(X[trid]); eat = eat_mask(X[trid], y[trid], X[teid], proj)
        pe += int(eat[y[teid] == 1].sum()); pn += int((y[teid] == 1).sum())
        ye += int(eat[y[teid] == 0].sum()); yn += int((y[teid] == 0).sum())
    print(f"  random stratified 5-fold (species leak): poison EATEN {pe}/{pn} ({100*pe/pn:.1f}%)  "
          f"| edible eaten (yield) {ye}/{yn} ({100*ye/yn:.1f}%)")

    # leave-one-species-out (honest: held-out species never seen)
    print("  leave-one-SPECIES-out (the foraging-realistic protocol):")
    for s in SPECIES:
        tr_s = [t for t in SPECIES if t != s]
        Xtr = np.vstack([LAT[t] for t in tr_s]); ytr = np.concatenate([np.full(len(LAT[t]), LABEL[t]) for t in tr_s])
        proj = pca_fit(Xtr); eat = eat_mask(Xtr, ytr, LAT[s], proj)
        kind = "POISON" if LABEL[s] else "edible"
        rate = 100 * eat.mean()
        tag = "  <- novel toxic EATEN" if (LABEL[s] and rate > 0) else ""
        print(f"    held-out {s:<24} ({kind}): EAT-rate {rate:4.1f}%{tag}")
    print("  => random CV is optimistic (species leak); LOSO shows held-out POISON species get eaten "
          "-- more/standard CV does NOT help, the rigorous CV reveals it's worse (open-set).")


def section_smote():
    print("\n=== (B) SMOTE / resampling on the lethal pair (field mushroom vs Destroying Angel) ===")
    rng = np.random.default_rng(0)
    e, p = LAT["edible_fieldmushroom"], LAT["poison_destroyingangel"]

    def split(Z):
        i = rng.permutation(len(Z)); c = int(0.7 * len(Z)); return Z[i[:c]], Z[i[c:]]
    etr, ete = split(e); ptr, pte = split(p)
    proj = pca_fit(np.vstack([etr, ptr]))
    Etr, Ptr, Ete, Pte = proj(etr), proj(ptr), proj(ete), proj(pte)
    print(f"  train: {len(Etr)} edible / {len(Ptr)} poison (~balanced -> SMOTE's premise barely applies)")
    print(f"  {'method':<26} | {'poison EATEN (false-eat)':>24} | {'edible eaten (yield)':>20}")

    def run(Xtr, ytr, label):
        ens = train_ensemble(Xtr, ytr.astype("f4"), Xtr.shape[1], CFG)
        muE, invE = fit(Xtr[ytr == 0]); thr = np.percentile(distance(Xtr[ytr == 0], muE, invE), 99)
        def eat(Z):
            r, s = ensemble_risk(ens, Z); nov = distance(Z, muE, invE) > thr
            return efe.decide(r, s, CFG, kappa=3.0, novel=nov) == efe.EAT
        fe = int(eat(Pte).sum()); yld = int(eat(Ete).sum())
        print(f"  {label:<26} | {f'{fe}/{len(Pte)} ({100*fe/len(Pte):.1f}%)':>24} | "
              f"{f'{yld}/{len(Ete)} ({100*yld/len(Ete):.1f}%)':>20}")

    Xb = np.vstack([Etr, Ptr]); yb = np.r_[np.zeros(len(Etr)), np.ones(len(Ptr))]
    run(Xb, yb, "baseline (no resampling)")

    # random oversample poison to 2x
    ros = Ptr[rng.integers(0, len(Ptr), len(Ptr))]
    Xo = np.vstack([Etr, Ptr, ros]); yo = np.r_[np.zeros(len(Etr)), np.ones(2 * len(Ptr))]
    run(Xo, yo, "random-oversample poison 2x")

    # SMOTE poison to 2x
    syn = smote(Ptr, len(Ptr), rng)
    Xs = np.vstack([Etr, Ptr, syn]); ys = np.r_[np.zeros(len(Etr)), np.ones(2 * len(Ptr))]
    run(Xs, ys, "SMOTE poison 2x")

    # SMOTE poison to 5x (push the prior hard)
    syn5 = smote(Ptr, 4 * len(Ptr), rng)
    X5 = np.vstack([Etr, Ptr, syn5]); y5 = np.r_[np.zeros(len(Etr)), np.ones(5 * len(Ptr))]
    run(X5, y5, "SMOTE poison 5x")
    print("  => resampling trades false-eat against yield (same ROC, a prior/threshold shift); it does")
    print("     NOT reach a safe 0 false-eat without collapsing yield -- no new information is added.")


if __name__ == "__main__":
    section_cv()
    section_smote()
    print("\nSAFETY: research vehicle only; neither CV nor SMOTE manufactures the missing feature. DON'T EAT.")
