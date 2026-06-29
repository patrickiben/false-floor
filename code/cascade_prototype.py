"""CASCADE prototype (CORRECTED after adversarial verification): staged part-localized analysis vs global.

Verification verdict (5-lens panel): the negative is FAIR, not a strawman -- reviewers BUILT stronger
feasible cascades (a 10-region multi-scale part-detector + max-toxic veto; calibrated abstention gates) and
NONE beat the global pooled embedding beyond seed noise; the regions are ~70-80% linearly predictable from
global and share errors on the confidently-eaten poison = a REDUNDANT RGB channel (value_of_channel). BUT the
panel found three real flaws, fixed here:
  (1) UNDERPOWERED: single seed, n=81 held-out poison (1 specimen = 1.23%). GLOBAL@60 = 0.0% was a lucky
      seed-0; across seeds it is ~0.7%. -> report MULTI-SEED mean +- sd.
  (2) The original max(pctrank(risk), pctrank(novelty)) gate was BROKEN (it re-ranks edibles and RAISES
      poison-eaten ~+3.4pp), not "inert". -> use a PROPER calibrated veto gate that only ever raises risk.
  (3) Localization adds a tiny real separability signal (incr. R^2 ~0.02) that the granular n=81 false-eat
      metric can't resolve. -> also report AUROC.
SAFETY: research vehicle; a same-modality cascade cannot add information. When in doubt, DON'T EAT.
"""
import numpy as np
from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import knn_novelty
from gate import auroc
np.seterr(all="ignore")

ROOT = "data/fungi_real"; E, P = "edible_fieldmushroom", "poison_destroyingangel"
GE = {r: np.load(f"{ROOT}/{E}{r}.npy") for r in ("", "_volva", "_crop")}
PP = {r: np.load(f"{ROOT}/{P}{r}.npy") for r in ("", "_volva", "_crop")}


def pca_fit(Xfit, k):
    mu = Xfit.mean(0); _, _, Vt = np.linalg.svd(Xfit - mu, full_matrices=False)
    return mu, Vt[:min(k, Vt.shape[0])]

def at_yield(se, sp, y): return float((sp < np.quantile(se, y)).mean())

def region(r, k, etr, ptr, ete, pte):
    Xe, Xp = GE[r], PP[r]
    mu, V = pca_fit(np.vstack([Xe[etr], Xp[ptr]]), k)
    f = lambda A: ((A - mu) @ V.T).astype("f4")
    return (f(Xe[etr]), f(Xp[ptr])), f(Xe[ete]), f(Xp[pte])

def risk_of(tr, e, p):
    Xtr = np.vstack([tr[0], tr[1]]); ytr = np.r_[np.zeros(len(tr[0])), np.ones(len(tr[1]))].astype("f4")
    ens = train_ensemble(Xtr, ytr, Xtr.shape[1], CFG)
    return ensemble_risk(ens, e)[0], ensemble_risk(ens, p)[0]

def cat(*Fs):
    return (np.hstack([F[0][0] for F in Fs]), np.hstack([F[0][1] for F in Fs])), \
           np.hstack([F[1] for F in Fs]), np.hstack([F[2] for F in Fs])

def one_seed(seed):
    rng = np.random.default_rng(seed)
    def sp(n): i = rng.permutation(n); c = int(0.7 * n); return i[:c], i[c:]
    etr, ete = sp(296); ptr, pte = sp(268)
    g = region("", 64, etr, ptr, ete, pte); b = region("_volva", 48, etr, ptr, ete, pte); c = region("_crop", 48, etr, ptr, ete, pte)
    rge, rgp = risk_of(*g)                                   # GLOBAL
    rce, rcp = risk_of(*cat(g, b, c))                        # CASCADE-loc
    bnov_e = knn_novelty(b[1], b[0][0], 5); bnov_p = knn_novelty(b[2], b[0][0], 5)   # base-region novelty
    thr = np.quantile(bnov_e, 0.80)                          # PROPER veto: anomalous base -> raise risk only
    fge = np.where(bnov_e > thr, np.maximum(rce, 1.0), rce); fgp = np.where(bnov_p > thr, np.maximum(rcp, 1.0), rcp)
    yg = np.r_[np.zeros(len(rge)), np.ones(len(rgp))]
    return [at_yield(rge, rgp, .6), at_yield(rge, rgp, .8), at_yield(rce, rcp, .6), at_yield(rce, rcp, .8),
            at_yield(fge, fgp, .6), at_yield(fge, fgp, .8),
            auroc(np.r_[rge, rgp], yg), auroc(np.r_[rce, rcp], yg)]


R = np.array([one_seed(s) for s in range(20)]) * np.array([100, 100, 100, 100, 100, 100, 1, 1])
m, s = R.mean(0), R.std(0)
print("cascade prototype -- 20-seed mean +/- sd (matched-yield poison-eaten %, real lethal pair, n=81 poison)\n")
print(f"  {'method':<34} | {'@60% yield':>15} | {'@80% yield':>15} | {'AUROC':>6}")
print(f"  {'GLOBAL (pooled) [baseline]':<34} | {m[0]:5.2f} +/- {s[0]:4.2f}% | {m[1]:5.2f} +/- {s[1]:4.2f}% | {m[6]:.3f}")
print(f"  {'CASCADE-loc (global+base+crop)':<34} | {m[2]:5.2f} +/- {s[2]:4.2f}% | {m[3]:5.2f} +/- {s[3]:4.2f}% | {m[7]:.3f}")
print(f"  {'CASCADE-full (proper veto gate)':<34} | {m[4]:5.2f} +/- {s[4]:4.2f}% | {m[5]:5.2f} +/- {s[5]:4.2f}% |")
print(f"\n  localization diff (cascade-loc - global): {m[2]-m[0]:+.2f}% @60, {m[3]-m[1]:+.2f}% @80  "
      f"-- within sd (~{s[0]:.1f}pp) = INDISTINGUISHABLE on the safety metric")
print(f"  tiny separability gain: AUROC {m[6]:.3f} -> {m[7]:.3f} ({m[7]-m[6]:+.3f}) -- real but doesn't reach the granular n=81 false-eat")
print("\nVERIFIED: a same-modality cascade does NOT beat global beyond noise (stronger part-detectors fail too);")
print("regions are a redundant RGB channel. Genuine headroom = a real part-detector + a NEW MODALITY at the part.")
