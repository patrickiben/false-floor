"""DECISIVE noise check: is GLOBAL vs strong-localizer (S1 multi-region max-toxic) a real
difference, or is it within seed noise? n_test_poison=81 => 1 specimen = 1.23%, and every
number in the prototype moves by 1-2 specimens. Re-run across SEEDS (split + ensemble) and
report the distribution + paired win/tie/loss. Reuses prototype machinery exactly.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from config import CFG
from model import train_ensemble, ensemble_risk
np.seterr(all="ignore")

ROOT = "data/fungi_real"
E, P = "edible_fieldmushroom", "poison_destroyingangel"
gE = np.load(f"{ROOT}/{E}.npy"); gP = np.load(f"{ROOT}/{P}.npy")
rgE = np.load(f"{ROOT}/strawman_{E}_regions.npy"); rgP = np.load(f"{ROOT}/strawman_{P}_regions.npy")
R = rgE.shape[1]

def at_yield(se, sp, yld): return float((sp < np.quantile(se, yld)).mean())

def pca(fit, *apply, k=64):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    V = Vt[:min(k, Vt.shape[0])]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *apply)]

def run_one(seed):
    rng = np.random.default_rng(seed)
    def split(n): i = rng.permutation(n); c = int(0.7 * n); return i[:c], i[c:]
    ie_tr, ie_te = split(296); ip_tr, ip_te = split(268)
    torch.manual_seed(seed)

    # GLOBAL
    etr, ptr, ete, pte = gE[ie_tr], gP[ip_tr], gE[ie_te], gP[ip_te]
    Ptr, Pe, Pp = pca(np.vstack([etr, ptr]), ete, pte, k=64)
    Xtr = np.vstack([Ptr[:len(etr)], Ptr[len(etr):]])
    ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))].astype("f4")
    ens = train_ensemble(Xtr, ytr, 64, CFG)
    rge, rgp = ensemble_risk(ens, Pe)[0], ensemble_risk(ens, Pp)[0]

    # S1 multi-region max-toxic
    etr_reg = rgE[ie_tr].reshape(-1, 1280); ptr_reg = rgP[ip_tr].reshape(-1, 1280)
    mu = np.vstack([etr_reg, ptr_reg]).mean(0)
    _, _, Vt = np.linalg.svd(np.vstack([etr_reg, ptr_reg]) - mu, full_matrices=False); V = Vt[:64]
    def projr(X): return ((X.reshape(-1, 1280) - mu) @ V.T).astype("f4").reshape(X.shape[0], -1, 64)
    Petr, Pptr = projr(rgE[ie_tr]), projr(rgP[ip_tr])
    Xtr2 = np.vstack([Petr.reshape(-1, 64), Pptr.reshape(-1, 64)])
    ytr2 = np.r_[np.zeros(Petr.shape[0]*R), np.ones(Pptr.shape[0]*R)].astype("f4")
    ens2 = train_ensemble(Xtr2, ytr2, 64, CFG)
    def maxtox(X):
        N = X.shape[0]; r = ensemble_risk(ens2, projr(X).reshape(-1, 64))[0].reshape(N, R)
        return r.max(1)
    rme, rmp = maxtox(rgE[ie_te]), maxtox(rgP[ip_te])

    return {
        "g60": at_yield(rge, rgp, .6), "g80": at_yield(rge, rgp, .8),
        "s60": at_yield(rme, rmp, .6), "s80": at_yield(rme, rmp, .8),
    }

if __name__ == "__main__":
    seeds = list(range(20))
    rows = [run_one(s) for s in seeds]
    g60 = np.array([r["g60"] for r in rows]); s60 = np.array([r["s60"] for r in rows])
    g80 = np.array([r["g80"] for r in rows]); s80 = np.array([r["s80"] for r in rows])

    def summ(a): return f"mean {a.mean()*100:5.2f}%  sd {a.std()*100:4.2f}  range [{a.min()*100:.1f},{a.max()*100:.1f}]"
    print(f"GLOBAL  @60: {summ(g60)}")
    print(f"S1-MAX  @60: {summ(s60)}")
    print(f"GLOBAL  @80: {summ(g80)}")
    print(f"S1-MAX  @80: {summ(s80)}")
    d60, d80 = s60 - g60, s80 - g80   # negative = S1 safer
    print(f"\nPAIRED (S1 - GLOBAL), negative = localization SAFER:")
    print(f"  @60  mean {d60.mean()*100:+.2f}%  S1-wins {int((d60<0).sum())}  ties {int((d60==0).sum())}  S1-loses {int((d60>0).sum())}  (of {len(seeds)})")
    print(f"  @80  mean {d80.mean()*100:+.2f}%  S1-wins {int((d80<0).sum())}  ties {int((d80==0).sum())}  S1-loses {int((d80>0).sum())}  (of {len(seeds)})")
    print(f"\n1 specimen = 1.23%. Seed-0 (the prototype's only seed): "
          f"g80={g80[0]*100:.1f} s80={s80[0]*100:.1f}")
    print("If the paired mean is ~0 and wins~=losses, the localizer neither helps nor hurts")
    print("beyond noise -> the NO-HELP verdict is FAIR (just established on too few seeds).")
