"""ADVERSARIAL AUDIT of cascade_prototype.py: is GLOBAL 0.0%@60% representative or a lucky-seed
artifact? Re-run GLOBAL vs CASCADE-loc vs CASCADE-full across many RNG seeds (both the train/test
split AND the ensemble) and report the distribution of matched-yield poison-eaten for each.

Faithful re-implementation of the prototype's pipeline, parameterized by seed.
"""
import numpy as np
from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import knn_novelty
import torch
np.seterr(all="ignore")

ROOT = "data/fungi_real"
def load(slug, r=""): return np.load(f"{ROOT}/{slug}{r}.npy")
E, P = "edible_fieldmushroom", "poison_destroyingangel"

def pca(fit, *apply, k=64):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False); V = Vt[:min(k, Vt.shape[0])]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *apply)]

def at_yield(se, sp, yld):
    return float((sp < np.quantile(se, yld)).mean())

def pctrank(x): return x.argsort().argsort() / max(len(x) - 1, 1)

def run_once(split_seed, ens_seed_base):
    rng = np.random.default_rng(split_seed)
    def split(n): i = rng.permutation(n); c = int(0.7 * n); return i[:c], i[c:]
    ie_tr, ie_te = split(296); ip_tr, ip_te = split(268)

    def region(r):
        Xe, Xp = load(E, r), load(P, r)
        return Xe[ie_tr], Xp[ip_tr], Xe[ie_te], Xp[ip_te]

    def feats(r, k=64):
        etr, ptr, ete, pte = region(r)
        pool = np.vstack([etr, ptr])
        Ptr, Pe, Pp = pca(pool, ete, pte, k=k)
        return (Ptr[:len(etr)], Ptr[len(etr):]), Pe, Pp

    gE = feats("", 64); bE = feats("_volva", 48); cE = feats("_crop", 48)

    def risk_of(feat_tr, feat_e, feat_p):
        Xtr = np.vstack([feat_tr[0], feat_tr[1]])
        ytr = np.r_[np.zeros(len(feat_tr[0])), np.ones(len(feat_tr[1]))].astype("f4")
        # vary ensemble seeds so we don't reuse identical members each call
        torch.manual_seed(ens_seed_base)
        ens = train_ensemble(Xtr, ytr, Xtr.shape[1], CFG)
        return ensemble_risk(ens, feat_e)[0], ensemble_risk(ens, feat_p)[0]

    rge, rgp = risk_of(*gE)
    def cat(*Fs):
        tr_e = np.hstack([F[0][0] for F in Fs]); tr_p = np.hstack([F[0][1] for F in Fs])
        te_e = np.hstack([F[1] for F in Fs]); te_p = np.hstack([F[2] for F in Fs])
        return (tr_e, tr_p), te_e, te_p
    cl = cat(gE, bE, cE)
    rce, rcp = risk_of(*cl)

    btr_e = bE[0][0]
    bnov_e = knn_novelty(bE[1], btr_e, 5); bnov_p = knn_novelty(bE[2], btr_e, 5)
    allr = np.r_[rce, rcp]; alln = np.r_[bnov_e, bnov_p]
    comb = np.maximum(pctrank(allr), pctrank(alln)); ce, cp = comb[:len(rce)], comb[len(rce):]

    out = {}
    for y in (.6, .8):
        out[("GLOBAL", y)] = at_yield(rge, rgp, y)
        out[("CASC-loc", y)] = at_yield(rce, rcp, y)
        out[("CASC-full", y)] = at_yield(ce, cp, y)
    return out

if __name__ == "__main__":
    N = 40
    rows = {k: [] for k in
            [("GLOBAL", .6), ("CASC-loc", .6), ("CASC-full", .6),
             ("GLOBAL", .8), ("CASC-loc", .8), ("CASC-full", .8)]}
    for s in range(N):
        o = run_once(split_seed=s, ens_seed_base=1000 + s)
        for k in rows: rows[k].append(o[k] * 100)
        if s == 0:
            print("SEED-0 reproduction (should match prototype @60: G0.0 L1.2 F2.5):")
            print("  G60=%.1f L60=%.1f F60=%.1f | G80=%.1f L80=%.1f F80=%.1f" % (
                o[("GLOBAL", .6)]*100, o[("CASC-loc", .6)]*100, o[("CASC-full", .6)]*100,
                o[("GLOBAL", .8)]*100, o[("CASC-loc", .8)]*100, o[("CASC-full", .8)]*100))

    print(f"\nAcross {N} split+ensemble seeds  (matched-yield POISON-EATEN %, lower=safer)")
    print(f"  {'method@yield':<16} | {'mean':>6} | {'median':>6} | {'std':>5} | {'min':>5} | {'max':>5} | {'%==0':>5}")
    for k in rows:
        v = np.array(rows[k]); name = f"{k[0]}@{int(k[1]*100)}"
        print(f"  {name:<16} | {v.mean():6.2f} | {np.median(v):6.2f} | {v.std():5.2f} | "
              f"{v.min():5.1f} | {v.max():5.1f} | {(v==0).mean()*100:4.0f}%")

    print("\nPAIRED cascade-vs-global (positive = cascade EATS MORE poison = worse):")
    for y in (.6, .8):
        g = np.array(rows[("GLOBAL", y)]); l = np.array(rows[("CASC-loc", y)]); f = np.array(rows[("CASC-full", y)])
        dl = l - g; df = f - g
        print(f"  @{int(y*100)}%  loc-global: mean diff {dl.mean():+.2f}  (loc better in {(dl<0).mean()*100:.0f}%, worse in {(dl>0).mean()*100:.0f}%, tie {(dl==0).mean()*100:.0f}%)")
        print(f"        full-global: mean diff {df.mean():+.2f}  (full better in {(df<0).mean()*100:.0f}%, worse in {(df>0).mean()*100:.0f}%, tie {(df==0).mean()*100:.0f}%)")
