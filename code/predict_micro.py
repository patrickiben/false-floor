"""Can we PREDICT the microscopy (spores/basidia/tissue) from the MACRO photo, then use it to decide eat/no?
The no-go is the data-processing inequality: predicted-micro = f(macro) carries NO MORE edibility information
than the macro itself -> a redundant channel (value_of_channel) that is confidently-wrong on the deadly
lookalikes. Sharper domain point: microscopy is DIAGNOSTIC precisely because it's ORTHOGONAL to gross
morphology (that's why mycologists use it) -- so the macro->micro map isn't a function in the cases that
matter, and the predictor can only output E[micro|macro], a blend that can't recover the true species micro.

Demonstration on REAL lethal-pair I-JEPA latents (field mushroom vs Destroying Angel): a true micro channel
(AUROC ~0.95, independent of macro) is the spore print you actually MEASURE; we compare (a) PREDICTING it from
the macro latent (a trained regressor = best macro->micro map) vs (b) MEASURING it. Prediction is bounded by
the macro classifier and fails on the confidently-wrong cases; measurement drives false-eat down. SAFETY: research vehicle; DON'T EAT.
"""
import numpy as np
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_real"
load = lambda s: np.load(f"{ROOT}/{s}_patches.npy").mean(1)        # [N,256,1280] -> macro pooled latent [N,1280]
Le, Lp = load("edible_fieldmushroom"), load("poison_destroyingangel")
print(f"macro latents: edible {Le.shape}, poison {Lp.shape}", flush=True)


def pca(fit, *ap, k=48):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False); V = Vt[:k]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *ap)]

def ridge(X, y, lam=10.0):                                         # macro -> micro regressor (best macro->micro map)
    A = X.T @ X + lam * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ y)

def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))

def feat(re, rp, y):  # false-eat (% poison eaten) at matched yield y, using a risk where lower=safer
    thr = np.quantile(re, y); return float((rp < thr).mean()) * 100


DELTA = 2.33      # true micro shift -> measured-micro AUROC ~0.95 (spores ARE diagnostic & macro-independent)
res = {k: [] for k in ["macro", "pred", "meas", "macro+pred", "macro+meas"]}
cw = []           # confidently-wrong check: predicted micro on macro-fooled poison
for seed in range(20):
    rng = np.random.default_rng(seed)
    def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
    etr, ete = sp(len(Le)); ptr, pte = sp(len(Lp))
    Xtr = np.vstack([Le[etr], Lp[ptr]]); ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))]
    Ptr, Pe, Pp = pca(Xtr, Le[ete], Lp[pte])

    # macro classifier = the information ceiling for ANY function of the macro
    ens = train_ensemble(Ptr, ytr.astype("f4"), Ptr.shape[1], CFG)
    re_m, rp_m = ensemble_risk(ens, Pe)[0], ensemble_risk(ens, Pp)[0]

    # TRUE micro channel (independent of macro): species signature + noise
    micro_e = rng.normal(0, 1, len(ete)); micro_p = rng.normal(DELTA, 1, len(pte))   # measured spore feature
    # PREDICTED micro = best macro->micro regressor (trained on the SAME train split's true micro)
    micro_tr = np.r_[rng.normal(0, 1, len(etr)), rng.normal(DELTA, 1, len(ptr))]
    w = ridge(Ptr, micro_tr); pred_e, pred_p = Pe @ w, Pp @ w        # predicted from macro only

    # higher micro -> more poison-like; convert to a "risk" (lower=safer) by negating for the edible side... use directly:
    # decision risk: macro risk (low=safe) corroborated by micro (high=poison). Fuse = macro_risk + sigmoid-ish micro.
    fuse = lambda rm, mic: rm + (mic - micro_tr.mean()) * 0.15      # add micro evidence to macro risk
    res["macro"].append(feat(re_m, rp_m, .8))
    res["pred"].append(feat(-pred_e, -pred_p, .8))                  # predicted-micro alone (neg so low=safe)
    res["meas"].append(feat(-micro_e, -micro_p, .8))               # measured-micro alone
    res["macro+pred"].append(feat(fuse(re_m, pred_e), fuse(rp_m, pred_p), .8))
    res["macro+meas"].append(feat(fuse(re_m, micro_e), fuse(rp_m, micro_p), .8))
    # confidently-wrong subset: poison the macro ranks as safest (bottom 30% macro risk) -> what does PREDICTED micro say?
    safe = rp_m <= np.quantile(rp_m, .3)
    cw.append([auroc(micro_p, micro_e), auroc(pred_p, pred_e),
               pred_p[safe].mean() - pred_e.mean(), micro_p[safe].mean() - micro_e.mean()])

m = {k: (np.mean(v), np.std(v)) for k, v in res.items()}
cw = np.array(cw)
print(f"\nfalse-eat @80% yield (% Destroying Angels eaten), 20-seed mean +/- sd:")
for k in ["macro", "pred", "meas", "macro+pred", "macro+meas"]:
    tag = {"macro": "  (macro classifier = the ceiling)", "pred": "  PREDICTED micro from macro",
           "meas": "  MEASURED micro (real spore print)", "macro+pred": "  macro corroborated by PREDICTED",
           "macro+meas": "  macro corroborated by MEASURED"}[k]
    print(f"  {k:12} {m[k][0]:5.1f} +/- {m[k][1]:4.1f}%{tag}")
print(f"\n  micro AUROC: MEASURED {cw[:,0].mean():.2f} (independent, diagnostic) vs PREDICTED {cw[:,1].mean():.2f} "
      f"(bounded by macro -- data-processing)")
print(f"  on the confidently-wrong poison (macro says safest): PREDICTED micro signal {cw[:,2].mean():+.2f} "
      f"(~0 = looks edible, FAILS) vs MEASURED {cw[:,3].mean():+.2f} (still flags poison)")
print("""
READING: PREDICTED micro ~= macro classifier and adds ~nothing when corroborating it (redundant channel,
data-processing). It is near-zero on exactly the poison the macro got wrong -- because it's a function of that
same fooling macro. MEASURED micro (an independent spore print) is what drives false-eat down. The lesson is
not 'use microscopy as a feature' -- it's 'ACQUIRE the independent observation.' You cannot infer a measurement
you did not make; predicting the micrograph (even generatively) is still f(macro), bounded by the photo.
SAFETY: research vehicle only; when in doubt, DON'T EAT.
""")
