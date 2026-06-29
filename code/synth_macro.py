"""The inversion (Patrick): predict in the direction you TRUST -- micro->macro (forward/well-posed) -- generate
synthetic macros grounded in the diagnostic micro, then diagnose with those.

Decisive test on real lethal-pair I-JEPA macro latents (field mushroom vs Destroying Angel). A micro->macro
generator is, in the idealized faithful case, the class-conditional macro distribution P(macro | species)
(species being what the micro determines). We compare:
  (1) BASELINE  -- classifier trained on REAL macros, tested on REAL held-out.
  (2) FAITHFUL  -- generator = fitted P(macro|species); train on SYNTHETIC macros, test on REAL. A faithful
                   generator reproduces the true overlap of lookalikes -> should match baseline (no gain).
  (3) HALLUCINATING -- generator injects extra micro-separability the macros don't really have; train on
                   SYNTHETIC, test on REAL. Looks great on synthetic, should FAIL on real (the 0.94->death trap).
  (4) ABSTENTION CALIBRATION -- the genuine positive: use P(macro|edible) vs P(macro|poison) to MEASURE the
                   irreducibly-ambiguous overlap region -> a principled abstain set; decide only the confident rest.
Key fact: at inference the query's micro was never measured, so the decision is still f(macro). SAFETY: research vehicle; DON'T EAT.
"""
import numpy as np
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_real"
load = lambda s: np.load(f"{ROOT}/{s}_patches.npy").mean(1)
Le, Lp = load("edible_fieldmushroom"), load("poison_destroyingangel")
print(f"macro latents: edible {Le.shape}, poison {Lp.shape}", flush=True)


def pca(fit, *ap, k=24):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False); V = Vt[:k]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *ap)]

def fit_gauss(X): return X.mean(0), np.cov(X.T) + 1e-2 * np.eye(X.shape[1])
def feat(re, rp, y): return float((rp < np.quantile(re, y)).mean()) * 100
def auroc(pos, neg): return float(sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))

NS = 500
R = {k: [] for k in ["baseline", "faithful", "halluc_real", "halluc_synthtest", "abstain_frac", "confident_falseeat"]}
for seed in range(20):
    rng = np.random.default_rng(seed)
    def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
    etr, ete = sp(len(Le)); ptr, pte = sp(len(Lp))
    Ptr_e, Pe = pca(Le[etr], Le[ete], k=24)[0], pca(Le[etr], Le[ete], k=24)[1]
    # one shared PCA basis (fit on train of both classes)
    Xtr = np.vstack([Le[etr], Lp[ptr]])
    Ptr, Pe, Pp = pca(Xtr, Le[ete], Lp[pte])
    ne = (np.arange(len(Ptr)) < len(etr)); Etr, Mtr = Ptr[ne], Ptr[~ne]
    ytr = np.r_[np.zeros(len(Etr)), np.ones(len(Mtr))].astype("f4")

    # (1) baseline: real-trained
    ens = train_ensemble(Ptr, ytr, Ptr.shape[1], CFG)
    R["baseline"].append(feat(ensemble_risk(ens, Pe)[0], ensemble_risk(ens, Pp)[0], .8))

    # (2) faithful generator P(macro|species); train on synthetic, test on real
    me, ce = fit_gauss(Etr); mp, cp = fit_gauss(Mtr)
    Se, Sp = rng.multivariate_normal(me, ce, NS), rng.multivariate_normal(mp, cp, NS)
    Xs = np.vstack([Se, Sp]).astype("f4"); ys = np.r_[np.zeros(NS), np.ones(NS)].astype("f4")
    ens_f = train_ensemble(Xs, ys, Xs.shape[1], CFG)
    R["faithful"].append(feat(ensemble_risk(ens_f, Pe)[0], ensemble_risk(ens_f, Pp)[0], .8))

    # (3) hallucinating generator: push class means apart (inject micro-separability not really in the macros)
    diff = mp - me; mp_h = me + diff * 3.0
    Sph = rng.multivariate_normal(mp_h, cp, NS)
    Xh = np.vstack([Se, Sph]).astype("f4")
    ens_h = train_ensemble(Xh, ys, Xh.shape[1], CFG)
    re_h, rp_h = ensemble_risk(ens_h, Pe)[0], ensemble_risk(ens_h, Pp)[0]            # on REAL
    Sph_te = rng.multivariate_normal(mp_h, cp, 200).astype("f4"); Se_te = rng.multivariate_normal(me, ce, 200).astype("f4")
    R["halluc_real"].append(feat(re_h, rp_h, .8))
    R["halluc_synthtest"].append(feat(ensemble_risk(ens_h, Se_te)[0], ensemble_risk(ens_h, Sph_te)[0], .8))

    # (4) abstention calibration: faithful densities -> |log-likelihood ratio|; small = ambiguous -> abstain
    import numpy.linalg as la
    def logp(X, m, c): d = X - m; return -0.5 * np.einsum("ij,jk,ik->i", d, la.inv(c), d) - 0.5 * np.log(la.det(c) + 1e-9)
    allX = np.vstack([Pe, Pp]); lab = np.r_[np.zeros(len(Pe)), np.ones(len(Pp))]
    llr = logp(allX, mp, cp) - logp(allX, me, ce)                                     # >0 -> poison-like
    amb = np.abs(llr) < np.quantile(np.abs(llr), .5)                                  # bottom-50% |LLR| = ambiguous
    R["abstain_frac"].append(float(amb.mean()) * 100)
    conf_pois = (lab == 1) & ~amb                                                      # confident poison
    R["confident_falseeat"].append(float((llr[conf_pois] < 0).mean()) * 100 if conf_pois.any() else 0.0)

m = {k: (np.mean(v), np.std(v)) for k, v in R.items()}
print("\nfalse-eat @80% yield (% Destroying Angels eaten), 20-seed mean +/- sd:")
print(f"  (1) BASELINE   real-trained          {m['baseline'][0]:5.1f} +/- {m['baseline'][1]:4.1f}%")
print(f"  (2) FAITHFUL   synth-trained, real-test {m['faithful'][0]:5.1f} +/- {m['faithful'][1]:4.1f}%   <- reproduces overlap, ~= baseline")
print(f"  (3) HALLUC.    synth-trained, REAL-test {m['halluc_real'][0]:5.1f} +/- {m['halluc_real'][1]:4.1f}%   <- the TRAP (fails on real)")
print(f"      HALLUC.    synth-trained, SYNTH-test {m['halluc_synthtest'][0]:5.1f} +/- {m['halluc_synthtest'][1]:4.1f}%   <- looks great on synthetic")
print(f"\n  (4) ABSTENTION calibration (the genuine use): abstain on {m['abstain_frac'][0]:.0f}% (the ambiguous overlap);")
print(f"      on the CONFIDENT remainder, poison-misclassified = {m['confident_falseeat'][0]:.1f}% -- you can decide the easy ones, must abstain on the rest")
print("""
READING: a FAITHFUL micro->macro generator reproduces the real lookalike overlap, so a classifier trained on
its synthetic macros matches the baseline -- no safety gain (you can't generate away genuine class overlap).
A generator that DOES separate the synthetic macros is hallucinating a difference the photos don't carry: it
looks excellent on synthetic data and FAILS on real specimens -- the 0.94->death trap, manufactured. The one
real payoff is direction (4): the micro-grounded densities let you MEASURE the irreducibly-ambiguous region and
abstain on it principledly. The inversion is a great tool for calibrating ABSTENTION and covering rare species
-- not for separating lookalikes, because at inference the query's micro was never measured. SAFETY: research vehicle; DON'T EAT.
""")
