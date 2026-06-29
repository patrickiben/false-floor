"""GATE-FAIRNESS COUNTER-TEST (adversarial review).

Claim under review (cascade_prototype.py lines 77-90): the base-region novelty ABSTENTION GATE
is INERT -- it moves false-eat 1.2% -> 2.5% @60% (i.e. WORSE), so it is dismissed as a
"redundant channel".

My lens: is that gate FAIRLY SPECIFIED, or self-defeating like the earlier Beta-variance term?

The original gate (line 82):  comb = max( pctrank(risk) , pctrank(base_novelty) )
then thresholds comb at the 60/80 pct of the EDIBLE comb scores.

Three things to test:
  (A) The construction. MAX-of-ranks is NOT an abstention gate. It REPLACES the risk score with
      whichever of {risk, novelty} ranks the item higher -- this *re-ranks the edibles too*, so a
      benign edible with a moderately-novel base is pushed up and POLLUTES the edible quantile,
      raising the eat-threshold and letting more poison through. A real abstention gate REMOVES
      high-novelty items from the eat-set; it does not re-rank everyone.
  (B) The premise. The author claims the base channel's ERRORS are correlated with the global
      classifier's (redundant). Measure it: residual correlation, and incremental AUROC.
  (C) A sound gate. Implement (i) a true visibility/abstention gate (abstain if base is OOD by a
      held-out-calibrated threshold; eat only from the cleared set at matched EFFECTIVE yield),
      and (ii) a max-over-regions toxic-signal rule. Does false-eat drop at a yield cost, or not?

Matched-yield accounting: "yield" = fraction of EDIBLES you actually eat. Abstaining costs yield;
the fair comparison holds yield fixed across methods and reads false-eat (poison eaten).
"""
import numpy as np
from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import knn_novelty
from gate import auroc
np.seterr(all="ignore")

ROOT = "data/fungi_real"
def load(slug, r=""): return np.load(f"{ROOT}/{slug}{r}.npy")
E, P = "edible_fieldmushroom", "poison_destroyingangel"

rng = np.random.default_rng(0)
def split(n): i = rng.permutation(n); c = int(0.7 * n); return i[:c], i[c:]
ie_tr, ie_te = split(296); ip_tr, ip_te = split(268)

def region(r):
    Xe, Xp = load(E, r), load(P, r)
    return Xe[ie_tr], Xp[ip_tr], Xe[ie_te], Xp[ip_te]

def pca(fit, *apply, k=64):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False); V = Vt[:min(k, Vt.shape[0])]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *apply)]

def feats(r, k=64):
    etr, ptr, ete, pte = region(r)
    pool = np.vstack([etr, ptr]); Ptr, Pe, Pp = pca(pool, ete, pte, k=k)
    return (Ptr[:len(etr)], Ptr[len(etr):]), Pe, Pp

def risk_of(feat_tr, feat_e, feat_p):
    Xtr = np.vstack([feat_tr[0], feat_tr[1]]); ytr = np.r_[np.zeros(len(feat_tr[0])), np.ones(len(feat_tr[1]))].astype("f4")
    ens = train_ensemble(Xtr, ytr, Xtr.shape[1], CFG)
    return ensemble_risk(ens, feat_e)[0], ensemble_risk(ens, feat_p)[0]

def cat(*Fs):
    tr_e = np.hstack([F[0][0] for F in Fs]); tr_p = np.hstack([F[0][1] for F in Fs])
    te_e = np.hstack([F[1] for F in Fs]); te_p = np.hstack([F[2] for F in Fs])
    return (tr_e, tr_p), te_e, te_p

def pctrank(x): return x.argsort().argsort() / max(len(x) - 1, 1)
def at_yield(se, sp, yld): return float((sp < np.quantile(se, yld)).mean())

gE = feats("", 64); bE = feats("_volva", 48); cE = feats("_crop", 48)
cl = cat(gE, bE, cE); rce, rcp = risk_of(*cl)
btr_e = bE[0][0]
# base-novelty computed against EDIBLE training bases (as the original does)
bnov_e = knn_novelty(bE[1], btr_e, 5); bnov_p = knn_novelty(bE[2], btr_e, 5)

ne, npz = len(rce), len(rcp)
lab = np.r_[np.zeros(ne), np.ones(npz)]

print("=" * 78)
print("(B) PREMISE CHECK: is the base channel REDUNDANT with the global/cascade classifier?")
print("=" * 78)
print(f"  cascade-risk AUROC (poison=1)      : {auroc(np.r_[rce, rcp], lab):.3f}")
print(f"  base-novelty AUROC (poison=1)      : {auroc(np.r_[bnov_e, bnov_p], lab):.3f}")
# error-correlation: among the items, correlate risk-rank with novelty-rank SEPARATELY within class
rr = pctrank(np.r_[rce, rcp]); nn = pctrank(np.r_[bnov_e, bnov_p])
re_rank, rp_rank = rr[:ne], rr[ne:]; ne_rank, np_rank = nn[:ne], nn[ne:]
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
print(f"  rank-corr(risk,novelty) within EDIBLE : {corr(re_rank, ne_rank):+.3f}")
print(f"  rank-corr(risk,novelty) within POISON : {corr(rp_rank, np_rank):+.3f}")
# the redundancy test that matters: does novelty add AUROC over risk via logistic combo?
from numpy.linalg import lstsq
Xc = np.column_stack([np.ones(ne + npz), rr, nn]); beta, *_ = lstsq(Xc, lab, rcond=None)
comb_lin = Xc @ beta
print(f"  AUROC risk+novelty (linear fuse)   : {auroc(comb_lin, lab):.3f}   "
      f"(if > risk-alone, the channel is NOT fully redundant)")
print()

print("=" * 78)
print("(A) DECONSTRUCTING line 82: max(rank_risk, rank_novelty) is not an abstention gate")
print("=" * 78)
allr = np.r_[rce, rcp]; alln = np.r_[bnov_e, bnov_p]
comb_max = np.maximum(pctrank(allr), pctrank(alln))
ce_m, cp_m = comb_max[:ne], comb_max[ne:]
for yld in (.6, .8):
    print(f"  @{int(yld*100)}% yield  risk-only={at_yield(rce, rcp, yld)*100:4.1f}%   "
          f"MAX-comb(line82)={at_yield(ce_m, cp_m, yld)*100:4.1f}%   "
          f"AUROC(MAX)={auroc(comb_max, lab):.3f}")
print("  -> MAX pulls the strong risk signal DOWN toward the weaker novelty rank, and re-ranks")
print("     edibles, inflating the eat-threshold. That is a construction artifact, not 'inert'.")
print()

print("=" * 78)
print("(C) PROPERLY-SPECIFIED ABSTENTION GATE (visibility gate, held-out-calibrated)")
print("=" * 78)
# A real gate: ABSTAIN (refuse to eat) when base-novelty exceeds a threshold calibrated on a
# CLEAN held-out edible calibration split. Eat decision on the CLEARED set uses risk only.
# To calibrate honestly we split the EDIBLE TEST set into calib/eval so the gate threshold never
# sees the evaluation edibles' labels-as-poison. (Poison is never used to set the gate.)
nge = ne
cal_idx = np.arange(0, nge, 2)      # half the edibles calibrate the gate
ev_idx = np.arange(1, nge, 2)       # other half evaluated
bnov_e_cal, bnov_e_ev = bnov_e[cal_idx], bnov_e[ev_idx]
rce_ev = rce[ev_idx]

def gated_false_eat(gate_pct, eat_yld):
    """Abstain if base-novelty > gate_pct-percentile of CALIB edible novelty.
    Among CLEARED items, eat the lowest-risk fraction so that eat_yld of *all* edibles are eaten.
    Returns (false_eat_among_poison, effective_edible_yield)."""
    thr_nov = np.percentile(bnov_e_cal, gate_pct)
    e_clear = bnov_e_ev <= thr_nov
    p_clear = bnov_p <= thr_nov
    # risk threshold set so that eat_yld of ALL eval edibles get eaten (cleared AND low-risk)
    # operate on cleared edibles' risk; choose risk-thr to hit overall yield target
    # overall edible yield = (#edibles cleared AND risk<thr) / (#eval edibles)
    re_clear = rce_ev[e_clear]
    if len(re_clear) == 0: return 1.0, 0.0
    # target count of edibles eaten
    target = int(round(eat_yld * len(rce_ev)))
    target = min(target, len(re_clear))
    if target <= 0: return 0.0, 0.0
    risk_thr = np.sort(re_clear)[target - 1]
    eaten_e = e_clear & (rce_ev <= risk_thr)
    eaten_p = p_clear & (rcp <= risk_thr)
    yld = eaten_e.mean()
    fe = eaten_p.mean()
    return float(fe), float(yld)

# baseline (no gate) on the SAME eval split, matched yield
def baseline_false_eat(eat_yld):
    target = int(round(eat_yld * len(rce_ev)))
    risk_thr = np.sort(rce_ev)[min(target, len(rce_ev)) - 1]
    return float((rcp <= risk_thr).mean()), float((rce_ev <= risk_thr).mean())

print("  Matched EFFECTIVE yield. Gate abstains on high base-novelty; eat = cleared & low-risk.")
print(f"  {'target yld':>10} | {'baseline FE/yld':>20} | {'gated(99pct) FE/yld':>22} | {'gated(95pct) FE/yld':>22}")
for yld in (.6, .8):
    b_fe, b_y = baseline_false_eat(yld)
    g99_fe, g99_y = gated_false_eat(99.0, yld)
    g95_fe, g95_y = gated_false_eat(95.0, yld)
    print(f"  {yld:>10.2f} | {f'{b_fe*100:4.1f}% / {b_y*100:4.1f}%':>20} | "
          f"{f'{g99_fe*100:4.1f}% / {g99_y*100:4.1f}%':>22} | "
          f"{f'{g95_fe*100:4.1f}% / {g95_y*100:4.1f}%':>22}")
print()

print("=" * 78)
print("(C2) MAX-OVER-REGIONS TOXIC-SIGNAL RULE (eat only if NO region flags poison)")
print("=" * 78)
# Train a per-region poison classifier on global, base, crop; danger = MAX risk across regions;
# eat only if max-region-risk is low. This is the 'any part can veto' rule.
rge, rgp = risk_of(*gE)        # global-only risk
rbe, rbp = risk_of(*bE)        # base-only risk
rxe, rxp = risk_of(*cE)        # crop-only risk
maxe = np.maximum.reduce([rge, rbe, rxe]); maxp = np.maximum.reduce([rgp, rbp, rxp])
print(f"  global-only  AUROC={auroc(np.r_[rge, rgp], lab):.3f}  "
      f"base-only AUROC={auroc(np.r_[rbe, rbp], lab):.3f}  "
      f"crop-only AUROC={auroc(np.r_[rxe, rxp], lab):.3f}")
print(f"  MAX-over-regions AUROC={auroc(np.r_[maxe, maxp], lab):.3f}")
for yld in (.6, .8):
    print(f"  @{int(yld*100)}% yield  global-only={at_yield(rge, rgp, yld)*100:4.1f}%   "
          f"MAX-over-regions={at_yield(maxe, maxp, yld)*100:4.1f}%   "
          f"cascade-concat={at_yield(rce, rcp, yld)*100:4.1f}%")
