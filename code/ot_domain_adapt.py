"""OT pushed CONSTRUCTIVELY (not just "Wasserstein vanishes on the floor"): Optimal-Transport domain
adaptation for the curated->field distribution shift.

SOURCE = curated I-JEPA latents (clean GBIF photos, labeled). TARGET = the --augment latents (blur +
stem-base occlusion = field-condition proxy), treated as UNLABELED. We train a classifier on SOURCE
and classify the field TARGET, with three adaptations -- and FAIR baselines so OT has to earn it:
    DIRECT   : no adaptation (train curated, test field)
    CORAL    : align target 2nd-order stats to source (a simple, cheap baseline)
    OT (Sinkhorn) : entropic-OT transport target->source + barycentric map, then classify
Honest question: does OT beat BOTH direct and CORAL? If it only ties CORAL, OT isn't worth it here.
Lethal pair: field mushroom (edible) vs Destroying Angel (poison). SAFETY: research vehicle. DON'T EAT.
"""
import numpy as np
from gate import auroc
np.seterr(all="ignore")

ROOT = "data/fungi_real"
Se, Sp = np.load(f"{ROOT}/edible_fieldmushroom.npy"), np.load(f"{ROOT}/poison_destroyingangel.npy")
Te, Tp = np.load(f"{ROOT}/edible_fieldmushroom_augment.npy"), np.load(f"{ROOT}/poison_destroyingangel_augment.npy")
S = np.vstack([Se, Sp]); ys = np.r_[np.zeros(len(Se)), np.ones(len(Sp))]
T = np.vstack([Te, Tp]); yt = np.r_[np.zeros(len(Te)), np.ones(len(Tp))]

mu = S.mean(0); _, _, Vt = np.linalg.svd(S - mu, full_matrices=False); V = Vt[:64]
proj = lambda X: ((X - mu) @ V.T).astype("f8")
S64, T64 = proj(S), proj(T)
ec, pc = S64[ys == 0].mean(0), S64[ys == 1].mean(0)          # nearest-centroid source classifier
risk = lambda X: np.linalg.norm(X - ec, axis=1) - np.linalg.norm(X - pc, axis=1)   # high -> poison


def poison_at_yield(r, y, yld=0.6):                          # eat if risk<thr; thr set so yld of edibles eaten
    thr = np.quantile(r[y == 0], yld); return float((r[y == 1] < thr).mean())


def coral(Xs, Xt):                                           # whiten target cov, recolor to source cov
    def root(C, p):
        w, Q = np.linalg.eigh(C + 1e-3 * np.eye(len(C))); w = np.clip(w, 1e-6, None)
        return Q @ np.diag(w ** p) @ Q.T
    A = root(np.cov(Xt.T), -0.5) @ root(np.cov(Xs.T), 0.5)
    return (Xt - Xt.mean(0)) @ A + Xs.mean(0)


def sinkhorn(M, reg, iters=400):                             # entropic OT plan, marginals uniform
    K = np.exp(-M / reg); a = np.full(M.shape[0], 1 / M.shape[0]); b = np.full(M.shape[1], 1 / M.shape[1])
    u = np.ones(M.shape[0])
    for _ in range(iters):
        v = b / (K.T @ u + 1e-300); u = a / (K @ v + 1e-300)
    return u[:, None] * K * v[None, :]


def report(name, r):
    print(f"  {name:<34} | AUROC {auroc(r, yt):.3f} | poison-eaten@60% yield {poison_at_yield(r, yt)*100:5.1f}%")


print(f"source(curated) {len(S)}  target(field/augment) {len(T)}  |  lethal pair, source-trained classifier\n")
print("  classify the FIELD (augmented) latents, source-trained classifier:")
report("DIRECT (no adaptation)", risk(T64))
report("CORAL (2nd-order match) [baseline]", risk(coral(S64, T64)))

M = ((T64[:, None, :] - S64[None, :, :]) ** 2).sum(-1)       # cost target->source
P = sinkhorn(M, reg=0.3 * np.median(M))
mapped = (P @ S64) / (P.sum(1, keepdims=True) + 1e-300)      # barycentric map of target into source
report("OT-Sinkhorn (transport + map)", risk(mapped))

# honest sanity check: target[j] is the augment of source[j] (paired). Does OT recover that?
nE = len(Te)
args = P.argmax(1)
same_class = np.mean([(args[j] < len(Se)) == (j < nE) for j in range(len(T))])
print(f"\n  OT correspondence sanity: target's top-mass source is the SAME class {same_class*100:.0f}% of the time")
print("\nVERDICT: OT earns its keep only if it beats BOTH direct AND CORAL. If it ties CORAL, the simpler")
print("method wins and OT is not worth it here. SAFETY: research vehicle only; when in doubt, DON'T EAT.")
