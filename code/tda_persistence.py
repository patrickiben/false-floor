"""TDA pushed for REAL (not name-dropped), applying the attached PomSeg method (Oda & Imoto, Cell
Reports Methods 2026): cubical SUBLEVEL-SET persistent homology on the grayscale mushroom images,
where persistence (distance from the diagonal) separates real structure from noise.

Two genuine, fairly-baselined tests:
  (A) IMAGE topology classifies the lethal pair?  Cubical PH (sublevel + superlevel, H0 + H1) ->
      persistence-statistics feature vector -> classify field mushroom (edible) vs Destroying Angel
      (poison). FAIR baselines: trivial image stats, and the I-JEPA latent (the thing to beat, 0.94).
  (B) Are the latent MANIFOLDS even topologically distinguishable?  Rips PH (H1) of the edible vs
      poison latent clouds; bottleneck distance vs a within-class null. If topology can't tell them
      apart beyond chance, the manifolds don't differ topologically -- the honest answer to the
      question I kept name-dropping.
SAFETY: research vehicle only; topological separability is not edibility. When in doubt, DON'T EAT.
"""
import glob
import numpy as np
from PIL import Image
from gudhi import CubicalComplex
from ripser import ripser
from persim import bottleneck
from config import CFG
from model import train_ensemble, ensemble_risk
from gate import auroc

ROOT = "data/fungi_real"
PAIR = ["edible_fieldmushroom", "poison_destroyingangel"]


# ---------------- (A) cubical sublevel-set PH on images (the PomSeg method) ----------------
def topo_features(gray):
    """Persistence statistics from sublevel + superlevel cubical PH, dims H0 (components) & H1 (holes)."""
    g = gray.astype(float); feats = []
    for s in (1.0, -1.0):                                   # sublevel, then superlevel (negate)
        cc = CubicalComplex(top_dimensional_cells=s * g); cc.persistence(min_persistence=0)
        cap = float((s * g).max())
        for dim in (0, 1):
            iv = cc.persistence_intervals_in_dimension(dim)
            if len(iv) == 0: feats += [0, 0, 0, 0, 0]; continue
            iv = np.where(np.isfinite(iv), iv, cap)
            pers = iv[:, 1] - iv[:, 0]; pers = pers[pers > 1e-9]
            if len(pers) == 0: feats += [0, 0, 0, 0, 0]; continue
            p = pers / pers.sum(); ent = float(-(p * np.log(p + 1e-12)).sum())
            feats += [len(pers), float(pers.sum()), float(pers.max()), float(pers.mean()), ent]
    return feats


def image_stats(gray):                                      # trivial baseline: intensity + edge content
    g = gray.astype(float) / 255.0
    gx, gy = np.gradient(g)
    return [g.mean(), g.std(), np.abs(gx).mean() + np.abs(gy).mean(), np.percentile(g, 90), np.percentile(g, 10)]


def load_gray(slug):
    return [np.array(Image.open(p).convert("L").resize((96, 96)), dtype=float)
            for p in sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))]


def evaluate_features(X, y, rng):
    i = rng.permutation(len(y)); c = int(0.7 * len(y)); tr, te = i[:c], i[c:]
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
    Xn = ((X - mu) / sd).astype("f4")
    ens = train_ensemble(Xn[tr], y[tr].astype("f4"), X.shape[1], CFG)
    r, _ = ensemble_risk(ens, Xn[te])
    return auroc(r, y[te])


print("(A) IMAGE topology vs baselines -- classify the lethal pair (AUROC, held-out):")
imgs = {s: load_gray(s) for s in PAIR}
y = np.r_[np.zeros(len(imgs[PAIR[0]])), np.ones(len(imgs[PAIR[1]]))]
TOPO = np.array([topo_features(g) for s in PAIR for g in imgs[s]])
STAT = np.array([image_stats(g) for s in PAIR for g in imgs[s]])
LAT = np.vstack([np.load(f"{ROOT}/{PAIR[0]}.npy"), np.load(f"{ROOT}/{PAIR[1]}.npy")])
# PCA-64 of the I-JEPA latent for the baseline
mu = LAT.mean(0); _, _, Vt = np.linalg.svd(LAT - mu, full_matrices=False); LAT64 = (LAT - mu) @ Vt[:64].T
rng = np.random.default_rng(0)
au_topo = evaluate_features(TOPO, y, np.random.default_rng(0))
au_stat = evaluate_features(STAT, y, np.random.default_rng(0))
au_lat = evaluate_features(LAT64, y, np.random.default_rng(0)) if len(LAT64) == len(y) else float("nan")
au_both = evaluate_features(np.hstack([TOPO, LAT64]), y, np.random.default_rng(0)) if len(LAT64) == len(y) else float("nan")
print(f"    trivial image stats          AUROC {au_stat:.3f}")
print(f"    cubical PH topology (PomSeg)  AUROC {au_topo:.3f}")
print(f"    I-JEPA latent (the baseline)  AUROC {au_lat:.3f}")
print(f"    I-JEPA + topology (does PH add?) AUROC {au_both:.3f}")

# ---------------- (B) are the latent MANIFOLDS topologically distinguishable? ----------------
print("\n(B) Rips PH of the latent clouds -- do edible/poison manifolds differ TOPOLOGICALLY?")
def pca(X, k=15): m = X.mean(0); _, _, V = np.linalg.svd(X - m, full_matrices=False); return (X - m) @ V[:k].T
def h1(P): return ripser(P, maxdim=1, n_perm=min(120, len(P)))["dgms"][1]
E = pca(np.load(f"{ROOT}/{PAIR[0]}.npy")); Po = pca(np.load(f"{ROOT}/{PAIR[1]}.npy"))
rng = np.random.default_rng(0)
dE, dP = h1(E), h1(Po)
cross = bottleneck(dE, dP)
# within-class null: split edible in half, bottleneck between halves
ie = rng.permutation(len(E)); h = len(E) // 2
null = bottleneck(h1(E[ie[:h]]), h1(E[ie[h:]]))
print(f"    bottleneck(edible-H1, poison-H1) = {cross:.3f}  |  within-edible null = {null:.3f}")
print(f"    => {'topology DISTINGUISHES (cross >> null)' if cross > 2 * null else 'NO topological distinction beyond noise'}")
print("\nVERDICT: TDA earns its keep only if image-PH beats trivial stats AND adds over the I-JEPA latent,")
print("or the manifolds are topologically distinguishable. SAFETY: research vehicle only; DON'T EAT.")
