"""JEPA dense FEATURE-FIELD test -- the honest, runnable core of the JEPA + Feature-3DGS proposal,
minus the 3D occluded-part inference (which hallucinates the hidden volva from the visible cap = the
confidently-wrong residual in a 3D costume).

Claim under test (the proposal's central one): mean-pooling I-JEPA's 256 ViT-H/14 patch tokens washes
out the small LOCAL volva; DENSE / part-structured aggregation should preserve it and lower the
confidently-wrong false-eat. We compare aggregations of the patch tokens:
    mean        -- global mean-pool (the baseline we've used all along)
    max         -- preserve the STRONGEST local response (a distinctive volva patch survives)
    meanmaxstd  -- richer pooling (mean + max + std)
    partcluster -- the proposal's ZERO-SHOT part segmentation: k-means on patches -> 'organic' anatomical
                   parts -> pool each part -> classify on the part-structured representation
Multi-seed, matched-yield false-eat on the lethal pair + open-set death cap. Anti-rigging: identical
patches & classifier, only the aggregation differs; multi-seed CIs (the cascade/trained-encoder lesson).
Honest prior (single-modality RGB = redundant channel; trained encoder helped in-category not novelty):
expect modest/no gain on the residual, floor intact. SAFETY: research vehicle; when in doubt, DON'T EAT.
"""
import glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_real"
LE, LP, DC = "edible_fieldmushroom", "poison_destroyingangel", "heldout_deathcap"
_proc = _model = None


def encode_patches(slug):
    cache = f"{ROOT}/{slug}_patches.npy"
    imgs = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
    if os.path.exists(cache) and len(np.load(cache, mmap_mode="r")) == len(imgs):
        return np.load(cache)
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    global _proc, _model
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None:
        print("  loading I-JEPA ViT-H/14 ...", flush=True)
        _proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        _model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    out = []
    for k in range(0, len(imgs), 8):
        batch = [Image.open(p).convert("RGB") for p in imgs[k:k + 8]]
        with torch.no_grad():
            h = _model(**_proc(images=batch, return_tensors="pt").to(dev)).last_hidden_state   # [B,256,1280]
        out.append(h.cpu().numpy().astype("f4"))
    P = np.concatenate(out); np.save(cache, P); print(f"  encoded {slug}: {P.shape}", flush=True)
    return P


print("encoding patch tokens (keeping the dense 16x16 grid, not mean-pooled) ...", flush=True)
PAT = {s: encode_patches(s) for s in (LE, LP, DC)}
print({s: PAT[s].shape for s in PAT}, flush=True)


def kmeans(X, K, iters=12, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (a == k).any(): C[k] = X[a == k].mean(0)
    return C


def agg(P, method, parts=None):
    if method == "mean": return P.mean(1)
    if method == "max": return P.max(1)
    if method == "meanmaxstd": return np.concatenate([P.mean(1), P.max(1), P.std(1)], 1)
    if method == "partcluster":
        F = []
        for i in range(len(P)):
            a = ((P[i][:, None, :] - parts[None]) ** 2).sum(-1).argmin(1)
            F.append(np.concatenate([P[i][a == k].mean(0) if (a == k).any() else np.zeros(P.shape[-1])
                                     for k in range(len(parts))]))
        return np.array(F, dtype="f4")


def pca(fit, *ap, k=64):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False); V = Vt[:min(k, Vt.shape[0])]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *ap)]

def at_yield(re, rp, y): return float((rp < np.quantile(re, y)).mean())


print(f"\ndense vs pooled I-JEPA features -- 10-seed mean +/- sd (matched-yield poison-eaten %)\n")
print(f"  {'aggregation':<26} | {'lethal @60%':>13} | {'lethal @80%':>13} | {'death-cap open-set':>18}")
for meth in ["mean", "max", "meanmaxstd", "partcluster"]:
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        etr, ete = sp(len(PAT[LE])); ptr, pte = sp(len(PAT[LP]))
        parts = None
        if meth == "partcluster":
            samp = np.vstack([PAT[LE][etr].reshape(-1, 1280)[::6], PAT[LP][ptr].reshape(-1, 1280)[::6]])
            parts = kmeans(samp, 6, seed=seed)
        Fe, Fp, Fd = agg(PAT[LE], meth, parts), agg(PAT[LP], meth, parts), agg(PAT[DC], meth, parts)
        Xtr = np.vstack([Fe[etr], Fp[ptr]]); ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))].astype("f4")
        Ptr, Pe, Pp, Pd = pca(Xtr, Fe[ete], Fp[pte], Fd, k=64)
        ens = train_ensemble(Ptr, ytr, Ptr.shape[1], CFG)
        re, rp, rd = ensemble_risk(ens, Pe)[0], ensemble_risk(ens, Pp)[0], ensemble_risk(ens, Pd)[0]
        res.append([at_yield(re, rp, .6) * 100, at_yield(re, rp, .8) * 100,
                    float((rd < np.quantile(re, .6)).mean()) * 100])
    res = np.array(res); m, s = res.mean(0), res.std(0)
    tag = "  [baseline]" if meth == "mean" else ""
    print(f"  {meth:<26} | {m[0]:5.1f} +/- {s[0]:4.1f}% | {m[1]:5.1f} +/- {s[1]:4.1f}% | {m[2]:5.1f} +/- {s[2]:4.1f}%{tag}")

print("\n=> if no dense aggregation beats 'mean' beyond seed noise, the proposal's local-volva-via-patches")
print("   claim doesn't survive on RGB alone (single-modality redundant channel; floor intact). If one")
print("   does, it's a real in-category representation gain (like the trained encoder) -- still not novelty.")
print("SAFETY: research vehicle only; localized features are not edibility. When in doubt, DON'T EAT.")
