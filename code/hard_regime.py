"""The decisive scoping test. The 5-pair x 3-encoder grid showed strong frozen encoders (DINOv2/CLIP) solve
the lethal pairs to ~0% false-eat ON CURATED PHOTOS (survives cropping) -- correcting our over-strong "no
encoder helps." So the question is no longer "can a model separate these?" (on good photos, yes) but "does
that success survive the regime where foraging deaths actually happen?" We test the genuine lookalike pairs
with the STRONGEST encoder (DINOv2) under:
  (A) CURATED        -- the benchmark regime (baseline).
  (B) FIELD-DEGRADED -- downsample + blur + BASE OCCLUSION (remove the bottom third = the un-excavated volva,
                        the #1 field failure) + color jitter. The diagnostic info is genuinely removed.
  (C) LOOKALIKE OPEN-SET -- novel white toxins held out entirely (A. phalloides, A. virosa, Clitocybe dealbata)
                        presented to the Agaricus/Amanita-bisporigera classifier: do they get eaten?
Prediction (scoped thesis): curated ~0% -> field-degraded residual RETURNS and open-set eaten is high, even for
DINOv2 -- the floor bites where the photo lacks the feature, which benchmarks don't test. SAFETY: research vehicle; DON'T EAT.
"""
import os, glob, warnings, math
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import torch
from PIL import Image, ImageFilter
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_big"
FULL = os.environ.get("FULL", "0") == "1"      # FULL=1 -> all available images; SEPARATE __full cache (originals preserved)
NPER = 10**9 if FULL else 200
SUF = "__full" if FULL else ""
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def cp_upper(x, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson UPPER bound on a binomial rate (x of n). No scipy."""
    if n == 0: return float("nan")
    if x >= n: return 1.0
    def cdf_le(p):
        if p <= 0.0: return 1.0
        if p >= 1.0: return 0.0
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                            + k * math.log(p) + (n - k) * math.log1p(-p)) for k in range(x + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_le(mid) > alpha: lo = mid
        else: hi = mid
    return hi
PAIRS = [("Agaricus / Destroying-Angel", "Agaricus_campestris", "Amanita_bisporigera"),
         ("Puffball / Deathcap-button",  "Lycoperdon_perlatum",  "Amanita_phalloides")]
LOOKALIKE_OPENSET = ["Amanita_virosa", "Clitocybe_dealbata", "Amanita_muscaria"]   # novel white/pale toxins
SPECIES = sorted(set([s for _, e, p in PAIRS for s in (e, p)] + LOOKALIKE_OPENSET))


def degrade(im, rng):
    im = im.resize((56, 56)).resize((224, 224), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2.0))  # res + blur
    a = np.array(im); a[150:, :, :] = 110          # occlude bottom third = un-excavated base / hidden volva
    a = np.clip(a.astype("f4") * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255).astype("uint8")  # light/color
    return Image.fromarray(a)


def load_dinov2():
    from transformers import AutoModel, AutoImageProcessor
    pr = AutoImageProcessor.from_pretrained("facebook/dinov2-large"); m = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(dev)
    def enc(ims):
        with torch.no_grad(): return m(**pr(images=ims, return_tensors="pt").to(dev)).last_hidden_state[:, 0].cpu().numpy()
    return enc


def encode(slug, mode, enc, seed=0):
    cache = f"{ROOT}/{slug}__DINOv2-L__{mode}{SUF}.npy"
    if os.path.exists(cache): return np.load(cache)
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))[:NPER]; feats = []
    for k in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB") for p in paths[k:k + 16]]
        if mode == "degraded": ims = [degrade(im, rng) for im in ims]
        feats.append(enc(ims).astype("f4"))
    out = np.concatenate(feats); np.save(cache, out); return out


def evalcell(Xe, Xp, Xo=None):
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        etr, ete = sp(len(Xe)); ptr, pte = sp(len(Xp))
        Xtr = np.vstack([Xe[etr], Xp[ptr]]); ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))].astype("f4")
        mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:48]; pj = lambda A: ((A - mu) @ V.T).astype("f4")
        ens = train_ensemble(pj(Xtr), ytr, 48, CFG)
        re = ensemble_risk(ens, pj(Xe[ete]))[0]; rp = ensemble_risk(ens, pj(Xp[pte]))[0]
        thr = np.quantile(re, .8)
        au = float(sum((b > a) + .5 * (b == a) for b in rp for a in re) / (len(rp) * len(re)))
        oe = float((ensemble_risk(ens, pj(Xo))[0] < thr).mean()) * 100 if Xo is not None else np.nan
        res.append([au, float((rp < thr).mean()) * 100, oe])
    # also return n = unique poison TEST specimens/seed, and n_open = open-set size (both NOT seed-multiplied)
    return np.array(res).mean(0), len(Xp) - int(.7 * len(Xp)), (len(Xo) if Xo is not None else 0)


print("encoding DINOv2 (curated cached + field-degraded) ...", flush=True)
enc = load_dinov2()
CUR = {s: encode(s, "curated", enc) for s in SPECIES}
DEG = {s: encode(s, "degraded", enc) for s in SPECIES}
Xo = np.vstack([CUR[s] for s in LOOKALIKE_OPENSET])

print(f"\n{'='*100}\nDINOv2-L: does curated success survive the field / open-set regime?  (FE@80 with one-sided 95%\nClopper-Pearson UPPER bound; n = unique poison TEST specimens/seed, not seed-multiplied)\n{'='*100}")
print(f"\n  {'pair':26} | {'CURATED (AUROC / FE@80)':>30} | {'FIELD-DEGRADED (AUROC / FE@80)':>32}")
for pname, e, p in PAIRS:
    c, nc, _ = evalcell(CUR[e], CUR[p]); d, nd, _ = evalcell(DEG[e], DEG[p])
    uc = cp_upper(round(c[1] / 100 * nc), nc) * 100; ud = cp_upper(round(d[1] / 100 * nd), nd) * 100
    print(f"  {pname:26} | {c[0]:.2f} / {c[1]:4.1f}% (UCB {uc:4.1f}%, n={nc}) | {d[0]:.2f} / {d[1]:4.1f}% (UCB {ud:4.1f}%, n={nd})")

# lookalike open-set on the curated Agaricus/Amanita-bisporigera classifier
o, no, nopen = evalcell(CUR["Agaricus_campestris"], CUR["Amanita_bisporigera"], Xo)
uo = cp_upper(round(o[2] / 100 * nopen), nopen) * 100
print(f"\n  LOOKALIKE open-set (novel white toxins A.virosa/Clitocybe/A.muscaria) eaten by the")
print(f"  curated Agaricus/Destroying-Angel classifier: {o[2]:.0f}% (95% UCB {uo:.1f}%, n={nopen})  (curated in-dist FE was {o[1]:.1f}%, n={no})")
print("""
READING: if the curated ~0% false-eat RETURNS under field degradation (base occluded) and the lookalike
open-set is eaten at a high rate, then the strongest encoder's benchmark success does NOT transfer to the
regime where deaths occur. The floor is real but regime-specific: it bites where the photo lacks the feature.
Benchmark saturation hides field danger -- the sharper, honest thesis. SAFETY: research vehicle; DON'T EAT.
""")
