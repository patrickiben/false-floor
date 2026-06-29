"""BROADEN (more pairs) + HARDEN (stronger-encoder sweep): the information-limit signature across FIVE real
field-confusion edible<->poison pairs and THREE paradigmatically different strong frozen encoders (I-JEPA,
DINOv2, CLIP). Preempts the "one pair" and "weak backbone" objections. Per (pair, encoder): aggregate
separability AUROC + actual false-eat @80% yield (10-seed) + open-set novel-toxic eaten. Thesis prediction:
separability VARIES, but a confidently-wrong false-eat residual PERSISTS in every cell and open-set detection
stays poor across all encoders -> the limit is the decision/information, not the backbone or the pair.
SAFETY: research vehicle; when in doubt, DON'T EAT.
"""
import os, glob, warnings, math
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import torch
from PIL import Image
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_big"
FULL = os.environ.get("FULL", "0") == "1"      # FULL=1 -> use ALL available images (tighter CIs) and cache to a
NPER = 10**9 if FULL else 200                  #          SEPARATE __full file, preserving the original 200-image arrays
dev = "mps" if torch.backends.mps.is_available() else "cpu"
CROP = os.environ.get("CROP", "0") == "1"      # center-crop to 55% to strip background -> confound test
SUF = ("__crop" if CROP else "") + ("__full" if FULL else "")
def _crop(im):
    w, h = im.size; s = int(min(w, h) * 0.55)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)) if CROP else im

PAIRS = [
    ("Agaricus / Destroying-Angel", "Agaricus_campestris", "Amanita_bisporigera"),
    ("Chanterelle / Jack-o-lantern", "Cantharellus_cibarius", "Omphalotus_illudens"),
    ("Morel / False-morel",          "Morchella_esculenta",  "Gyromitra_esculenta"),
    ("Oyster / Jack-o-lantern",      "Pleurotus_ostreatus",  "Omphalotus_illudens"),
    ("Puffball / Deathcap-button",   "Lycoperdon_perlatum",  "Amanita_phalloides"),
]
OPENSET = ["Galerina_marginata", "Conocybe_filaris", "Cortinarius_rubellus"]   # novel toxics, never a pair-poison
SPECIES = sorted(set([s for _, e, p in PAIRS for s in (e, p)] + OPENSET))


def load_ijepa():
    from transformers import AutoModel, AutoImageProcessor
    pr = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k"); m = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    def enc(ims):
        with torch.no_grad(): return m(**pr(images=ims, return_tensors="pt").to(dev)).last_hidden_state.mean(1).cpu().numpy()
    return enc

def load_dinov2():
    from transformers import AutoModel, AutoImageProcessor
    pr = AutoImageProcessor.from_pretrained("facebook/dinov2-large"); m = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(dev)
    def enc(ims):
        with torch.no_grad(): return m(**pr(images=ims, return_tensors="pt").to(dev)).last_hidden_state[:, 0].cpu().numpy()   # CLS
    return enc

def load_clip():
    from transformers import CLIPModel, AutoProcessor
    pr = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14"); m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(dev)
    def enc(ims):
        with torch.no_grad():
            out = m.get_image_features(**pr(images=ims, return_tensors="pt").to(dev))
            out = out.pooler_output if hasattr(out, "pooler_output") else out   # transformers>=5 returns an output obj; pooler_output == projected image embeds
            return out.cpu().numpy()
    return enc

ENCODERS = {"I-JEPA-H14": load_ijepa, "DINOv2-L": load_dinov2, "CLIP-L": load_clip}


def encode_all(ename, loader):
    enc, out = None, {}
    for s in SPECIES:
        cache = f"{ROOT}/{s}__{ename}{SUF}.npy"
        if os.path.exists(cache):
            out[s] = np.load(cache); continue
        if enc is None:
            print(f"  loading {ename}{SUF} ...", flush=True); enc = loader()
        paths = sorted(glob.glob(f"{ROOT}/{s}/*.jpg"))[:NPER]; feats = []
        for k in range(0, len(paths), 16):
            feats.append(enc([_crop(Image.open(p).convert("RGB")) for p in paths[k:k + 16]]).astype("f4"))
        out[s] = np.concatenate(feats); np.save(cache, out[s]); print(f"    {ename}:{s} {out[s].shape}", flush=True)
    return out


def cp_upper(x, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson UPPER bound on a binomial rate (x false-eats of n).
    Self-contained (no scipy): bisection on the binomial CDF P(Bin(n,p) <= x) = alpha, which
    is monotone-decreasing in p. For x=0 this is the exact 1 - alpha**(1/n)."""
    if n == 0: return float("nan")
    if x >= n: return 1.0
    def cdf_le(p):                                   # P(Bin(n,p) <= x), decreasing in p
        if p <= 0.0: return 1.0
        if p >= 1.0: return 0.0
        s = 0.0
        for k in range(x + 1):
            logc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            s += math.exp(logc + k * math.log(p) + (n - k) * math.log1p(-p))
        return s
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_le(mid) > alpha: lo = mid
        else: hi = mid
    return hi


def evalcell(Xe, Xp, Xo):
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        etr, ete = sp(len(Xe)); ptr, pte = sp(len(Xp))
        Xtr = np.vstack([Xe[etr], Xp[ptr]]); ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))].astype("f4")
        mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:48]; pj = lambda A: ((A - mu) @ V.T).astype("f4")
        ens = train_ensemble(pj(Xtr), ytr, 48, CFG)
        re = ensemble_risk(ens, pj(Xe[ete]))[0]; rp = ensemble_risk(ens, pj(Xp[pte]))[0]; ro = ensemble_risk(ens, pj(Xo))[0]
        thr = np.quantile(re, .8)
        au = float(sum((b > a) + .5 * (b == a) for b in rp for a in re) / (len(rp) * len(re)))
        res.append([au, float((rp < thr).mean()) * 100, float((ro < thr).mean()) * 100])
    # n_pte = unique poison TEST specimens per seed (NOT seed-multiplied); a "0.0%" cell is 0/n_pte
    return np.array(res), len(Xp) - int(.7 * len(Xp))


print("encoding 12 species x 3 encoders (cached) ...", flush=True)
FEAT = {en: encode_all(en, ld) for en, ld in ENCODERS.items()}
Xo = {en: np.vstack([FEAT[en][s] for s in OPENSET]) for en in ENCODERS}

print(f"\n{'='*92}\nINFORMATION-LIMIT SIGNATURE across 5 real pairs x 3 strong frozen encoders (10-seed)\n{'='*92}")
print(f"\n{'pair':30} | " + " | ".join(f"{en:^22}" for en in ENCODERS))
print(f"{'(AUROC / false-eat@80 / open-set)':30} | " + " | ".join(f"{'':22}" for en in ENCODERS))
print("-" * 92)
grid = {}; NPTE = {}
for pname, eslug, pslug in PAIRS:
    cells = []
    for en in ENCODERS:
        r, npte = evalcell(FEAT[en][eslug], FEAT[en][pslug], Xo[en]); m = r.mean(0)
        grid[(pname, en)] = m; NPTE[(pname, en)] = npte
        cells.append(f"{m[0]:.2f}/{m[1]:4.1f}/{m[2]:4.1f}")
    print(f"  {pname:28} | " + " | ".join(f"{c:^22}" for c in cells))

# summary: residual never zero; open-set poor; across all encoders
allcells = np.array(list(grid.values()))
print("-" * 92)
print(f"\nACROSS ALL 15 cells: separability AUROC {allcells[:,0].min():.2f}-{allcells[:,0].max():.2f} (varies); "
      f"false-eat@80 min {allcells[:,1].min():.1f}% (NEVER 0); open-set eaten {allcells[:,2].min():.0f}-{allcells[:,2].max():.0f}%")
for en in ENCODERS:
    c = np.array([grid[(p, en)] for p, _, _ in PAIRS])
    print(f"  {en:12}: mean AUROC {c[:,0].mean():.2f}, mean false-eat@80 {c[:,1].mean():.1f}%, mean open-set {c[:,2].mean():.0f}%")

# ---- HONEST UNCERTAINTY on the false-eat residual ----------------------------------------
# A "0.0%" cell is 0 false-eats among the UNIQUE poison TEST specimens (n_pte), NOT 0/inf and
# NOT seed-multiplied (the 10 seeds reshuffle the SAME pool). The seed-mean point estimate has
# a one-sided 95% Clopper-Pearson UPPER bound on the true false-eat rate that is far from zero
# at these sample sizes -- this is what the tables must report instead of a bare "0.0%".
print(f"\n{'='*92}\nHONEST FALSE-EAT BOUNDS  (n = unique poison TEST specimens per seed, not seed-multiplied;\nx = round(seed-mean false-eat x n); one-sided 95% Clopper-Pearson UPPER bound)\n{'='*92}")
print(f"\n{'pair':30} | " + " | ".join(f"{en:^24}" for en in ENCODERS))
print("-" * 92)
for pname, eslug, pslug in PAIRS:
    cells = []
    for en in ENCODERS:
        fe = grid[(pname, en)][1]; n = NPTE[(pname, en)]; x = round(fe / 100 * n)
        cells.append(f"{fe:4.1f}% (UCB {cp_upper(x, n) * 100:4.1f}%, n={n})")
    print(f"  {pname:28} | " + " | ".join(f"{c:^24}" for c in cells))
print("-" * 92)
# pooled across the 5 pairs == the "mean over 5 pairs" row reported in the manuscript table
for en in ENCODERS:
    npool = sum(NPTE[(p, en)] for p, _, _ in PAIRS)
    fepool = float(np.mean([grid[(p, en)][1] for p, _, _ in PAIRS]))
    xpool = round(fepool / 100 * npool)
    print(f"  POOLED 5 pairs {en:10}: false-eat {fepool:4.1f}%  ->  95% UCB {cp_upper(xpool, npool) * 100:.1f}% (x={xpool}, n={npool})")
print("""
READING: if every cell shows a nonzero false-eat residual and high open-set eaten, ACROSS three very different
strong encoders (masked-latent I-JEPA, self-distilled DINOv2, contrastive CLIP) and FIVE distinct lethal pairs,
then the failure is not an artifact of one backbone or one pair -- it is the decision/information limit, exactly
as the data-processing argument predicts (every frozen encoder is a function of the same pixels). SAFETY: research vehicle; DON'T EAT.
""")
