"""DECISIVE test of the information-limit claim (path B). The curated grid showed strong frozen encoders
(DINOv2/CLIP) ace the lethal pairs to ~0% -- but those pairs are separable when photographed well. Here we
test the GENUINELY-coincident, expert-acknowledged 'you need a microscope/spore-print' confusions:
  Honey-mushroom (Armillaria, E) vs Galerina (deadly)        -- the most common FATAL forage confusion
  Sheathed-woodtuft (Kuehneromyces, E) vs Galerina (deadly)  -- canonical pair separable only by spore ornamentation
  Honey-mushroom (Armillaria, E) vs Sulphur-tuft (Hypholoma, toxic)
  Puffball (Lycoperdon, E) vs Earthball (Scleroderma, toxic)
With an EASY curated pair (Agaricus vs Destroying-Angel) as a positive control. THE TEST: if DINOv2/CLIP fail
HERE (separability << 1, false-eat high) while acing the control, the information limit is REAL but narrow --
confined to the genuinely-feature-absent regime. If they ace these too, the limit is empirically DEAD and the
paper must reframe. We report whatever it shows. SAFETY: research vehicle; DON'T EAT.
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
NPER = 10**9 if FULL else 300                  #          SEPARATE __full file, preserving the original 300-image arrays
dev = "mps" if torch.backends.mps.is_available() else "cpu"
CROP = os.environ.get("CROP", "0") == "1"; SUF = ("__crop" if CROP else "") + ("__full" if FULL else "")
def _crop(im):
    w, h = im.size; s = int(min(w, h) * 0.5)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)) if CROP else im
PAIRS = [
    ("[control] Agaricus / Destroying-Angel", "Agaricus_campestris", "Amanita_bisporigera"),
    ("Honey-mushroom / Galerina (fatal)",     "Armillaria_mellea", "Galerina_marginata"),
    ("Woodtuft / Galerina (microscope)",      "Kuehneromyces_mutabilis", "Galerina_marginata"),
    ("Honey-mushroom / Sulphur-tuft",         "Armillaria_mellea", "Hypholoma_fasciculare"),
    ("Puffball / Earthball",                  "Lycoperdon_perlatum", "Scleroderma_citrinum"),
]


def load_ijepa():
    from transformers import AutoModel, AutoImageProcessor
    pr = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k"); m = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    return lambda ims: m(**pr(images=ims, return_tensors="pt").to(dev)).last_hidden_state.mean(1).detach().cpu().numpy()

def load_dinov2():
    from transformers import AutoModel, AutoImageProcessor
    pr = AutoImageProcessor.from_pretrained("facebook/dinov2-large"); m = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(dev)
    return lambda ims: m(**pr(images=ims, return_tensors="pt").to(dev)).last_hidden_state[:, 0].detach().cpu().numpy()

def load_clip():
    from transformers import CLIPModel, AutoProcessor
    pr = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14"); m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(dev)
    def enc(ims):
        out = m.get_image_features(**pr(images=ims, return_tensors="pt").to(dev))
        out = out.pooler_output if hasattr(out, "pooler_output") else out   # transformers>=5 returns an output obj; pooler_output == projected image embeds
        return out.detach().cpu().numpy()
    return enc

ENCODERS = {"I-JEPA-H14": load_ijepa, "DINOv2-L": load_dinov2, "CLIP-L": load_clip}


def encode_all(ename, loader, species):
    enc, out = None, {}
    for s in species:
        cache = f"{ROOT}/{s}__{ename}{SUF}.npy"
        if os.path.exists(cache): out[s] = np.load(cache); continue
        paths = sorted(glob.glob(f"{ROOT}/{s}/*.jpg"))[:NPER]
        if len(paths) < 40: out[s] = None; continue
        if enc is None: print(f"  loading {ename}{SUF} ...", flush=True); enc = loader()
        with torch.no_grad():
            feats = [enc([_crop(Image.open(p).convert("RGB")) for p in paths[k:k + 16]]).astype("f4") for k in range(0, len(paths), 16)]
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


def evalcell(Xe, Xp):
    if Xe is None or Xp is None: return None, 0
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        etr, ete = sp(len(Xe)); ptr, pte = sp(len(Xp))
        Xtr = np.vstack([Xe[etr], Xp[ptr]]); ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))].astype("f4")
        mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:48]; pj = lambda A: ((A - mu) @ V.T).astype("f4")
        ens = train_ensemble(pj(Xtr), ytr, 48, CFG)
        re = ensemble_risk(ens, pj(Xe[ete]))[0]; rp = ensemble_risk(ens, pj(Xp[pte]))[0]
        au = float(sum((b > a) + .5 * (b == a) for b in rp for a in re) / (len(rp) * len(re)))
        res.append([au, float((rp < np.quantile(re, .8)).mean()) * 100])
    # n_pte = unique poison TEST specimens per seed (NOT seed-multiplied); a "0.0%" cell is 0/n_pte
    return np.array(res).mean(0), len(Xp) - int(.7 * len(Xp))


species = sorted(set(s for _, e, p in PAIRS for s in (e, p)))
print(f"species needed (counts): " + ", ".join(f"{s}={len(glob.glob(f'{ROOT}/{s}/*.jpg'))}" for s in species), flush=True)
FEAT = {en: encode_all(en, ld, species) for en, ld in ENCODERS.items()}

print(f"\n{'='*82}\nGENUINELY-COINCIDENT pairs vs strong encoders (10-seed: AUROC / false-eat@80)\n{'='*82}")
print(f"\n  {'pair':40} | " + " | ".join(f"{en:^16}" for en in ENCODERS))
print("-" * 82)
RES = {}
for pname, e, p in PAIRS:
    cells = []
    for en in ENCODERS:
        m, npte = evalcell(FEAT[en].get(e), FEAT[en].get(p))
        RES[(pname, en)] = (m, npte)
        cells.append("  n/a  " if m is None else f"{m[0]:.2f} / {m[1]:4.1f}%")
    print(f"  {pname:40} | " + " | ".join(f"{c:^16}" for c in cells))
print("-" * 82)

# ---- HONEST UNCERTAINTY on the false-eat residual ----------------------------------------
# A "0.0%" cell is 0 false-eats among the UNIQUE poison TEST specimens (n_pte), NOT 0/inf and
# NOT seed-multiplied (the 10 seeds reshuffle the SAME pool). Each pair's n_pte differs because
# the cached poison arrays differ in size. The one-sided 95% Clopper-Pearson UPPER bound is what
# the tables must report instead of a bare "0.0%".
print(f"\n{'='*100}\nHONEST FALSE-EAT BOUNDS  (n = unique poison TEST specimens per seed, not seed-multiplied;\nx = round(seed-mean false-eat x n); one-sided 95% Clopper-Pearson UPPER bound)\n{'='*100}")
print(f"\n  {'pair':40} | " + " | ".join(f"{en:^24}" for en in ENCODERS))
print("-" * 100)
for pname, e, p in PAIRS:
    cells = []
    for en in ENCODERS:
        m, n = RES[(pname, en)]
        if m is None:
            cells.append("          n/a           ")
        else:
            x = round(m[1] / 100 * n)
            cells.append(f"{m[1]:4.1f}% (UCB {cp_upper(x, n) * 100:4.1f}%, n={n})")
    print(f"  {pname:40} | " + " | ".join(f"{c:^24}" for c in cells))
print("-" * 100)
print("""
READING: the CONTROL pair should be ~1.00/0.0% (encoders work). If the genuinely-coincident pairs (esp.
Honey-mushroom/Galerina and Woodtuft/Galerina) show LOW AUROC and HIGH false-eat for DINOv2/CLIP too, then a
strong encoder cannot separate the truly-undecidable cases -- the information limit is REAL, confined to the
feature-absent regime that curated benchmarks miss. If DINOv2/CLIP ace these as well, the limit is empirically
dead and the paper reframes. SAFETY: research vehicle; topology/latents are not edibility; DON'T EAT.
""")
