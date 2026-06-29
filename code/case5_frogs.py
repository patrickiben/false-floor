"""CASE 5 (EXTERNAL claim): poison-frog mimicry. Two documented Nature claims of near-identical look-alikes:
  Mullerian -- the mimic poison frog Ranitomeya imitator co-evolved to look near-identical to its model species
    R. variabilis / R. fantastica (Symula et al., Nature 2001). Both toxic: a pure mimic-vs-model separability test.
  Batesian  -- the NON-toxic Allobates zaparo mimics the TOXIC Ameerega (Darst & Cummings, Nature 2006): a genuine
    'is it poisonous?' decision, 'false-safe' = a toxic model cleared as the harmless mimic.
Same three-encoder representation-invariance protocol + Clopper-Pearson bounds; pairs with <40 images are skipped.
Amphibia. SAFETY: research vehicle; never handle wild poison frogs.
"""
import os, glob, warnings, math
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import torch
from PIL import Image
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_big"; NPER = 300
dev = "mps" if torch.backends.mps.is_available() else "cpu"
PAIRS = [   # (name, side-A slug, side-B slug); B-cleared@80%-A-yield reported
    ("Mullerian: variabilis / imitator", "Ranitomeya_variabilis", "Ranitomeya_imitator"),
    ("Mullerian: fantastica / imitator", "Ranitomeya_fantastica", "Ranitomeya_imitator"),
    ("Batesian: Allobates / Ameerega-bilinguis", "Allobates_zaparo", "Ameerega_bilinguis"),
    ("Batesian: Allobates / Ameerega-parvula",   "Allobates_zaparo", "Ameerega_parvula"),
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
        return (out.pooler_output if hasattr(out, "pooler_output") else out).detach().cpu().numpy()
    return enc

ENCODERS = {"I-JEPA-H14": load_ijepa, "DINOv2-L": load_dinov2, "CLIP-L": load_clip}


def encode_all(ename, loader, species):
    enc, out = None, {}
    for s in species:
        cache = f"{ROOT}/{s}__{ename}.npy"
        if os.path.exists(cache): out[s] = np.load(cache); continue
        paths = sorted(glob.glob(f"{ROOT}/{s}/*.jpg"))[:NPER]
        if len(paths) < 40: out[s] = None; continue
        if enc is None: print(f"  loading {ename} ...", flush=True); enc = loader()
        ims = []
        for p in paths:
            try: ims.append(Image.open(p).convert("RGB"))
            except Exception: pass
        with torch.no_grad():
            feats = [enc(ims[k:k + 16]).astype("f4") for k in range(0, len(ims), 16)]
        if not feats: out[s] = None; continue
        out[s] = np.concatenate(feats); np.save(cache, out[s]); print(f"    {ename}:{s} {out[s].shape}", flush=True)
    return out


def cp_upper(x, n, alpha=0.05):
    if n == 0: return float("nan")
    if x >= n: return 1.0
    def cdf_le(p):
        if p <= 0.0: return 1.0
        if p >= 1.0: return 0.0
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)) for k in range(x + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_le(mid) > alpha: lo = mid
        else: hi = mid
    return hi


def evalcell(Xa, Xb):
    if Xa is None or Xb is None: return None, 0
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        a_tr, a_te = sp(len(Xa)); b_tr, b_te = sp(len(Xb))
        Xtr = np.vstack([Xa[a_tr], Xb[b_tr]]); ytr = np.r_[np.zeros(len(a_tr)), np.ones(len(b_tr))].astype("f4")
        mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:48]; pj = lambda A: ((A - mu) @ V.T).astype("f4")
        ens = train_ensemble(pj(Xtr), ytr, 48, CFG)
        ra = ensemble_risk(ens, pj(Xa[a_te]))[0]; rb = ensemble_risk(ens, pj(Xb[b_te]))[0]
        au = float(sum((x > y) + .5 * (x == y) for x in rb for y in ra) / (len(rb) * len(ra)))
        res.append([au, float((rb < np.quantile(ra, .8)).mean()) * 100])
    return np.array(res).mean(0), len(Xb) - int(.7 * len(Xb))


species = sorted(set(s for _, a, b in PAIRS for s in (a, b)))
print(f"counts: " + ", ".join(f"{s}={len(glob.glob(f'{ROOT}/{s}/*.jpg'))}" for s in species), flush=True)
FEAT = {en: encode_all(en, ld, species) for en, ld in ENCODERS.items()}

print(f"\n{'='*100}\nCASE 5 (poison-frog mimicry, EXTERNAL claim): 10-seed AUROC / B-cleared-as-A@80 / 95% UCB\n{'='*100}")
print(f"\n  {'pair':42} | " + " | ".join(f"{en:^18}" for en in ENCODERS))
print("-" * 100)
for pname, a, b in PAIRS:
    cells = []
    for en in ENCODERS:
        m, nb = evalcell(FEAT[en].get(a), FEAT[en].get(b))
        if m is None: cells.append("  n/a (few imgs)  "); continue
        x = round(m[1] / 100 * nb); ucb = cp_upper(x, nb) * 100
        cells.append(f"{m[0]:.2f}/{m[1]:4.1f}%(UCB{ucb:4.1f},n={nb})")
    print(f"  {pname:42} | " + " | ".join(f"{c:^18}" for c in cells))
print("-" * 100)
print("READING: external claim = these mimic/model frogs are near-identical (Symula 2001; Darst & Cummings 2006). If")
print("DINOv2 >> I-JEPA, a strong frozen encoder separates them -> the external near-identical claim does not bind a strong model.")
