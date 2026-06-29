"""THIRD case study (EXTERNAL claim): coral-snake mimicry. Biology calls these near-indistinguishable
(Pfennig et al., Nature 2001). Does a strong frozen encoder separate the venomous coral snake (Micrurus) from
its harmless Lampropeltis/Cemophora mimics? 'false-safe' = a venomous coral cleared as if it were the harmless
mimic (the lethal error). Same three-encoder representation-invariance protocol + Clopper-Pearson bounds.
SAFETY: research vehicle; never a field guide; if unsure, leave the snake alone.
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
PAIRS = [   # (name, HARMLESS-mimic slug, VENOMOUS slug)
    ("Coral / Scarlet-kingsnake", "Lampropeltis_elapsoides", "Micrurus_fulvius"),
    ("Coral / Milk-snake",        "Lampropeltis_triangulum", "Micrurus_fulvius"),
    ("Coral / Scarlet-snake",     "Cemophora_coccinea",      "Micrurus_fulvius"),
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


def evalcell(Xs, Xv):   # Xs = harmless-mimic (safe), Xv = venomous
    if Xs is None or Xv is None: return None, 0
    res = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        str_, ste = sp(len(Xs)); vtr, vte = sp(len(Xv))
        Xtr = np.vstack([Xs[str_], Xv[vtr]]); ytr = np.r_[np.zeros(len(str_)), np.ones(len(vtr))].astype("f4")
        mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:48]; pj = lambda A: ((A - mu) @ V.T).astype("f4")
        ens = train_ensemble(pj(Xtr), ytr, 48, CFG)
        rs = ensemble_risk(ens, pj(Xs[ste]))[0]; rv = ensemble_risk(ens, pj(Xv[vte]))[0]
        au = float(sum((b > a) + .5 * (b == a) for b in rv for a in rs) / (len(rv) * len(rs)))
        res.append([au, float((rv < np.quantile(rs, .8)).mean()) * 100])   # venomous cleared as safe @80% mimic-yield
    return np.array(res).mean(0), len(Xv) - int(.7 * len(Xv))


species = sorted(set(s for _, e, p in PAIRS for s in (e, p)))
print(f"counts: " + ", ".join(f"{s}={len(glob.glob(f'{ROOT}/{s}/*.jpg'))}" for s in species), flush=True)
FEAT = {en: encode_all(en, ld, species) for en, ld in ENCODERS.items()}

print(f"\n{'='*98}\nCASE 3 (coral-snake mimicry, EXTERNAL claim): 10-seed AUROC / false-SAFE@80 / 95% UCB\n{'='*98}")
print(f"\n  {'pair':30} | " + " | ".join(f"{en:^22}" for en in ENCODERS))
print("-" * 98)
for pname, e, p in PAIRS:
    cells = []
    for en in ENCODERS:
        m, nv = evalcell(FEAT[en].get(e), FEAT[en].get(p))
        if m is None: cells.append("       n/a       "); continue
        x = round(m[1] / 100 * nv); ucb = cp_upper(x, nv) * 100
        cells.append(f"{m[0]:.2f} / {m[1]:4.1f}% (UCB {ucb:4.1f},n={nv})")
    print(f"  {pname:30} | " + " | ".join(f"{c:^22}" for c in cells))
print("-" * 98)
print("""
READING: an EXTERNAL, peer-reviewed claim (Pfennig et al., Nature 2001) holds coral snakes and their mimics to
be near-indistinguishable look-alikes. WEAK vs STRONG encoder WITHIN each pair: if DINOv2 >> I-JEPA, a strong
frozen encoder separates what biology calls near-identical -- the representation-invariance test re-examines an
external limit claim and finds it does not bind a strong model (n=2 own + 1 external). If both fail, it is a
genuine limit (positive control). SAFETY: research vehicle; if unsure, leave the snake alone.
""")
