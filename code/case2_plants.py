"""SECOND CASE STUDY test: does the representation-invariance protocol (§5.1) reproduce on a NON-FUNGAL lethal
decision? The classic deadly Apiaceae confusions (poison/water hemlock vs wild carrot), I-JEPA vs DINOv2 vs
CLIP, with honest Clopper-Pearson upper bounds. Reported whichever way it breaks: a weak->strong gap = the
artifact pattern generalizes; both encoders failing = a positive control (the protocol certifying a GENUINE
limit). SAFETY: research vehicle; DON'T EAT.
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
PAIRS = [   # (name, EDIBLE slug, POISON slug)
    ("Poison-hemlock / Wild-carrot", "Daucus_carota", "Conium_maculatum"),
    ("Water-hemlock / Wild-carrot",  "Daucus_carota", "Cicuta_maculata"),
    ("Poison-hemlock / Cow-parsley", "Anthriscus_sylvestris", "Conium_maculatum"),
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
        with torch.no_grad():
            feats = [enc([Image.open(p).convert("RGB") for p in paths[k:k + 16]]).astype("f4") for k in range(0, len(paths), 16)]
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
    return np.array(res).mean(0), len(Xp) - int(.7 * len(Xp))


species = sorted(set(s for _, e, p in PAIRS for s in (e, p)))
print(f"counts: " + ", ".join(f"{s}={len(glob.glob(f'{ROOT}/{s}/*.jpg'))}" for s in species), flush=True)
FEAT = {en: encode_all(en, ld, species) for en, ld in ENCODERS.items()}

print(f"\n{'='*98}\nCASE 2 (deadly plants): the same protocol on a non-fungal decision (10-seed AUROC / false-eat@80 / 95% UCB)\n{'='*98}")
print(f"\n  {'pair':32} | " + " | ".join(f"{en:^22}" for en in ENCODERS))
print("-" * 98)
for pname, e, p in PAIRS:
    cells = []
    for en in ENCODERS:
        m, npte = evalcell(FEAT[en].get(e), FEAT[en].get(p))
        if m is None: cells.append("       n/a       "); continue
        x = round(m[1] / 100 * npte); ucb = cp_upper(x, npte) * 100
        cells.append(f"{m[0]:.2f} / {m[1]:4.1f}% (UCB {ucb:4.1f},n={npte})")
    print(f"  {pname:32} | " + " | ".join(f"{c:^22}" for c in cells))
print("-" * 98)
print("""
READING: compare the WEAK (I-JEPA) vs STRONG (DINOv2/CLIP) false-eat WITHIN each pair. If DINOv2 >> I-JEPA, the
weak-baseline-artifact pattern reproduces on a second, non-fungal lethal decision (n=1 -> n=2 across kingdoms,
the protocol generalizes). If BOTH fail similarly, the representation-invariance test is certifying a GENUINE
limit (a positive control for the protocol). SAFETY: research vehicle; when in doubt, DON'T EAT.
""")
