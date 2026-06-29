"""Mechanistic ablation (referee request): is the weak encoder's deficit the pretraining OBJECTIVE
(masked-latent I-JEPA vs self-distilled DINOv2) or our AGGREGATION (mean-pool of patch tokens vs the CLS
embedding)? We re-encode three hard pairs keeping PATCH tokens and compare:
  I-JEPA  {mean-pool, max-pool}          (no CLS token exists)
  DINOv2  {CLS, mean-pool, max-pool}
If DINOv2-mean-pool stays strong while I-JEPA-{mean,max} stay weak, the deficit is the OBJECTIVE, not the
aggregation -- i.e. mean-pooling did not cause the false limit; the masked-latent features did. SAFETY: research vehicle; DON'T EAT.
"""
import os, glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import torch
from PIL import Image
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_big"; NPER = 250
dev = "mps" if torch.backends.mps.is_available() else "cpu"
PAIRS = [
    ("recluse / house-spider", "Kukulcania_hibernalis", "Loxosceles_reclusa"),
    ("honey-mushroom / Galerina", "Armillaria_mellea", "Galerina_marginata"),
    ("Agaricus / Destroying-Angel", "Agaricus_campestris", "Amanita_bisporigera"),
]
CONFIGS = ["IJ_mean", "IJ_max", "DN_cls", "DN_mean", "DN_max"]
SPECIES = sorted(set(s for _, e, p in PAIRS for s in (e, p)))


def encode_aggs(slug):
    if all(os.path.exists(f"{ROOT}/{slug}__{k}.npy") for k in CONFIGS):
        return {k: np.load(f"{ROOT}/{slug}__{k}.npy") for k in CONFIGS}
    from transformers import AutoModel, AutoImageProcessor
    global _ij, _ijp, _dn, _dnp
    if "_ij" not in globals():
        _ijp = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k"); _ij = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
        _dnp = AutoImageProcessor.from_pretrained("facebook/dinov2-large"); _dn = AutoModel.from_pretrained("facebook/dinov2-large").eval().to(dev)
    paths = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))[:NPER]
    acc = {k: [] for k in CONFIGS}
    for j in range(0, len(paths), 16):
        ims = []
        for p in paths[j:j + 16]:
            try: ims.append(Image.open(p).convert("RGB"))
            except Exception: pass
        if not ims: continue
        with torch.no_grad():
            h = _ij(**_ijp(images=ims, return_tensors="pt").to(dev)).last_hidden_state          # [B,256,1280] all patches
            acc["IJ_mean"].append(h.mean(1).cpu().numpy()); acc["IJ_max"].append(h.max(1).values.cpu().numpy())
            g = _dn(**_dnp(images=ims, return_tensors="pt").to(dev)).last_hidden_state            # [B,257,1024] CLS + patches
            acc["DN_cls"].append(g[:, 0].cpu().numpy())
            acc["DN_mean"].append(g[:, 1:].mean(1).cpu().numpy()); acc["DN_max"].append(g[:, 1:].max(1).values.cpu().numpy())
    out = {k: np.concatenate(v).astype("f4") for k, v in acc.items()}
    for k in CONFIGS: np.save(f"{ROOT}/{slug}__{k}.npy", out[k])
    print(f"  encoded {slug}: {out['IJ_mean'].shape[0]} imgs", flush=True)
    return out


def evalcell(Xe, Xp):
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
    return np.array(res).mean(0)


print("encoding (keeping patch tokens; both encoders) ...", flush=True)
FEAT = {s: encode_aggs(s) for s in SPECIES}

print(f"\n{'='*92}\nOBJECTIVE vs AGGREGATION ablation (AUROC / false-safe@80, 10-seed)\n{'='*92}")
print(f"\n  {'pair':28} | " + " | ".join(f"{c:^13}" for c in CONFIGS))
print("-" * 92)
for pname, e, p in PAIRS:
    cells = [f"{(m:=evalcell(FEAT[e][c], FEAT[p][c]))[0]:.2f}/{m[1]:4.1f}%" for c in CONFIGS]
    print(f"  {pname:28} | " + " | ".join(f"{c:^13}" for c in cells))
print("-" * 92)
print("""
READING: compare DN_cls vs DN_mean (does DINOv2 stay strong WITHOUT the CLS token, under mean-pooling?) and
IJ_mean vs IJ_max (does a different I-JEPA aggregation rescue it?). If DN_mean ~= DN_cls (strong) and IJ_max ~=
IJ_mean (weak), the deficit is the OBJECTIVE (masked-latent vs self-distilled), NOT our mean-pooling -- the
'weak baseline' was the pretraining choice, and the aggregation is a red herring. SAFETY: research vehicle; DON'T EAT.
""")
