"""CONTROL for the big-train result: FROZEN I-JEPA on the SAME new scraped data, MULTI-SEED.

The trained ResNet-50 got lethal-pair false-eat 0.7%/2.7% and open-set 1.6% -- apparently far better than
the old frozen-I-JEPA ~5%. But that compared trained-ResNet-on-NEW-data to frozen-I-JEPA-on-OLD-data (3
confounded changes). This isolates the variable: run FROZEN I-JEPA on the SAME new images, same splits,
multi-seed. If FROZEN ~= TRAINED, the gain was DATA (more/cleaner images), not training; if TRAINED << FROZEN,
training genuinely helped the representation. SAFETY: research vehicle; when in doubt, DON'T EAT.
"""
import glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_big"
SPECIES = {
    "Agaricus campestris": (0, "train"), "Agaricus arvensis": (0, "train"), "Cantharellus cibarius": (0, "train"),
    "Morchella esculenta": (0, "train"), "Pleurotus ostreatus": (0, "train"), "Macrolepiota procera": (0, "train"),
    "Hydnum repandum": (0, "train"), "Lycoperdon perlatum": (0, "train"),
    "Amanita phalloides": (1, "train"), "Amanita bisporigera": (1, "train"), "Amanita muscaria": (1, "train"),
    "Galerina marginata": (1, "train"), "Omphalotus illudens": (1, "train"), "Clitocybe dealbata": (1, "train"),
    "Amanita virosa": (1, "openset"), "Conocybe filaris": (1, "openset"),
}
LETHAL_E, LETHAL_P = "Agaricus_campestris", "Amanita_bisporigera"


def encode_all():
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = model = None
    for latin in SPECIES:
        slug = latin.replace(" ", "_"); cache = f"{ROOT}/{slug}_ijepa.npy"
        imgs = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
        if not imgs: continue
        if os.path.exists(cache) and len(np.load(cache)) == len(imgs): continue
        if model is None:
            print("loading I-JEPA ViT-H/14 ...", flush=True)
            proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
            model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
        out = []
        for k in range(0, len(imgs), 16):
            batch = [Image.open(p).convert("RGB") for p in imgs[k:k + 16]]
            with torch.no_grad():
                z = model(**proc(images=batch, return_tensors="pt").to(dev)).last_hidden_state.mean(1)
            out.append(z.cpu().numpy())
        np.save(cache, np.concatenate(out).astype("f4")); print(f"  encoded {slug}: {len(imgs)}", flush=True)


encode_all()
LAT = {l.replace(" ", "_"): np.load(f"{ROOT}/{l.replace(' ', '_')}_ijepa.npy")
       for l in SPECIES if os.path.exists(f"{ROOT}/{l.replace(' ', '_')}_ijepa.npy")}


def at_yield(re, rp, y): return float((rp < np.quantile(re, y)).mean())

res = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    Xtr, ytr, lte, ltp, osx = [], [], [], [], []
    for latin, (edib, role) in SPECIES.items():
        slug = latin.replace(" ", "_")
        if slug not in LAT: continue
        X = LAT[slug]
        if role == "openset": osx.append(X); continue
        i = rng.permutation(len(X)); c = int(0.7 * len(X))
        Xtr.append(X[i[:c]]); ytr.append(np.full(c, edib))
        if slug == LETHAL_E: lte.append(X[i[c:]])
        if slug == LETHAL_P: ltp.append(X[i[c:]])
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr).astype("f4")
    mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:64]
    proj = lambda A: ((A - mu) @ V.T).astype("f4")
    ens = train_ensemble(proj(Xtr), ytr, 64, CFG)
    re = ensemble_risk(ens, proj(np.vstack(lte)))[0]; rp = ensemble_risk(ens, proj(np.vstack(ltp)))[0]
    osr = ensemble_risk(ens, proj(np.vstack(osx)))[0]
    res.append([at_yield(re, rp, .6) * 100, at_yield(re, rp, .8) * 100, float((osr < np.quantile(re, .6)).mean()) * 100])

res = np.array(res); m, s = res.mean(0), res.std(0)
print("\n=== FROZEN I-JEPA on the SAME new data, 10-seed mean +/- sd (the control) ===")
print(f"  lethal-pair poison-eaten @60%: {m[0]:.1f} +/- {s[0]:.1f}%   @80%: {m[1]:.1f} +/- {s[1]:.1f}%")
print(f"  open-set deadly eaten (at lethal 60%-yield thr): {m[2]:.1f} +/- {s[2]:.1f}%")
print(f"\n  TRAINED ResNet-50 (single seed): lethal 0.7% / 2.7%, open-set 1.6%")
print("  => if FROZEN ~= TRAINED, the gain was DATA not training; if TRAINED << FROZEN, training helped.")
print("  (open-set caveat either way: A. virosa is a SISTER Amanita to 5 trained Amanitas = in-genus, not novel.)")
