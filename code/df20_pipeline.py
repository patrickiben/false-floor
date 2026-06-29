"""Lever #4 wiring stub: run the existing battery on Danish Fungi 2020 with the CAUSAL metadata
(habitat + substrate) that GBIF leaves null. UNTESTED here (needs DF20 downloaded ~100GB); the shape
mirrors metadata_fusion.py / geo_confound.py exactly so it drops in on the Mac.

Usage:
    python df20_pipeline.py --csv DF20_metadata.csv --imgroot /path/to/DF20/images \
                            --toxmap species_toxicity.csv   [--pair Agaricus_campestris Amanita_bisporigera]

CSV expected columns (DF20/FungiCLEF schema; rename as needed):
    image_path, scientificName (or species), habitat, substrate, month, Latitude, Longitude, poisonous
toxmap: scientificName,edible   (edible in {0,1}; or use the DF20 'poisonous' flag directly)

SAFETY: research vehicle only; habitat/substrate is a corroborator, not a clearance. DON'T EAT.
"""
from __future__ import annotations
import argparse, os, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

from config import CFG
from model import train_ensemble, ensemble_risk
from gate import auroc


def load_meta(csv):
    import csv as _csv
    rows = list(_csv.DictReader(open(csv)))
    return rows


def encode_images(paths, imgroot):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    out, keep = [], []
    for k in range(0, len(paths), 16):
        batch, idx = [], []
        for j, p in enumerate(paths[k:k + 16]):
            fp = os.path.join(imgroot, p)
            if os.path.exists(fp):
                batch.append(Image.open(fp).convert("RGB")); idx.append(k + j)
        if not batch: continue
        with torch.no_grad():
            z = model(**proc(images=batch, return_tensors="pt").to(dev)).last_hidden_state.mean(1)
        out.append(z.cpu().numpy()); keep += idx
    return np.concatenate(out).astype("f4"), keep


def meta_feats(rows, habitats, substrates):
    """geography + season + ONE-HOT habitat & substrate (the causal fields GBIF lacked)."""
    feats = []
    for r in rows:
        mo = float(r.get("month") or 6); lat = float(r.get("Latitude") or 0); lon = float(r.get("Longitude") or 0)
        base = [np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12), lat/90, lon/180, abs(lat)/90]
        hot_h = [1.0 if (r.get("habitat") or "") == h else 0.0 for h in habitats]
        hot_s = [1.0 if (r.get("substrate") or "") == s else 0.0 for s in substrates]
        feats.append(base + hot_h + hot_s)
    return np.array(feats, dtype="f4")


def poison_at_yield(re, rp, yld):
    thr = np.quantile(re, yld); return float((rp < thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--imgroot", required=True)
    ap.add_argument("--toxmap", default=""); ap.add_argument("--pair", nargs=2, default=None)
    a = ap.parse_args()

    rows = load_meta(a.csv)
    spcol = "scientificName" if "scientificName" in rows[0] else "species"
    if a.pair:
        want = set(a.pair); rows = [r for r in rows if r[spcol].replace(" ", "_") in want or r[spcol] in want]
    # edibility label: prefer toxmap, else DF20 'poisonous' flag
    tox = {}
    if a.toxmap:
        import csv as _csv
        for r in _csv.DictReader(open(a.toxmap)): tox[r["scientificName"]] = 1 - int(r["edible"])
    def label(r):
        if r[spcol] in tox: return tox[r[spcol]]
        return int(r.get("poisonous") or 0)
    y = np.array([label(r) for r in rows])

    habitats = sorted({r.get("habitat") or "" for r in rows}); substrates = sorted({r.get("substrate") or "" for r in rows})
    print(f"rows {len(rows)} | edible {int((y==0).sum())} poison {int((y==1).sum())} | "
          f"{len(habitats)} habitats, {len(substrates)} substrates")

    paths = [r.get("image_path") or r.get("ImageUniqueID") for r in rows]
    Z, keep = encode_images(paths, a.imgroot)
    rows = [rows[i] for i in keep]; y = y[keep]; Mf = meta_feats(rows, habitats, substrates)

    rng = np.random.default_rng(0); n = len(y); idx = rng.permutation(n); c = int(0.7*n); tr, te = idx[:c], idx[c:]
    mu = Z[tr].mean(0); _, _, Vt = np.linalg.svd(Z[tr]-mu, full_matrices=False); V = Vt[:64]
    Ztr, Zte = (Z[tr]-mu)@V.T, (Z[te]-mu)@V.T
    r_lat = ensemble_risk(train_ensemble(Ztr, y[tr].astype("f4"), 64, CFG), Zte)[0]
    r_met = ensemble_risk(train_ensemble(Mf[tr], y[tr].astype("f4"), Mf.shape[1], CFG), Mf[te])[0]
    yte = y[te]; z = lambda r: (r-r.mean())/(r.std()+1e-9); r_fus = (z(r_lat)+z(r_met))/2
    print(f"\nstandalone AUROC -> latent {auroc(r_lat,yte):.2f}  metadata+habitat/substrate {auroc(r_met,yte):.2f}  fused {auroc(r_fus,yte):.2f}")
    print("poison-eaten @ matched yield:")
    for nm,(re_,rp) in {"latent":(r_lat[yte==0],r_lat[yte==1]),"meta(+habitat/substrate)":(r_met[yte==0],r_met[yte==1]),
                        "fused":(r_fus[yte==0],r_fus[yte==1])}.items():
        print(f"  {nm:26} @60% {poison_at_yield(re_,rp,0.6)*100:4.1f}%  @80% {poison_at_yield(re_,rp,0.8)*100:4.1f}%")
    print("\nKEY TEST: does the habitat/substrate signal SURVIVE tightening to co-occurrence (geo_confound.py")
    print("protocol)? If yes -> a real causal corroborator GBIF couldn't give. Still never a clearance. DON'T EAT.")


if __name__ == "__main__":
    main()
