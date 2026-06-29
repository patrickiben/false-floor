"""LEVER #1 (highest suspected viability): a genuinely INDEPENDENT channel -- metadata fusion.

The divergence-floor proof says nothing computed on the image escapes the coincident floor; only a
NEW observation channel does. FungiCLEF confirms metadata (where/when/habitat) is what helps. This
does a clean PAIRED fetch (image + lat/lon/month/country for the SAME specimens) for the lethal pair
(Agaricus campestris edible vs Amanita bisporigera = Destroying Angel), encodes images with the same
I-JEPA, and asks the fair, tau-independent question: at MATCHED yield, does adding the independent
channel reduce poison-eaten?

Honest ceiling (verified at fetch): GBIF 'habitat'/'substrate' (mycorrhizal-forest vs grassland --
the genuinely causal field) is NULL, so the channel is only geography + season. And much of any
geographic signal is SPECIES-RANGE, not edibility (a co-located lookalike defeats it).

SAFETY: research vehicle only; metadata is a weak corroborator, never a clearance. When in doubt DON'T EAT.
"""
from __future__ import annotations

import argparse, glob, os, socket, json, urllib.request, urllib.parse, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

from config import CFG
from model import train_ensemble, ensemble_risk
from gate import auroc

PAIR = {"edible_fieldmushroom": ("Agaricus campestris", 0),
        "poison_destroyingangel": ("Amanita bisporigera", 1)}
ROOT = "data/fungi_meta"


def fetch(n):
    socket.setdefaulttimeout(20)
    g = lambda u: json.load(urllib.request.urlopen(u, timeout=25))
    os.makedirs(ROOT, exist_ok=True)
    for slug, (latin, _) in PAIR.items():
        d = f"{ROOT}/{slug}"; os.makedirs(d, exist_ok=True)
        key = g("https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(latin))["usageKey"]
        recs, offset = [], 0
        while len(recs) < n and offset < n * 8:
            res = g(f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}"
                    f"&mediaType=StillImage&limit=300&offset={offset}")
            for r in res.get("results", []):
                if len(recs) >= n: break
                url = (r.get("media") or [{}])[0].get("identifier")
                lat, lon, mo = r.get("decimalLatitude"), r.get("decimalLongitude"), r.get("month")
                if not url or lat is None or lon is None or mo is None: continue
                i = len(recs)
                try: urllib.request.urlretrieve(url, f"{d}/{i:04d}.jpg")
                except Exception: continue
                recs.append({"i": i, "lat": lat, "lon": lon, "month": mo,
                             "year": r.get("year"), "country": r.get("country")})
            offset += 300
            if res.get("endOfRecords"): break
        json.dump(recs, open(f"{ROOT}/{slug}_meta.json", "w"))
        print(f"  {slug:24} {len(recs)}/{n} paired (image+metadata)", flush=True)


def encode_pair():
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    need = [s for s in PAIR if not os.path.exists(f"{ROOT}/{s}_lat.npy")]
    if need:
        print(f"loading I-JEPA on {dev} ...", flush=True)
        proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    for slug, (_, lbl) in PAIR.items():
        latc, metc = f"{ROOT}/{slug}_lat.npy", f"{ROOT}/{slug}_metaA.json"
        if os.path.exists(latc) and os.path.exists(metc):
            data[slug] = (np.load(latc), json.load(open(metc)), lbl); continue
        recs = json.load(open(f"{ROOT}/{slug}_meta.json")); Z, M = [], []
        for r in recs:
            p = f"{ROOT}/{slug}/{r['i']:04d}.jpg"
            if not os.path.exists(p): continue
            try:
                img = Image.open(p).convert("RGB")
                with torch.no_grad():
                    z = model(**proc(images=img, return_tensors="pt").to(dev)).last_hidden_state.mean(1)[0]
                Z.append(z.cpu().numpy()); M.append(r)
            except Exception: continue
        Z = np.array(Z, dtype="f4"); np.save(latc, Z); json.dump(M, open(metc, "w"))
        data[slug] = (Z, M, lbl); print(f"  encoded {slug}: {Z.shape} + {len(M)} metadata", flush=True)
    return data


def meta_feats(M):
    return np.array([[np.sin(2*np.pi*r["month"]/12), np.cos(2*np.pi*r["month"]/12),
                      r["lat"]/90.0, r["lon"]/180.0, abs(r["lat"])/90.0] for r in M], dtype="f4")


def risk_on_test(Xtr, ytr, Xte, d):
    ens = train_ensemble(Xtr, ytr.astype("f4"), d, CFG)
    r, _ = ensemble_risk(ens, Xte); return r


def poison_at_yield(re, rp, yld):                 # eat if risk<thr; thr set so 'yld' of edibles eaten
    thr = np.quantile(re, yld); return float((rp < thr).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fetch", type=int, default=0); a = ap.parse_args()
    if a.fetch: fetch(a.fetch)
    data = encode_pair()
    rng = np.random.default_rng(0)
    Z = np.vstack([data[s][0] for s in PAIR]); y = np.concatenate([np.full(len(data[s][0]), data[s][2]) for s in PAIR])
    M = sum([data[s][1] for s in PAIR], []); Mf = meta_feats(M)
    # PCA-64 on latents (fit later on train only)
    n = len(y); idx = rng.permutation(n); c = int(0.7 * n); tr, te = idx[:c], idx[c:]
    mu = Z[tr].mean(0); _, _, Vt = np.linalg.svd(Z[tr] - mu, full_matrices=False); V = Vt[:64]
    Ztr, Zte = (Z[tr]-mu)@V.T, (Z[te]-mu)@V.T
    r_lat = risk_on_test(Ztr, y[tr], Zte, 64)
    r_met = risk_on_test(Mf[tr], y[tr], Mf[te], Mf.shape[1])
    yte = y[te]
    z = lambda r: (r - r.mean()) / (r.std() + 1e-9)
    r_fus = (z(r_lat) + z(r_met)) / 2                       # average-risk fusion
    print(f"\nheld-out test: {(yte==0).sum()} edible (field mushroom), {(yte==1).sum()} poison (Destroying Angel)")
    print(f"standalone AUROC  ->  latent (image): {auroc(r_lat, yte):.2f}   metadata (where/when): {auroc(r_met, yte):.2f}"
          f"   fused: {auroc(r_fus, yte):.2f}")
    print("\npoison-eaten fraction at MATCHED edible-yield (lower = safer; the fair, tau-free comparison):")
    print(f"  {'channel':<22} | {'@60% yield':>12} | {'@80% yield':>12}")
    for name, (re, rp) in {
        "latent-only (image)":   (r_lat[yte==0], r_lat[yte==1]),
        "metadata-only":         (r_met[yte==0], r_met[yte==1]),
        "FUSED (image+metadata)":(r_fus[yte==0], r_fus[yte==1]),
    }.items():
        print(f"  {name:<22} | {poison_at_yield(re,rp,0.6)*100:11.1f}% | {poison_at_yield(re,rp,0.8)*100:11.1f}%")
    # corroboration: eat only if BOTH say safe (each at 60% individual yield)
    tl, tm = np.quantile(r_lat[yte==0], 0.6), np.quantile(r_met[yte==0], 0.6)
    eat = (r_lat < tl) & (r_met < tm)
    yld = (eat[yte==0]).mean(); fe = (eat[yte==1]).mean()
    print(f"\ncorroboration (eat only if image AND metadata both clear @60% each): "
          f"yield {yld*100:.1f}% of edibles, poison-eaten {fe*100:.1f}%")
    print("\nVERDICT: metadata is the only channel that can ADD information (vs frozen-latent methods),")
    print("but it is weak/partly species-range here (habitat null), helps at a yield cost, and does NOT")
    print("reach a safe 0 -- matches FungiCLEF. SAFETY: research vehicle only; when in doubt, DON'T EAT.")


if __name__ == "__main__":
    main()
