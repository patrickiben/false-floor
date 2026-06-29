"""Real I-JEPA evaluation battery. Measures, on REAL fungi latents:
  (1) lookalike-pair separability (incl. the lethal Agaricus-vs-Amanita pair);
  (2) the keeper -- kNN-geometry vs Mahalanobis OOD -- via the standing gate;
  (3) the real SAFETY number -- the actual EAT/DISCARD/INSPECT decision on the lethal pair, with
      the corroboration dial (false-eats vs abstentions);
  (4) open-set -- does a held-out TOXIC species (death cap, never trained) get flagged as OOD so
      the system abstains, instead of mapping it onto a trained edible?

Confound / robustness controls:
  --crop     center-crop before encoding (does separability survive removing background? = is it
             reading the FUNGUS or the FOREST?)
  --augment  blur + occlude the stem base (the volva!) + jitter -> a field-condition proxy
  --fetch N  download N imgs/species from GBIF first

Runs on MPS / CUDA / CPU automatically. SAFETY: latent separability is NOT edibility; clear
GBIF photos are a biased, optimistic sample; research vehicle only -- under any doubt, DON'T EAT.
"""
from __future__ import annotations

import argparse, glob, os, socket, json, urllib.request, urllib.parse, warnings
import numpy as np
warnings.filterwarnings("ignore")

from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import fit, distance, knn_novelty
from gate import auroc, gate
import efe

SPECIES = {
    "edible_chanterelle":     ("Cantharellus cibarius", "edible"),
    "poison_jackolantern":    ("Omphalotus illudens",   "poison"),
    "edible_fieldmushroom":   ("Agaricus campestris",   "edible"),
    "poison_destroyingangel": ("Amanita bisporigera",   "poison"),
    "heldout_morel":          ("Morchella esculenta",   "heldout"),
    "heldout_deathcap":       ("Amanita phalloides",    "heldout_toxic"),   # open-set: never trained
}
PAIRS = [
    ("edible_chanterelle",   "poison_jackolantern",    "chanterelle vs jack-o'-lantern"),
    ("edible_fieldmushroom", "poison_destroyingangel", "field mushroom vs DESTROYING ANGEL (lethal)"),
]
ROOT = "data/fungi_real"


def fetch(n):
    socket.setdefaulttimeout(12)
    g = lambda u: json.load(urllib.request.urlopen(u, timeout=15))
    for slug, (latin, _) in SPECIES.items():
        d = f"{ROOT}/{slug}"; os.makedirs(d, exist_ok=True)
        try:
            key = g("https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(latin))["usageKey"]
            occ = g(f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}&mediaType=StillImage&limit={n*3}")["results"]
        except Exception as e:
            print(f"  {slug}: GBIF error {e}"); continue
        got = 0
        for r in occ:
            if got >= n: break
            url = (r.get("media") or [{}])[0].get("identifier")
            if not url: continue
            try: urllib.request.urlretrieve(url, f"{d}/{got:04d}.jpg"); got += 1
            except Exception: pass
        print(f"  {slug:24} {got}/{n} images")


def _transform(img, mode):
    from PIL import Image, ImageFilter, ImageEnhance
    if mode == "crop":                                   # keep center 55% -> drop most background
        w, h = img.size; cw, ch = int(w * 0.55), int(h * 0.55)
        l, t = (w - cw) // 2, (h - ch) // 2
        return img.crop((l, t, l + cw, t + ch))
    if mode == "volva":                                  # crude stem-base/volva proxy: bottom-center region
        w, h = img.size; cw = int(w * 0.60); l = (w - cw) // 2; t = int(h * 0.45)
        return img.crop((l, t, l + cw, h))               # keep horizontal center 60%, bottom 55%
    if mode == "augment":                                # field-condition proxy
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        img = ImageEnhance.Color(img).enhance(0.7)
        w, h = img.size; box = Image.new("RGB", (int(w * 0.35), int(h * 0.35)), (110, 110, 110))
        img = img.copy(); img.paste(box, (int(w * 0.32), int(h * 0.55)))   # occlude the stem base / volva
        return img
    return img


def encode_all(mode=""):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    suffix = ("_" + mode) if mode else ""
    need = [s for s in SPECIES if glob.glob(f"{ROOT}/{s}/*.jpg")
            and (not os.path.exists(f"{ROOT}/{s}{suffix}.npy")
                 or len(np.load(f"{ROOT}/{s}{suffix}.npy")) != len(glob.glob(f"{ROOT}/{s}/*.jpg")))]
    lat = {}
    if need:
        print(f"loading I-JEPA ViT-H/14 on {dev} (mode={mode or 'raw'}) ...", flush=True)
        proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    for slug in SPECIES:
        cache = f"{ROOT}/{slug}{suffix}.npy"; imgs = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
        if os.path.exists(cache) and len(np.load(cache)) == len(imgs):
            lat[slug] = np.load(cache); continue
        out = []
        for p in imgs:
            try:
                img = _transform(Image.open(p).convert("RGB"), mode)
                with torch.no_grad():
                    z = model(**proc(images=img, return_tensors="pt").to(dev)).last_hidden_state.mean(1)[0]
                out.append(z.cpu().numpy())
            except Exception: pass
        lat[slug] = np.array(out, dtype="f4"); np.save(cache, lat[slug])
        if len(out): print(f"  encoded {slug}{suffix}: {lat[slug].shape}")
    return lat


def pca(Xfit, k):
    mu = Xfit.mean(0); _, _, Vt = np.linalg.svd(Xfit - mu, full_matrices=False)
    return mu, Vt[:min(k, Vt.shape[0])]
def proj(X, mu, V): return (X - mu) @ V.T
def have(lat, s, n=4): return s in lat and len(lat[s]) >= n


def sec_separability(lat, tag=""):
    print(f"\n=== (1) lookalike-pair separability{tag} (the safety question) ===")
    for e, p, name in PAIRS:
        if not (have(lat, e) and have(lat, p)): print(f"  {name}: too few images"); continue
        mu, V = pca(np.vstack([lat[e], lat[p]]), 16); E, P = proj(lat[e], mu, V), proj(lat[p], mu, V)
        ec, pc = E.mean(0), P.mean(0)
        sc = lambda Z: np.linalg.norm(Z - ec, axis=1) - np.linalg.norm(Z - pc, axis=1)
        au = auroc(np.r_[sc(E), sc(P)], np.r_[np.zeros(len(E)), np.ones(len(P))])
        t = "separable" if au > 0.8 else ("partly" if au > 0.65 else "COINCIDENT (floor)")
        print(f"  {name:46} AUROC {au:.2f}  ({t})")


def sec_keeper(lat):
    print("\n=== (2) keeper: kNN-geometry vs Mahalanobis OOD (held-out species), via gate ===")
    tr = [s for s in SPECIES if SPECIES[s][1] in ("edible", "poison") and have(lat, s)]
    ho = [s for s in SPECIES if SPECIES[s][1].startswith("heldout") and have(lat, s)]
    if not (tr and ho): print("  (insufficient data)"); return
    train = np.vstack([lat[s] for s in tr]); muT, VT = pca(train, 16)
    nhold = max(4, len(train) // 6)
    Xtr = proj(train[:-nhold], muT, VT); ind = proj(train[-nhold:], muT, VT); ood = proj(lat[ho[0]], muT, VT)
    y = np.r_[np.zeros(len(ind)), np.ones(len(ood))]; mu2, inv2 = fit(Xtr)
    maha = np.r_[distance(ind, mu2, inv2), distance(ood, mu2, inv2)]
    knn = np.r_[knn_novelty(ind, Xtr, 5), knn_novelty(ood, Xtr, 5)]
    print(f"  held-out={ho[0]}  Mahalanobis AUROC {auroc(maha, y):.2f}  kNN-geometry AUROC {auroc(knn, y):.2f}")
    print("  gate:", gate("kNN-geometry-OOD", knn, [maha], y))


def sec_decision_lethal(lat):
    print("\n=== (3) SAFETY: actual eat/discard/inspect decision on the lethal pair + corroboration dial ===")
    need = ["edible_chanterelle", "poison_jackolantern", "edible_fieldmushroom", "poison_destroyingangel"]
    if not all(have(lat, s) for s in need): print("  (lethal pair not present)"); return
    rng = np.random.default_rng(0)
    def split(X): i = rng.permutation(len(X)); c = int(0.7 * len(X)); return X[i[:c]], X[i[c:]]
    fm_tr, fm_te = split(lat["edible_fieldmushroom"]); da_tr, da_te = split(lat["poison_destroyingangel"])
    Xtr = np.vstack([lat["edible_chanterelle"], lat["poison_jackolantern"], fm_tr, da_tr])
    ytr = np.r_[np.zeros(len(lat["edible_chanterelle"])), np.ones(len(lat["poison_jackolantern"])),
                np.zeros(len(fm_tr)), np.ones(len(da_tr))].astype("f4")
    mu, V = pca(Xtr, 64); Ztr = proj(Xtr, mu, V)
    ens = train_ensemble(Ztr, ytr, 64, CFG)
    muE, invE = fit(Ztr[ytr == 0])
    Zda = proj(da_te, mu, V)                                       # held-out DESTROYING ANGELS (poison)
    r, s = ensemble_risk(ens, Zda)
    dist = distance(Zda, muE, invE); thr = np.percentile(distance(Ztr[ytr == 0], muE, invE), 99)
    novel = dist > thr
    n = len(Zda)
    print(f"  held-out destroying angels (poison) under test: {n}")
    print(f"  {'corroboration (kappa)':>22} | {'EATEN (false-eat)':>17} | {'abstain':>7} | {'discard':>7}")
    for k in (0.0, 1.0, 2.0, 3.0):
        d = efe.decide(r, s, CFG, kappa=k, novel=(novel if k > 0 else None))
        eaten = int((d == efe.EAT).sum()); ab = int((d == efe.INSPECT).sum()); dis = int((d == efe.DISCARD).sum())
        print(f"  {k:>22.1f} | {eaten:>17} | {ab:>7} | {dis:>7}{'   <- DEADLY' if eaten else '   <- none eaten'}")
    print("  (any EATEN here = a poisonous Destroying Angel the system would clear for eating.)")


def sec_openset_toxic(lat):
    print("\n=== (4) open-set: is a held-out TOXIC species (death cap) flagged as unknown? ===")
    tr = [s for s in SPECIES if SPECIES[s][1] in ("edible", "poison") and have(lat, s)]
    if not (tr and have(lat, "heldout_deathcap")): print("  (death cap not fetched -- add via --fetch)"); return
    train = np.vstack([lat[s] for s in tr]); muT, VT = pca(train, 16)
    nhold = max(4, len(train) // 6)
    Xtr = proj(train[:-nhold], muT, VT); ind = proj(train[-nhold:], muT, VT)
    dc = proj(lat["heldout_deathcap"], muT, VT)
    y = np.r_[np.zeros(len(ind)), np.ones(len(dc))]
    knn = np.r_[knn_novelty(ind, Xtr, 5), knn_novelty(dc, Xtr, 5)]
    print(f"  death cap flagged-as-OOD AUROC (kNN-geometry): {auroc(knn, y):.2f}  "
          f"(high = the system knows it's seeing an unknown species and can abstain)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=0)
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--volva", action="store_true")
    a = ap.parse_args()
    if a.fetch: fetch(a.fetch)
    mode = "crop" if a.crop else ("augment" if a.augment else ("volva" if a.volva else ""))
    lat = encode_all(mode)
    for s in SPECIES:
        if have(lat, s, 1): print(f"latents {s}{('_'+mode) if mode else ''}: {lat[s].shape}")
    sec_separability(lat, tag=f" [{mode}]" if mode else "")
    sec_keeper(lat)
    sec_decision_lethal(lat)
    sec_openset_toxic(lat)
    print("\nSAFETY: separability in latent space is NOT edibility; research vehicle only -- when in doubt, DON'T EAT.")


if __name__ == "__main__":
    main()
