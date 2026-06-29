"""COUNTER-TEST (adversarial reviewer): is the cascade negative a STRAWMAN?

The prototype's "localization" is ONE fixed bottom-center box (_volva: center-60%-wide x
bottom-55%, real_ijepa_test.py:71-73) and ONE fixed center box (_crop: center 55%, line 67-70).
A fixed box misses the destroying-angel volva whenever the mushroom is rotated / off-center /
top-down. That is a WEAK localizer. If localization truly cannot help, it must fail even with a
STRONG, still-feasible localizer built from the same raw jpgs + the same I-JEPA encoder.

We build three stronger localizers and test each at MATCHED YIELD vs GLOBAL, reusing the
prototype's EXACT ensemble + yield machinery (model.train_ensemble / at_yield):

  S1  MULTI-REGION grid: 9 overlapping crops at 2 scales covering the whole frame (so the base
      is captured wherever it sits). Per-region toxic classifier; decision = MAX toxic risk over
      regions ("if ANY part screams volva, don't eat") -- the part-detector ideal.
  S2  OTSU FOREGROUND: segment the mushroom from background (cv2 Otsu+largest-blob), encode the
      masked foreground -> embedding not diluted by leaf-litter. Fuse with global.
  S3  EVERYTHING fused: global + all region feats + foreground feats, one classifier -- the
      strongest fair concatenation shot localization can get.

Cached to data/fungi_real/strawman_*.npy so it only re-encodes once.
SAFETY: research vehicle, never foraging advice.
"""
import os, glob, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import cv2
from PIL import Image
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_real"
E, P = "edible_fieldmushroom", "poison_destroyingangel"

# ---------- localizer image transforms ----------
def grid_regions(img):
    """9 crops: 3x3 overlapping grid at 0.5 scale + 1 bottom-strip + 1 center -- covers base
    wherever it lands (off-center / rotated). Returns list of PIL crops."""
    w, h = img.size
    outs = []
    cw, ch = int(w * 0.5), int(h * 0.5)
    for fy in (0.0, 0.25, 0.5):
        for fx in (0.0, 0.25, 0.5):
            l, t = int(w * fx), int(h * fy)
            outs.append(img.crop((l, t, min(l + cw, w), min(t + ch, h))))
    # plus a wide bottom strip (base band, scale-robust) and the whole frame
    outs.append(img.crop((0, int(h * 0.55), w, h)))
    return outs

def otsu_foreground(img):
    """Segment largest connected foreground blob (Otsu on saturation+value), paste onto neutral
    gray so the encoder sees the fungus, not the background."""
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    # foreground tends to be either saturated OR bright vs background litter; combine S and V
    s = hsv[..., 1]; v = hsv[..., 2]
    feat = cv2.normalize(s.astype("f4") * 0.5 + (255 - v).astype("f4") * 0.0 + v.astype("f4") * 0.5,
                         None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    feat = cv2.GaussianBlur(feat, (0, 0), 3)
    _, m = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pick the polarity whose blob is more central (mushroom is usually centered)
    def central_score(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) < 50: return -1
        cy, cx = ys.mean() / mask.shape[0], xs.mean() / mask.shape[1]
        return -((cy - 0.5) ** 2 + (cx - 0.5) ** 2)
    if central_score(255 - m) > central_score(m):
        m = 255 - m
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype("uint8"), 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype("uint8") * 255
    m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB) > 0
    out = np.where(m3, arr, 128).astype("uint8")
    return Image.fromarray(out)

# ---------- encode (cached) ----------
def encode_set(species_jpgs, transform, batch=16):
    import torch
    from transformers import AutoModel, AutoImageProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    outs = []
    for i in range(0, len(species_jpgs), batch):
        chunk = species_jpgs[i:i + batch]
        imgs = [transform(Image.open(p).convert("RGB")) for p in chunk]
        with torch.no_grad():
            z = model(**proc(images=imgs, return_tensors="pt").to(dev)).last_hidden_state.mean(1)
        outs.append(z.cpu().numpy())
    return np.vstack(outs).astype("f4")

def get_foreground(slug):
    cache = f"{ROOT}/strawman_{slug}_fg.npy"
    if os.path.exists(cache): return np.load(cache)
    jp = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
    t0 = time.time(); X = encode_set(jp, otsu_foreground)
    np.save(cache, X); print(f"  encoded fg {slug}: {X.shape} in {time.time()-t0:.0f}s")
    return X

def get_regions(slug):
    """Returns [N, R, 1280] region embeddings (R regions per specimen)."""
    cache = f"{ROOT}/strawman_{slug}_regions.npy"
    if os.path.exists(cache): return np.load(cache)
    jp = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
    import torch
    from transformers import AutoModel, AutoImageProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
    out = []; t0 = time.time()
    for p in jp:
        regs = grid_regions(Image.open(p).convert("RGB"))
        with torch.no_grad():
            z = model(**proc(images=regs, return_tensors="pt").to(dev)).last_hidden_state.mean(1)
        out.append(z.cpu().numpy())
    X = np.stack(out).astype("f4"); np.save(cache, X)
    print(f"  encoded regions {slug}: {X.shape} in {time.time()-t0:.0f}s")
    return X

# ---------- shared eval machinery (identical to prototype) ----------
rng = np.random.default_rng(0)
def split(n): i = rng.permutation(n); c = int(0.7 * n); return i[:c], i[c:]
ie_tr, ie_te = split(296); ip_tr, ip_te = split(268)

def pca(fit, *apply, k=64):
    mu = fit.mean(0); _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    V = Vt[:min(k, Vt.shape[0])]
    return [((A - mu) @ V.T).astype("f4") for A in (fit, *apply)]

def at_yield(se, sp, yld):
    return float((sp < np.quantile(se, yld)).mean())

def risk_from_feats(tr_e, tr_p, te_e, te_p):
    Xtr = np.vstack([tr_e, tr_p]); ytr = np.r_[np.zeros(len(tr_e)), np.ones(len(tr_p))].astype("f4")
    ens = train_ensemble(Xtr, ytr, Xtr.shape[1], CFG)
    return ensemble_risk(ens, te_e)[0], ensemble_risk(ens, te_p)[0]

def feats_from_raw(Xe, Xp, k=64):
    """Xe/Xp full per-species embeddings [N,1280] -> split + PCA(fit on train pool)."""
    etr, ptr = Xe[ie_tr], Xp[ip_tr]; ete, pte = Xe[ie_te], Xp[ip_te]
    Ptr, Pe, Pp = pca(np.vstack([etr, ptr]), ete, pte, k=k)
    return (Ptr[:len(etr)], Ptr[len(etr):], Pe, Pp)

# ============================================================
if __name__ == "__main__":
    print("STRAWMAN-LOCALIZATION COUNTER-TEST")
    print("(stronger localizers vs GLOBAL, matched yield, prototype's own ensemble)\n")

    gE = np.load(f"{ROOT}/{E}.npy"); gP = np.load(f"{ROOT}/{P}.npy")
    # GLOBAL baseline (reproduce prototype number to anchor)
    ftr_e, ftr_p, fe, fp = feats_from_raw(gE, gP, 64)
    rge, rgp = risk_from_feats(ftr_e, ftr_p, fe, fp)
    print(f"  {'method':<46} | {'@60%':>7} | {'@80%':>7}")
    print(f"  {'GLOBAL (anchor, must match 0.0/2.5)':<46} | {at_yield(rge,rgp,.6)*100:6.1f}% | {at_yield(rge,rgp,.8)*100:6.1f}%")

    # ---- S2 foreground ----
    fgE = get_foreground(E); fgP = get_foreground(P)
    fetr_e, fetr_p, fee, fep = feats_from_raw(fgE, fgP, 64)
    rfe, rfp = risk_from_feats(fetr_e, fetr_p, fee, fep)
    print(f"  {'S2a FOREGROUND only (Otsu-seg)':<46} | {at_yield(rfe,rfp,.6)*100:6.1f}% | {at_yield(rfe,rfp,.8)*100:6.1f}%")
    # fuse global+foreground
    cat_e = np.hstack([ftr_e, fetr_e]); cat_p = np.hstack([ftr_p, fetr_p])
    cat_te_e = np.hstack([fe, fee]); cat_te_p = np.hstack([fp, fep])
    rfe2, rfp2 = risk_from_feats(cat_e, cat_p, cat_te_e, cat_te_p)
    print(f"  {'S2b GLOBAL + FOREGROUND (fused)':<46} | {at_yield(rfe2,rfp2,.6)*100:6.1f}% | {at_yield(rfe2,rfp2,.8)*100:6.1f}%")

    # ---- S1 multi-region MAX-toxic ----
    rgE = get_regions(E); rgP = get_regions(P)   # [N, R, 1280]
    R = rgE.shape[1]
    # train ONE classifier on all regions pooled (region-agnostic part detector), score every
    # region of every test specimen, take MAX toxic risk per specimen.
    # PCA fit on pooled train regions.
    etr_reg = rgE[ie_tr].reshape(-1, 1280); ptr_reg = rgP[ip_tr].reshape(-1, 1280)
    mu = np.vstack([etr_reg, ptr_reg]).mean(0)
    _, _, Vt = np.linalg.svd(np.vstack([etr_reg, ptr_reg]) - mu, full_matrices=False)
    V = Vt[:64]
    def projr(X): return ((X.reshape(-1, 1280) - mu) @ V.T).astype("f4").reshape(X.shape[0], -1, 64)
    Petr = projr(rgE[ie_tr]); Pptr = projr(rgP[ip_tr])
    Pete = projr(rgE[ie_te]); Ppte = projr(rgP[ip_te])
    # train on pooled regions, label = species edibility of the parent specimen
    Xtr = np.vstack([Petr.reshape(-1, 64), Pptr.reshape(-1, 64)])
    ytr = np.r_[np.zeros(Petr.shape[0]*R), np.ones(Pptr.shape[0]*R)].astype("f4")
    ens = train_ensemble(Xtr, ytr, 64, CFG)
    def maxtox(Pz):
        N = Pz.shape[0]; r = ensemble_risk(ens, Pz.reshape(-1, 64))[0].reshape(N, R)
        return r.max(1)  # if ANY region looks toxic -> high score
    rme, rmp = maxtox(Pete), maxtox(Ppte)
    print(f"  {'S1 MULTI-REGION MAX-toxic (part detector)':<46} | {at_yield(rme,rmp,.6)*100:6.1f}% | {at_yield(rme,rmp,.8)*100:6.1f}%")
    # also MEAN-region (smoother) for contrast
    def meantox(Pz):
        N = Pz.shape[0]; r = ensemble_risk(ens, Pz.reshape(-1, 64))[0].reshape(N, R)
        return r.mean(1)
    rme2, rmp2 = meantox(Pete), meantox(Ppte)
    print(f"  {'S1b MULTI-REGION MEAN-toxic':<46} | {at_yield(rme2,rmp2,.6)*100:6.1f}% | {at_yield(rme2,rmp2,.8)*100:6.1f}%")

    # ---- S3 EVERYTHING fused: global + foreground + region-mean-feature ----
    regmean_e_tr = projr(rgE[ie_tr]).mean(1); regmean_p_tr = projr(rgP[ip_tr]).mean(1)
    regmean_e_te = projr(rgE[ie_te]).mean(1); regmean_p_te = projr(rgP[ip_te]).mean(1)
    big_tr_e = np.hstack([ftr_e, fetr_e, regmean_e_tr]); big_tr_p = np.hstack([ftr_p, fetr_p, regmean_p_tr])
    big_te_e = np.hstack([fe, fee, regmean_e_te]); big_te_p = np.hstack([fp, fep, regmean_p_te])
    rbe, rbp = risk_from_feats(big_tr_e, big_tr_p, big_te_e, big_te_p)
    print(f"  {'S3 GLOBAL+FOREGROUND+REGIONS (full fuse)':<46} | {at_yield(rbe,rbp,.6)*100:6.1f}% | {at_yield(rbe,rbp,.8)*100:6.1f}%")

    # ---- S4: ensemble MAX over (global decision OR region-max-toxic) -- the safety-OR a real
    # part-detector cascade would use: eat only if BOTH global-safe AND no region toxic ----
    def pctrank(x): return x.argsort().argsort() / max(len(x) - 1, 1)
    allg = np.r_[rge, rgp]; allm = np.r_[rme, rmp]
    comb = np.maximum(pctrank(allg), pctrank(allm))
    ce, cp = comb[:len(rge)], comb[len(rge):]
    print(f"  {'S4 GLOBAL-OR-REGIONMAX (safety OR cascade)':<46} | {at_yield(ce,cp,.6)*100:6.1f}% | {at_yield(ce,cp,.8)*100:6.1f}%")

    print("\nVERDICT TEST: if EVERY strong localizer >= GLOBAL at both yields, the no-help")
    print("negative is FAIR (localization is a redundant RGB channel, as claimed).")
    print("If ANY strong localizer BEATS GLOBAL (strictly lower poison-eaten), the _volva box")
    print("was a STRAWMAN and the cascade was dismissed prematurely.")
