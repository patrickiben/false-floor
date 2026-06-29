"""AI super-resolution of the macro photo -- does upscaling RECOVER diagnostic information, or hallucinate it?
Data-processing inequality: SR output = f(low-res photo) carries no more edibility info; it invents plausible
high-frequency detail from a natural-image prior. Rigorous, non-strawman test: deliberately DOWNSAMPLE real
lethal-pair photos (remove known information), then upscale back with (a) bicubic and (b) a REAL AI SR model
(Swin2SR x4), re-encode with the SAME I-JEPA, and ask whether ANY upscaling RECOVERS the separability /
false-eat that downsampling destroyed. If SR can't recover deliberately-removed information, it certainly
can't reveal never-captured sub-optical detail (spores are below a phone's optical resolution -- not in the
photons). We also flag the danger: SR makes images LOOK more diagnostic -> false confidence. SAFETY: research vehicle; DON'T EAT.
"""
import glob, warnings
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
import torch
from PIL import Image
from config import CFG
from model import train_ensemble, ensemble_risk

LE, LP = "edible_fieldmushroom", "poison_destroyingangel"
ROOT = "data/fungi_real"; NPER = 120; DOWN = 56
SR_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

from transformers import AutoModel, AutoImageProcessor, Swin2SRForImageSuperResolution
print("loading I-JEPA + Swin2SR ...", flush=True)
jproc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
jmodel = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").eval().to(dev)
sproc = AutoImageProcessor.from_pretrained(SR_ID)
smodel = Swin2SRForImageSuperResolution.from_pretrained(SR_ID).eval().to(dev)


@torch.no_grad()
def ijepa(imgs):
    out = []
    for k in range(0, len(imgs), 8):
        h = jmodel(**jproc(images=imgs[k:k + 8], return_tensors="pt").to(dev)).last_hidden_state
        out.append(h.mean(1).cpu().numpy().astype("f4"))
    return np.concatenate(out)


@torch.no_grad()
def sr_up(im):
    inp = sproc(images=im, return_tensors="pt").to(dev)
    rec = smodel(**inp).reconstruction.squeeze(0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((rec.transpose(1, 2, 0) * 255).round().astype("uint8"))


def encode_set(slug):
    paths = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))[:NPER]
    O, B, S = [], [], []
    for p in paths:
        orig = Image.open(p).convert("RGB").resize((224, 224))
        low = orig.resize((DOWN, DOWN), Image.BICUBIC)               # remove information
        O.append(orig)
        B.append(low.resize((224, 224), Image.BICUBIC))             # bicubic upscale
        S.append(sr_up(low).resize((224, 224)))                     # AI SR upscale (x4: 56->224)
    print(f"  encoded {slug} ({len(paths)} imgs x 3 conditions)", flush=True)
    return ijepa(O), ijepa(B), ijepa(S)


Oe, Be, Se = encode_set(LE)
Op, Bp, Sp = encode_set(LP)


def evalcond(Xe, Xp):
    res = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        def sp(n): i = rng.permutation(n); c = int(.7 * n); return i[:c], i[c:]
        etr, ete = sp(len(Xe)); ptr, pte = sp(len(Xp))
        Xtr = np.vstack([Xe[etr], Xp[ptr]]); ytr = np.r_[np.zeros(len(etr)), np.ones(len(ptr))].astype("f4")
        mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:48]
        pj = lambda A: ((A - mu) @ V.T).astype("f4")
        ens = train_ensemble(pj(Xtr), ytr, 48, CFG)
        re = ensemble_risk(ens, pj(Xe[ete]))[0]; rp = ensemble_risk(ens, pj(Xp[pte]))[0]
        fe = float((rp < np.quantile(re, .8)).mean()) * 100
        au = float(sum((b > a) + .5 * (b == a) for b in rp for a in re) / (len(rp) * len(re)))
        res.append([fe, au])
    return np.array(res)


print("\nlethal-pair separability + false-eat by capture/upscale condition (15-seed mean +/- sd):\n")
print(f"  {'condition':30} | {'false-eat @80%':>16} | {'AUROC':>7}")
out = {}
for name, (Xe, Xp) in [("ORIGINAL 224", (Oe, Op)), ("downsampled 56 -> BICUBIC", (Be, Bp)),
                       ("downsampled 56 -> AI-SR (Swin2SR)", (Se, Sp))]:
    r = evalcond(Xe, Xp); out[name] = r
    print(f"  {name:30} | {r[:,0].mean():6.1f} +/- {r[:,0].std():4.1f}%   | {r[:,1].mean():.3f}")

o, b, s = out["ORIGINAL 224"], out["downsampled 56 -> BICUBIC"], out["downsampled 56 -> AI-SR (Swin2SR)"]
print(f"\n  info lost to downsampling:    AUROC {o[:,1].mean():.3f} -> {b[:,1].mean():.3f}  (false-eat {o[:,0].mean():.1f}% -> {b[:,0].mean():.1f}%)")
print(f"  AI-SR recovery over bicubic:  AUROC {b[:,1].mean():.3f} -> {s[:,1].mean():.3f}  (delta {s[:,1].mean()-b[:,1].mean():+.3f}); reaches original? {'NO' if s[:,1].mean() < o[:,1].mean()-0.005 else 'maybe'}")
print("""
READING: if AI-SR does NOT recover the AUROC/false-eat lost to downsampling (lands ~= bicubic, below original),
then upscaling cannot restore information the sensor didn't capture -- it re-renders plausible detail from a
prior (data-processing inequality). Since it can't recover DELIBERATELY-removed info, it cannot reveal
never-captured sub-optical detail (spores). Any SR>bicubic gain that does NOT track real recovery is
hallucinated texture = the dangerous part (the image LOOKS more diagnostic -> false confidence). SAFETY: research vehicle; DON'T EAT.
""")
