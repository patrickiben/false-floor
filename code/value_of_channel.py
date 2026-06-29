"""LEVER #2 (besides-JEPA): value of an INDEPENDENT input channel (spore print / chemical test / DNA).

We proved only a new channel adds information. This quantifies: given the REAL JEPA risk on the
held-out lethal pair, how reliable must a NEW channel be -- and how INDEPENDENT -- to drive the
lethal false-eat toward 0 at usable yield? Corroboration = eat only if JEPA AND the channel clear it.

Key contrast:
  - INDEPENDENT channel (errors uncorrelated with JEPA): keys on real biology (toxin/spore), so it
    can catch the confidently-wrong poison JEPA misses.
  - REDUNDANT channel (another image model; errors ~ JEPA's): high AUROC but fails on the SAME cases
    -> corroboration buys almost nothing. (This is why a fancier image net doesn't help.)

SAFETY: research vehicle only; even a strong channel is a corroborator, not a clearance. DON'T EAT.
"""
from __future__ import annotations
import numpy as np
from statistics import NormalDist
np.seterr(all="ignore")
from config import CFG
from model import train_ensemble, ensemble_risk

ROOT = "data/fungi_real"
L = {s: np.load(f"{ROOT}/{s}.npy") for s in
     ["edible_chanterelle", "poison_jackolantern", "edible_fieldmushroom", "poison_destroyingangel"]}
rng = np.random.default_rng(0)
def split(X): i = rng.permutation(len(X)); c = int(0.7 * len(X)); return X[i[:c]], X[i[c:]]
fm_tr, fm_te = split(L["edible_fieldmushroom"]); da_tr, da_te = split(L["poison_destroyingangel"])
Xtr = np.vstack([L["edible_chanterelle"], L["poison_jackolantern"], fm_tr, da_tr])
ytr = np.r_[np.zeros(len(L["edible_chanterelle"])), np.ones(len(L["poison_jackolantern"])),
            np.zeros(len(fm_tr)), np.ones(len(da_tr))].astype("f4")
mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:64]
ens = train_ensemble((Xtr - mu) @ V.T, ytr, 64, CFG)
re = ensemble_risk(ens, (fm_te - mu) @ V.T)[0]      # JEPA risk, held-out edibles (want low)
rp = ensemble_risk(ens, (da_te - mu) @ V.T)[0]      # JEPA risk, held-out Destroying Angels (poison)

YLD = 0.60                                            # operate JEPA at 60% edible yield
tj = np.quantile(re, YLD)
fe_jepa = float((rp < tj).mean())
print(f"held-out: {len(re)} edible, {len(rp)} poison.  JEPA alone @60% yield -> false-eat {fe_jepa*100:.1f}%\n")
print("fused (eat only if JEPA AND channel clear), 300 noise draws averaged:")
print(f"  {'channel AUROC':>13} | {'INDEPENDENT: false-eat / yield':>32} | {'REDUNDANT(image-like): false-eat / yield':>42}")

def dprime(au): return np.sqrt(2) * NormalDist().inv_cdf(au)

for au in (0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
    d = dprime(au); fe_i = yl_i = fe_r = yl_r = 0.0; N = 300
    for _ in range(N):
        # INDEPENDENT channel: fresh noise, separates classes at AUROC=au
        ce = rng.normal(0, 1, len(re)); cp = rng.normal(d, 1, len(rp))
        tc = np.quantile(ce, YLD)
        yl_i += ((re < tj) & (ce < tc))[:].mean(); fe_i += ((rp < tj) & (cp < tc)).mean()
        # REDUNDANT channel: aligned with JEPA risk (same errors) + tiny noise, scaled to AUROC=au
        ze = (re - re.mean()) / (re.std() + 1e-9); zp = (rp - rp.mean()) / (rp.std() + 1e-9)
        ce2 = ze * d + rng.normal(0, 1, len(re)); cp2 = zp * d + d + rng.normal(0, 1, len(rp))
        tc2 = np.quantile(ce2, YLD)
        yl_r += ((re < tj) & (ce2 < tc2)).mean(); fe_r += ((rp < tj) & (cp2 < tc2)).mean()
    print(f"  {au:>13.2f} | {f'{fe_i/N*100:5.1f}% / {yl_i/N*100:4.1f}%':>32} | "
          f"{f'{fe_r/N*100:5.1f}% / {yl_r/N*100:4.1f}%':>42}")

print("\nReads as: an INDEPENDENT channel drives false-eat toward 0 as its reliability rises (needs")
print("~0.95-0.99 AUROC -- i.e. a real toxin/spore/DNA assay -- and costs yield); a REDUNDANT image-")
print("like channel barely helps even at high AUROC (it fails on the same confidently-wrong poison).")
print("That is the quantitative case for a new SENSOR, not a new model. SAFETY: when in doubt, DON'T EAT.")
