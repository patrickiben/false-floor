"""Decision-curve analysis (net benefit) for the eat decision -- the statistician-legible way to
state the result. Net benefit per specimen of a "trust the model, eat if it says safe" policy vs
the default "eat nothing" (NB=0), as a function of the harm:benefit ratio H:B (eating a poison
vs eating an edible). For a lethal false-eat, H >> B, so the threshold-probability is extreme.

If, at realistic lethal H:B, no decision threshold beats "eat nothing" (NB <= 0), then DCA's
verdict is: the model provides NO net benefit for the eat decision -- abstaining dominates. That
is the "no operating point is both safe and useful" finding, in the apparatus a clinical
statistician expects.  Runs on the cached real I-JEPA latents.
"""
from __future__ import annotations

import numpy as np
np.seterr(all="ignore")

from config import CFG
from model import train_ensemble, ensemble_risk

CFG.ensemble_k, CFG.steps = 8, 300
L = {s: np.load(f"data/fungi_real/{s}.npy") for s in
     ["edible_chanterelle", "poison_jackolantern", "edible_fieldmushroom", "poison_destroyingangel"]}
rng = np.random.default_rng(0)
spl = {}
for s in L:
    i = rng.permutation(len(L[s])); c = int(0.7 * len(L[s])); spl[s] = (L[s][i[:c]], L[s][i[c:]])

Xtr = np.vstack([spl[s][0] for s in L])
ytr = np.concatenate([(np.zeros if "edible" in s else np.ones)(len(spl[s][0])) for s in L]).astype("f4")
mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:64]
proj = lambda X: (X - mu) @ V.T
ens = train_ensemble(proj(Xtr), ytr, 64, CFG)

edible_te, poison_te = spl["edible_fieldmushroom"][1], spl["poison_destroyingangel"][1]
re_, _ = ensemble_risk(ens, proj(edible_te))      # P(poison) for held-out edibles (want low)
rp, _ = ensemble_risk(ens, proj(poison_te))       # P(poison) for held-out Destroying Angels (want high)
N = len(re_) + len(rp)

print(f"held-out lethal pair: {len(edible_te)} field mushrooms (edible), {len(rp)} Destroying Angels (poison)")
print("decision-curve analysis -- net benefit of 'eat if model says safe' vs 'eat nothing' (NB=0):\n")
print(f"  {'harm:benefit (eat poison : eat edible)':>40} | {'best net benefit':>16} | decision")
for HB in (1, 5, 20, 100, 1000):
    best_nb, best = 0.0, (0, 0, 0.0)              # 'eat nothing' baseline = NB 0
    for tau in np.linspace(0, 1, 201):
        eat_ed = int((re_ < tau).sum()); eat_po = int((rp < tau).sum())
        nb = (1.0 * eat_ed - HB * eat_po) / N
        if nb > best_nb: best_nb, best = nb, (eat_ed, eat_po, tau)
    if best_nb <= 0:
        dec = "EAT NOTHING dominates -- model adds no net benefit (any eating threshold also eats a poison)"
    else:
        dec = f"eat @tau={best[2]:.2f}: {best[0]} edible, {best[1]} poison eaten"
    print(f"  {str(HB)+':1':>40} | {best_nb:>16.4f} | {dec}")

print("\nReads as: above a modest harm:benefit ratio, NO threshold beats abstaining -- because the")
print("confidently-wrong Destroying Angels sit at lower model-risk than some edibles, so any tau that")
print("eats an edible also clears a poison. SAFETY: research vehicle only; when in doubt, DON'T EAT.")
