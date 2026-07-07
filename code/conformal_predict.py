"""POINT #2: Conformal prediction with the 'any-toxic-in-set -> DO NOT EAT' rule.

Split-conformal (LAC/THR) over the trained SPECIES {chanterelle(E), jack-o'-lantern(P),
field mushroom(E), destroying angel(P)}. For each test mushroom we form a prediction set with a
distribution-free marginal coverage guarantee 1-alpha, then EAT only if the set is non-empty and
contains ONLY edible species (the safety rule). Scored on the SAME held-out specimens as everything.

Two honest stress points this exposes:
  (a) coverage-vs-yield: to make the set include the true toxic label ~always (safety), sets blow up
      and edibles also catch a toxic -> yield collapses. The guarantee has a price.
  (b) OPEN-SET break: conformal's guarantee needs exchangeability. A novel Death Cap is NOT a trained
      class, so the set (over trained classes) can NEVER contain 'death cap' -> the guarantee is VOID
      exactly where danger is highest. It is only caught if it happens to map onto the toxic Amanita.

SAFETY: research vehicle only; a marginal guarantee is not a per-mushroom guarantee. When in doubt DON'T EAT.
"""
import numpy as np, torch, torch.nn as nn
np.seterr(all="ignore"); torch.manual_seed(0)

ROOT = "data/fungi_real"
SPEC = {"edible_chanterelle": 0, "poison_jackolantern": 1, "edible_fieldmushroom": 2, "poison_destroyingangel": 3}
EDIBLE = {0: True, 1: False, 2: True, 3: False}        # class -> edible?
NAME = {0: "chanterelle", 1: "jack-o'-lantern", 2: "field mushroom", 3: "DESTROYING ANGEL"}
L = {s: np.load(f"{ROOT}/{s}.npy") for s in SPEC}
DC = np.load(f"{ROOT}/heldout_deathcap.npy")           # open-set toxic, never trained

rng = np.random.default_rng(0)
def split3(X):                                          # 50% train / 25% calib / 25% test
    i = rng.permutation(len(X)); a = int(0.5 * len(X)); b = int(0.75 * len(X))
    return X[i[:a]], X[i[a:b]], X[i[b:]]
tr, ca, te = {}, {}, {}
for s in SPEC: tr[s], ca[s], te[s] = split3(L[s])

Xtr = np.vstack([tr[s] for s in SPEC]); ytr = np.concatenate([np.full(len(tr[s]), SPEC[s]) for s in SPEC])
mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:64]
proj = lambda X: ((X - mu) @ V.T).astype("f4")

class Net(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 4))
    def forward(s, x): return s.f(x)
net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
Xt, yt = torch.tensor(proj(Xtr)), torch.tensor(ytr)
for _ in range(1200):
    opt.zero_grad(); nn.functional.cross_entropy(net(Xt), yt).backward(); opt.step()
net.eval()
def softmax(X):
    with torch.no_grad(): return torch.softmax(net(torch.tensor(proj(X))), 1).numpy()

# split-conformal LAC: nonconformity = 1 - p[true]; threshold = (1-alpha) quantile on calibration
Pca = np.vstack([softmax(ca[s]) for s in SPEC]); yca = np.concatenate([np.full(len(ca[s]), SPEC[s]) for s in SPEC])
cal_scores = 1 - Pca[np.arange(len(yca)), yca]
def qhat(alpha):
    n = len(cal_scores); k = min(n, int(np.ceil((n + 1) * (1 - alpha)))); return np.sort(cal_scores)[k - 1]
def pset(P, qh): return [np.where(P[i] >= 1 - qh)[0] for i in range(len(P))]
def eat(sets): return np.array([len(s) > 0 and all(EDIBLE[c] for c in s) for s in sets])

Pda, Pfm, Pdc = softmax(te["poison_destroyingangel"]), softmax(te["edible_fieldmushroom"]), softmax(DC)
Pte = np.vstack([softmax(te[s]) for s in SPEC]); yte = np.concatenate([np.full(len(te[s]), SPEC[s]) for s in SPEC])

print("Where does a Death Cap's mass land (point prediction, no conformal)?")
arg = Pdc.argmax(1);
for c in range(4): print(f"   {NAME[c]:18}: {100*np.mean(arg==c):4.0f}%", end="")
print(f"\n   -> {100*np.mean([EDIBLE[c] for c in arg]):.0f}% of Death Caps point-predict to an EDIBLE class\n")

print("split-conformal LAC, rule = EAT iff prediction set is non-empty AND all-edible:")
print(f"  {'target cov':>10} | {'emp cov(test)':>13} | {'DA false-eat':>12} | {'DeathCap false-eat':>18} | {'field yield':>11} | {'avg|set| DA/DC':>14}")
for alpha in (0.10, 0.05, 0.01, 0.001):
    qh = qhat(alpha)
    cov = np.mean([yte[i] in np.where(Pte[i] >= 1 - qh)[0] for i in range(len(yte))])
    Sda, Sfm, Sdc = pset(Pda, qh), pset(Pfm, qh), pset(Pdc, qh)
    fe_da, yld, fe_dc = eat(Sda).mean(), eat(Sfm).mean(), eat(Sdc).mean()
    szda, szdc = np.mean([len(s) for s in Sda]), np.mean([len(s) for s in Sdc])
    print(f"  {1-alpha:>10.3f} | {cov:>13.3f} | {fe_da*100:>11.1f}% | {fe_dc*100:>17.1f}% | {yld*100:>10.1f}% | {szda:.2f}/{szdc:.2f}")

print("\nDA false-eat falls with coverage (the true toxic label enters the set) BUT field yield falls too")
print("(coverage-vs-yield, now with a guarantee). DeathCap false-eat = the OPEN-SET break: the guarantee")
print("is void for a class the model never saw -- it's only caught when it maps onto the toxic Amanita.")
print("SAFETY: research vehicle only; marginal coverage is not per-mushroom safety. When in doubt, DON'T EAT.")
