"""Stress-test the metadata result: is the 0.91 metadata AUROC real edibility signal, or just
SPECIES-RANGE separation? If it's range, then restricting to a region where BOTH species co-occur
(where a forager actually faces the choice) should collapse the signal. Uses cached paired data.
"""
from __future__ import annotations
import json, numpy as np
from collections import Counter
np.seterr(all="ignore")
from config import CFG
from model import train_ensemble, ensemble_risk
from gate import auroc

ROOT = "data/fungi_meta"
PAIR = {"edible_fieldmushroom": 0, "poison_destroyingangel": 1}


def load(s): return np.load(f"{ROOT}/{s}_lat.npy"), json.load(open(f"{ROOT}/{s}_metaA.json"))
def mf(M): return np.array([[np.sin(2*np.pi*r["month"]/12), np.cos(2*np.pi*r["month"]/12),
                             r["lat"]/90, r["lon"]/180, abs(r["lat"])/90] for r in M], dtype="f4")


def evalset(M, y, rng):
    """metadata-only AUROC + poison-eaten@60% yield on a 70/30 split of the given subset."""
    X = mf(M); n = len(y)
    if min((y == 0).sum(), (y == 1).sum()) < 8: return None
    idx = rng.permutation(n); c = int(0.7 * n); tr, te = idx[:c], idx[c:]
    if min((y[te] == 0).sum(), (y[te] == 1).sum()) < 3: return None
    ens = train_ensemble(X[tr], y[tr].astype("f4"), X.shape[1], CFG)
    r, _ = ensemble_risk(ens, X[te]); yte = y[te]
    au = auroc(r, yte)
    re, rp = r[yte == 0], r[yte == 1]
    thr = np.quantile(re, 0.6); pe = float((rp < thr).mean())
    return au, pe, int((y == 0).sum()), int((y == 1).sum())


def main():
    rng = np.random.default_rng(0)
    M = {s: load(s)[1] for s in PAIR}
    print("=== geographic distribution (where the records actually are) ===")
    for s in PAIR:
        cc = Counter(r.get("country") for r in M[s])
        print(f"  {s:24} top countries: {cc.most_common(5)}")

    def subset(pred, label):
        Mall, yall = [], []
        for s, lab in PAIR.items():
            keep = [r for r in M[s] if pred(r)]
            Mall += keep; yall += [lab] * len(keep)
        res = evalset(Mall, np.array(yall), np.random.default_rng(0))
        if res is None:
            print(f"  {label:38} insufficient co-occurring records"); return
        au, pe, ne, npo = res
        print(f"  {label:38} metadata AUROC {au:.2f}  poison-eaten@60% {pe*100:4.1f}%   (n={ne} edible / {npo} poison)")

    print("\n=== metadata-only signal as the region tightens toward co-occurrence ===")
    subset(lambda r: True,                                              "GLOBAL (pooled, all records)")
    subset(lambda r: r.get("country") == "United States of America",    "United States only")
    subset(lambda r: r.get("country") == "United States of America"
                     and -90 <= r["lon"] <= -67 and 35 <= r["lat"] <= 47, "eastern US box (true co-occurrence)")
    print("\nIf AUROC falls toward 0.5 (and poison-eaten rises) as the region tightens, the 'metadata")
    print("solves it' result was SPECIES-RANGE, not edibility -- a co-located lookalike defeats it.")
    print("SAFETY: location is a weak corroborator at best, never a clearance. When in doubt, DON'T EAT.")


if __name__ == "__main__":
    main()
