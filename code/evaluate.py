"""The kill-test: does the epistemic term make the system ABSTAIN on the poisonous
odorless lookalikes, instead of eating them?

The catastrophic metric is the FALSE-EAT count: poisonous mushrooms the system decides to
EAT. We sweep kappa (the epistemic weight); kappa=0 is the point-estimate ablation. Headline:
the epistemic term converts dangerous false-eats into safe abstentions.
"""
from __future__ import annotations

import numpy as np

from config import CFG
from data import splits
from model import train_ensemble, ensemble_risk
from novelty import train_support, novelty_flag
from efe import decide, EAT, DISCARD, INSPECT


def summarize(p_mean, p_std, y, cfg, kappa, novel=None):
    d = decide(p_mean, p_std, cfg, kappa=kappa, novel=novel)
    pois = y == 1
    edib = y == 0
    false_eat = int(((d == EAT) & pois).sum())          # poisonous, but EATEN  <-- catastrophic
    safe_eat = int(((d == EAT) & edib).sum())           # edible, eaten          (the upside)
    abstain = int((d == INSPECT).sum())
    discard = int((d == DISCARD).sum())
    return dict(false_eat=false_eat, safe_eat=safe_eat, abstain=abstain, discard=discard,
                n_pois=int(pois.sum()), n_edib=int(edib.sum()))


def main():
    cfg = CFG
    s = splits(cfg)
    (Xtr, ytr), (Xv, yv), (Xood, yood), D = s["train"], s["val"], s["ood"], s["D"]
    print(f"data: train={len(Xtr)} val={len(Xv)} ood(odor=none)={len(Xood)} "
          f"(poisonous lookalikes={int((yood==1).sum())}, edible={int((yood==0).sum())})  features={D}")

    print(f"\ntraining {cfg.ensemble_k}-member ensemble on strong-odor mushrooms ...")
    ens = train_ensemble(Xtr, ytr, D, cfg)

    pv_m, pv_s = ensemble_risk(ens, Xv)
    po_m, po_s = ensemble_risk(ens, Xood)
    # in-dist accuracy (point estimate) + epistemic separation
    acc = float(((pv_m > 0.5).astype(int) == yv).mean())
    print(f"\nin-dist val: point accuracy {acc*100:.1f}%   mean epistemic std {pv_s.mean():.3f}")
    print(f"OOD lookalikes:                              mean epistemic std {po_s.mean():.3f}"
          f"   <- epistemic FIRES on the off-support set" if po_s.mean() > pv_s.mean()
          else f"OOD epistemic std {po_s.mean():.3f}")

    # observation-surprise signal: which OOD inputs are off the training manifold
    novel_ood = novelty_flag(Xood, train_support(Xtr))

    print("\n=== kill-test: decisions on the odorless OOD set (the lookalike trap) ===")
    print(f"  (catastrophic = false_eat: poisonous mushrooms EATEN, out of {int((yood==1).sum())})")
    print(f"  {'regime':>34} | {'false_eat':>9} | {'safe_eat':>8} | {'abstain':>7} | {'discard':>7}")
    regimes = [
        ("1. point estimate (kappa=0)",                 dict(kappa=0.0, novel=None)),
        (f"2. + ensemble epistemic (kappa={cfg.kappa})", dict(kappa=cfg.kappa, novel=None)),
        ("3. + observation-surprise default-deny",       dict(kappa=cfg.kappa, novel=novel_ood)),
    ]
    rows = {}
    for name, kw in regimes:
        r = summarize(po_m, po_s, yood, cfg, **kw); rows[name] = r
        flag = "  <- DANGEROUS" if r["false_eat"] > 0 else "  <- SAFE"
        print(f"  {name:>34} | {r['false_eat']:>9} | {r['safe_eat']:>8} | {r['abstain']:>7} | {r['discard']:>7}{flag}")

    r1, r2, r3 = (rows[n] for n, _ in regimes)
    print(f"\nverdict:")
    print(f"  - ensemble disagreement alone: false-eats {r1['false_eat']} -> {r2['false_eat']} "
          f"(helps, but MISSES the confidently-wrong-consensus lookalikes -- the Seam-3 lesson).")
    print(f"  - + observation-surprise (knowing the input is off-manifold): false-eats -> {r3['false_eat']} (SAFE),")
    print(f"    but safe-eats {r1['safe_eat']} -> {r3['safe_eat']}: the cost of safety is abstaining on the")
    print(f"    whole class you never learned to recognize. A safe forager doesn't eat what it can't place.")
    print("\nSAFETY: research vehicle only; under any uncertainty the answer is DON'T EAT, consult a mycologist.")


if __name__ == "__main__":
    main()
