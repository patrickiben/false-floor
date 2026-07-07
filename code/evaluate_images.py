"""Image-track kill-test: the SAME three regimes as the tabular demo, now on I-JEPA latents
(or planted synthetic latents), with continuous-latent observation-surprise (Mahalanobis).

  python evaluate_images.py              # synthetic smoke (no weights/data)
  python evaluate_images.py --real       # real I-JEPA over data/fungi/
"""
from __future__ import annotations

import argparse

import numpy as np

from config import CFG
import images as imgs
from model import train_ensemble, ensemble_risk
from novelty_latent import novelty_flag
from efe import decide, EAT, INSPECT


def summarize(p_mean, p_std, y, cfg, kappa, novel=None):
    d = decide(p_mean, p_std, cfg, kappa=kappa, novel=novel)
    pois, edib = y == 1, y == 0
    return dict(false_eat=int(((d == EAT) & pois).sum()),
                safe_eat=int(((d == EAT) & edib).sum()),
                abstain=int((d == INSPECT).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real I-JEPA over data/fungi/")
    args = ap.parse_args()
    cfg = CFG

    s = imgs.real(cfg) if args.real else imgs.synthetic(cfg)
    (Xtr, ytr), (Xv, yv), (Xood, yood), D = s["train"], s["val"], s["ood"], s["D"]
    mode = "REAL I-JEPA" if args.real else "SYNTHETIC latents"
    print(f"[{mode}] train={len(Xtr)} val={len(Xv)} ood={len(Xood)} "
          f"(poisonous lookalikes={int((yood==1).sum())}, edible={int((yood==0).sum())})  latent_dim={D}")

    ens = train_ensemble(Xtr, ytr, D, cfg)
    pv_m, pv_s = ensemble_risk(ens, Xv)
    po_m, po_s = ensemble_risk(ens, Xood)
    print(f"in-dist val: point acc {float(((pv_m>0.5).astype(int)==yv).mean())*100:.1f}%  "
          f"epistemic std {pv_s.mean():.3f}   |   OOD epistemic std {po_s.mean():.3f}")

    novel_ood, _ = novelty_flag(Xood, Xtr, cfg.novelty_percentile)
    print(f"observation-surprise flags {int(novel_ood.sum())}/{len(Xood)} OOD inputs as off-manifold")

    print(f"\n=== image kill-test (poisonous EATEN of {int((yood==1).sum())}) ===")
    print(f"  {'regime':>38} | {'false_eat':>9} | {'safe_eat':>8} | {'abstain':>7}")
    for name, kw in [
        ("1. point estimate (kappa=0)", dict(kappa=0.0, novel=None)),
        (f"2. + ensemble epistemic (kappa={cfg.kappa})", dict(kappa=cfg.kappa, novel=None)),
        ("3. + observation-surprise (Mahalanobis)", dict(kappa=cfg.kappa, novel=novel_ood)),
    ]:
        r = summarize(po_m, po_s, yood, cfg, **kw)
        flag = "  <- SAFE" if r["false_eat"] == 0 else "  <- DANGEROUS"
        print(f"  {name:>38} | {r['false_eat']:>9} | {r['safe_eat']:>8} | {r['abstain']:>7}{flag}")

    print("\nSAFETY: research vehicle only; under any uncertainty the answer is DON'T EAT.")


if __name__ == "__main__":
    main()
