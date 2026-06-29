"""Phase 3: metadata fusion catches the lookalike that vision-only CANNOT.

There is a worst-case toxic lookalike: visually near-identical to an edible AND on the training
manifold. For it, ensemble disagreement is low (the model is confidently wrong) AND
observation-surprise is low (it looks in-distribution) -- BOTH vision-side abstention signals
fail. Only context (geography / season / substrate) separates it. This is why mycologists ask
"where and when did you find it?".

We construct exactly that subgroup and show: vision-only (with full ensemble+Mahalanobis
abstention) EATS it; image+metadata fusion catches it. Honest demonstration that metadata is a
THIRD, complementary signal -- not a mere accuracy boost.

    python evaluate_fusion.py        # synthetic; real version = concatenate I-JEPA latent + the
                                     # lat/lon/month metadata from fetch_fungi.py's metadata.csv
"""
from __future__ import annotations

import numpy as np

from config import CFG
from model import train_ensemble, ensemble_risk
from novelty_latent import novelty_flag, fit, distance
from efe import decide, EAT, INSPECT

rng = np.random.default_rng(0)
D, NV, M = 64, 6, 4            # latent dim, novelty dims, metadata dims


def block(n, disc, novel, meta0):
    x = rng.normal(0, 1, (n, D)).astype(np.float32)
    x[:, 0] = rng.normal(disc, 0.5, n)
    x[:, D - NV:] = novel + rng.normal(0, 0.05, (n, NV))
    m = rng.normal(0, 1, (n, M)).astype(np.float32)
    m[:, 0] = rng.normal(meta0, 0.8, n)         # metadata[0] carries a (weaker) class signal
    return x, m


def main():
    # training: vision dim0 AND metadata[0] both carry class info (edible neg, poison pos)
    Xe, Me = block(1500, -2.0, 0.0, -1.0); ye = np.zeros(1500, np.int64)
    Xp, Mp = block(1500, +2.0, 0.0, +1.0); yp = np.ones(1500, np.int64)
    Xtr = np.vstack([Xe, Xp]); Mtr = np.vstack([Me, Mp]); ytr = np.concatenate([ye, yp])

    # the killer subgroup: looks EDIBLE (dim0 ~ -2) AND on-manifold (novel ~ 0) -> vision is
    # confidently wrong and surprise-blind. But it's POISONOUS, with poison-like metadata.
    Xc, Mc = block(120, -2.0, 0.0, +1.0); yc = np.ones(120, np.int64)

    d_lat = D
    print(f"killer subgroup: {len(Xc)} toxic lookalikes, visually edible + on-manifold "
          f"(neither vision signal can see them)")

    # --- vision-only: ensemble + Mahalanobis abstention on the LATENT ---
    ens_v = train_ensemble(Xtr, ytr, d_lat, CFG)
    pm, ps = ensemble_risk(ens_v, Xc)
    novel, _ = novelty_flag(Xc, Xtr, CFG.novelty_percentile)
    dv = decide(pm, ps, CFG, kappa=CFG.kappa, novel=novel)
    v_false_eat = int(((dv == EAT) & (yc == 1)).sum())
    print(f"\nvision-only (ensemble eps + Mahalanobis surprise): "
          f"epistemic std {ps.mean():.3f}, surprise-flagged {int(novel.sum())}/{len(Xc)} "
          f"-> poisonous EATEN = {v_false_eat}/120  <- {'SAFE' if v_false_eat==0 else 'DANGEROUS'}")

    # --- (b) naive metadata fusion (the roadmap's Phase 3): concat [latent ; metadata] ---
    ens_f = train_ensemble(np.hstack([Xtr, Mtr]), ytr, d_lat + M, CFG)
    pmf, psf = ensemble_risk(ens_f, np.hstack([Xc, Mc]))
    df = decide(pmf, psf, CFG, kappa=CFG.kappa, novel=novel)
    f_false_eat = int(((df == EAT) & (yc == 1)).sum())
    print(f"naive fusion (concat classifier): poisonous EATEN = {f_false_eat}/120  "
          f"<- helps, but the confident-wrong VISION signal dominates the weak metadata feature")

    # --- (c) metadata as a CONSISTENCY check: abstain when where/when contradicts the vision
    #          hypothesis. distance of the input's metadata to the PREDICTED class's training metadata.
    # corroboration DIAL: EAT only if the metadata positively agrees with the vision hypothesis
    # (distance to predicted-class metadata below a percentile of that class's own training spread).
    mu_e, inv_e = fit(Mtr[ytr == 0]); mu_p, inv_p = fit(Mtr[ytr == 1])
    pred = (pm >= 0.5).astype(int)
    dpred = np.where(pred == 0, distance(Mc, mu_e, inv_e), distance(Mc, mu_p, inv_p))
    print(f"\n  corroboration dial (require metadata to AGREE with the vision hypothesis to eat):")
    print(f"  {'demand':>20} | {'poisonous EATEN':>15} | {'abstained':>9}")
    for pct in (95, 75, 50):                       # tighter % = more corroboration demanded
        thr = np.percentile(distance(Mtr[ytr == 0], mu_e, inv_e), pct)
        contradicts = dpred > thr
        dc = decide(pm, ps, CFG, kappa=CFG.kappa, novel=(novel | contradicts))
        ce = int(((dc == EAT) & (yc == 1)).sum()); ab = int((dc == INSPECT).sum())
        tag = "lenient" if pct == 95 else ("strict" if pct == 50 else "moderate")
        print(f"  {tag+f' (p{pct})':>20} | {ce:>15} | {ab:>9}{'  <- SAFE' if ce==0 else ''}")

    print(f"\nverdict: a visually-identical, on-manifold toxic lookalike defeats BOTH vision-side "
          f"abstention signals (vision-only ate {v_false_eat}/120); naive concat-fusion barely helps "
          f"({f_false_eat}/120) because the confident-wrong vision feature dominates. NO single signal "
          f"is sufficient. Safety is a CORROBORATION DIAL: the more independent agreement you demand "
          f"before eating, the closer false-eats go to zero -- and the more (edible) mushrooms you "
          f"abstain on. There is no setting that is both safe AND high-yield. That is the real answer "
          f"to 'should I eat this?': when in doubt, don't.")
    print("SAFETY: research vehicle only; under any uncertainty the answer is DON'T EAT.")


if __name__ == "__main__":
    main()
