"""'Should I eat this mushroom?' — active-inference epistemic abstention (mechanism demo).

The canonical JEPA / world-model decision-under-uncertainty question, mechanism-first on
the tabular UCI Mushroom set. The point is NOT in-distribution accuracy (this dataset is
trivially separable) — it is whether an EPISTEMIC term makes the system ABSTAIN on the
dangerous lookalikes it can't be sure about, instead of confidently eating them.

SAFETY: this is a RESEARCH VEHICLE, never foraging advice. The correct output under any real
uncertainty is DON'T EAT / consult a mycologist. The contribution is a method that knows when
to abstain.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cfg:
    csv: str = "data/mushroom.csv"

    # the lookalike split: train on STRONG-odor mushrooms (separable), hold out odor='none'
    # (~3528: mostly edible, but ~120 poisonous "smell-fine" lookalikes) as OOD.
    ood_odor_value: str = "n"
    odor_feature_index: int = 4        # 0-based index among the 22 features (odor is the 5th)

    # ensemble (epistemic engine, reused in spirit from the Seam-3 pilot)
    ensemble_k: int = 12
    steps: int = 400
    batch: int = 256
    lr: float = 1e-3
    bootstrap: bool = True             # per-member bootstrap -> more disagreement off-support

    # active-inference decision over {eat, discard, inspect}
    kappa: float = 2.0                 # epistemic risk-aversion weight (0 = no epistemic term)
    tau_eat: float = 0.15              # EAT only if pessimistic risk (mean + kappa*std) < this
    tau_poison: float = 0.85           # DISCARD if optimistic risk (mean - kappa*std) > this

    seed: int = 0

    # --- image track: I-JEPA on fungi photos ("look at THIS mushroom") ---
    ijepa_id: str = "facebook/ijepa_vith14_1k"   # ViT-H/14, hidden 1280 (verify id in HF)
    image_dir: str = "data/fungi"                # train/edible, train/poisonous, ood/{edible,poisonous}
    latent_dim: int = 1280                       # I-JEPA hidden (mean-pooled patch embeddings)
    novelty_percentile: float = 99.0             # Mahalanobis abstain threshold = this pct of train distances
    # synthetic-image smoke (planted in-dist vs off-manifold lookalike structure)
    syn_latent_dim: int = 64
    syn_novel_dims: int = 6                       # dims ~0 in train, nonzero in OOD = observation-surprise


CFG = Cfg()
