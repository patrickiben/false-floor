"""The active-inference-flavoured decision over {EAT, DISCARD, INSPECT}.

Given the ensemble's pragmatic risk p_mean and epistemic uncertainty p_std:
  pessimistic risk  ucb = clip(p_mean + kappa*p_std)   # assume the worst within our uncertainty
  optimistic  risk  lcb = clip(p_mean - kappa*p_std)
  EAT      iff ucb < tau_eat      (safe even under the worst-case reading)
  DISCARD  iff lcb > tau_poison   (poisonous even under the best-case reading)
  INSPECT  otherwise              (uncertain -> gather more info, the safe default)

The kappa*std term is the epistemic risk-aversion: where the ensemble disagrees (no support,
e.g. an odorless lookalike), ucb is pushed up and the system ABSTAINS. kappa=0 removes the
epistemic term entirely (point-estimate decision) -- the ablation.

HONEST: this is risk-sensitive decision under model uncertainty, motivated by active
inference's preference to avoid uncertain/surprising outcomes. It is NOT the full variational
expected-free-energy integral; we don't claim it is (same discipline as the planner's energy).
"""
from __future__ import annotations

import numpy as np

from config import Cfg, CFG

EAT, DISCARD, INSPECT = 0, 1, 2
NAMES = {EAT: "EAT", DISCARD: "DISCARD", INSPECT: "INSPECT"}


def decide(p_mean: np.ndarray, p_std: np.ndarray, cfg: Cfg = CFG, kappa: float = None,
           novel: np.ndarray = None) -> np.ndarray:
    k = cfg.kappa if kappa is None else kappa
    ucb = np.clip(p_mean + k * p_std, 0.0, 1.0)
    lcb = np.clip(p_mean - k * p_std, 0.0, 1.0)
    d = np.full(len(p_mean), INSPECT, dtype=np.int64)
    d[ucb < cfg.tau_eat] = EAT
    d[lcb > cfg.tau_poison] = DISCARD
    if novel is not None:                 # observation-surprise default-deny: never EAT the unrecognized
        d[novel & (d == EAT)] = INSPECT
    return d
