"""Canonical Clopper-Pearson binomial bound used throughout the paper's evaluation scripts.

`cp_upper` here is byte-identical to the inline copy in hard_regime.py, hard_pairs.py, pairs_encoders.py,
case2_plants.py ... case6_berries.py. Extracting the one canonical copy lets `test_cp_bounds.py`
machine-check it (property-based + differential against scipy/statsmodels) and lets `verify_cp_symbolic.py`
symbolically re-derive the Beta-binomial identity it rests on. The scripts keep their inline copies for
self-containment; `test_cp_bounds.py` asserts those copies are textually identical to this one.
"""
import math


def cp_upper(x, n, alpha=0.05):
    """One-sided (1-alpha) Clopper-Pearson UPPER bound on a binomial rate (x of n). No scipy."""
    if n == 0: return float("nan")
    if x >= n: return 1.0
    def cdf_le(p):
        if p <= 0.0: return 1.0
        if p >= 1.0: return 0.0
        return sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                            + k * math.log(p) + (n - k) * math.log1p(-p)) for k in range(x + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf_le(mid) > alpha: lo = mid
        else: hi = mid
    return hi
