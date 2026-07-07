"""The symmetrization-null gate -- the standing protocol EVERY new abstention signal must
pass before it earns a place. It enforces a falsification discipline, not a metric.

A candidate signal PASSES only if it adds incremental predictive power for false-eat BEYOND a
baseline that ALREADY INCLUDES the classifier's own risk + standard symmetric measures -- and,
if it claims to be DIRECTED, only if symmetrizing it destroys that added value. This is exactly
the check that catches (a) a signal redundant with the symmetric baseline (a tautology) and
(b) a circular "directed" test whose value is really symmetric.

Verdicts:
  FAIL-redundant            adds ~nothing over [risk + symmetric baselines] (the tautology trap)
  FAIL-direction-washes-out the value is in its SYMMETRIC part; directedness adds ~0
  PASS-separable-only       adds value, but ~0 on the coincident floor (a real but bounded keeper)
  PASS                      adds value, including on the hard regime
"""
from __future__ import annotations

import numpy as np


def _std(c):
    c = np.nan_to_num(np.asarray(c, dtype=float), posinf=0.0, neginf=0.0)
    s = c.std()
    return (c - c.mean()) / (s if s > 1e-9 else 1.0)


def _r2(cols, y):
    if not cols:
        return 0.0
    with np.errstate(all="ignore"):
        X = np.column_stack([np.ones(len(y))] + [_std(c) for c in cols])
        XtX = X.T @ X
        lam = 1e-6 * np.trace(XtX) / X.shape[1]   # tiny ridge -> stable under collinearity
        beta = np.linalg.solve(XtX + lam * np.eye(X.shape[1]), X.T @ y)
        pred = X @ beta
        denom = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - pred) ** 2).sum() / (denom if denom > 1e-9 else 1.0))


def gate(name, candidate, baseline_cols, y, *, sym_candidate=None,
         coincident_mask=None, tau=0.02):
    """y = per-point false-eat indicator (or danger score). baseline_cols already includes the
    classifier risk + symmetric separability. Returns a verdict dict."""
    y = np.asarray(y, dtype=float)
    r2_base = _r2(baseline_cols, y)
    r2_full = _r2(baseline_cols + [candidate], y)
    inc = r2_full - r2_base
    out = {"signal": name, "incremental_over_baseline": round(inc, 3)}

    dirval = None
    if sym_candidate is not None:
        r2_sym = _r2(baseline_cols + [sym_candidate], y)
        dirval = r2_full - r2_sym
        out["directedness_value"] = round(dirval, 3)

    inc_coin = None
    if coincident_mask is not None and int(np.sum(coincident_mask)) > 5:
        m = coincident_mask
        bc = [np.asarray(c)[m] for c in baseline_cols]
        inc_coin = _r2(bc + [np.asarray(candidate)[m]], y[m]) - _r2(bc, y[m])
        out["incremental_on_coincident"] = round(inc_coin, 3)

    if inc < tau:
        v = "FAIL-redundant"
    elif sym_candidate is not None and dirval < tau:
        v = "FAIL-direction-washes-out"
    elif inc_coin is not None and inc_coin < tau:
        v = "PASS-separable-only"
    else:
        v = "PASS"
    out["verdict"] = v
    return out


def auroc(score, label):
    """Rank-based AUROC; label 1 = positive. No sklearn."""
    score, label = np.asarray(score), np.asarray(label)
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = int((label == 1).sum()), int((label == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
