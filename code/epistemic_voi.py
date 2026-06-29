"""(1) A PROPER value-of-information epistemic term for Active Inference -- the fix the panel demanded.

The previous epistemic term used Beta VARIANCE, which is MAXIMIZED at mean 0.5, so it pulled the
planner TOWARD mediocre decoys (uncertain but clearly not worth it -- that is why it hurt). The
correct quantity is the EXPECTED VALUE OF PERFECT INFORMATION (EVPI): how much would learning a
patch's true edible-fraction improve FUTURE decisions? EVPI is high ONLY when the uncertainty
STRADDLES the decision threshold tau (an unknown patch that MIGHT be worth foraging), and ~0 for a
patch confidently known to be mediocre -- exactly the term that should drive principled exploration.

Honest, FAIR test (NO hierarchy claims -- the panel killed that): one fixed FLAT planner, compared
WITH vs WITHOUT each epistemic term, on the SAME world model. Generic landscapes + 6-day persistent
beliefs = the regime where learning can actually pay. The ablation is the whole point. SAFETY: research vehicle.
"""
import numpy as np
from math import lgamma
import forage_expedition as FE
np.seterr(all="ignore")

GRID = np.linspace(0.002, 0.998, 250)


def evpi(a, b, tau, future_steps):
    """Expected value of perfect info about g~Beta(a,b): E[harvest you'd gain/avoid by knowing g],
    scaled by the future forage-steps the knowledge enables. High only when uncertainty straddles tau."""
    logB = lgamma(a) + lgamma(b) - lgamma(a + b)
    pdf = np.exp((a - 1) * np.log(GRID) + (b - 1) * np.log(1 - GRID) - logB); pdf /= pdf.sum()
    mean = float((GRID * pdf).sum())
    if mean < tau:
        gain = float(((GRID - tau) * (GRID > tau) * pdf).sum())   # currently exclude; might be GOOD
    else:
        gain = float(((tau - GRID) * (GRID < tau) * pdf).sum())   # currently include; might be BAD
    return gain * future_steps


def plan(env, belief, pos, t, mode, epi_w, days_left, tau, ts):
    a, b = belief; best, bi = -1e9, None
    for i in range(env.P):
        if env.abund[i] <= 0: continue
        travel = env.travel_time(pos, i)
        stay = min(int(env.abund[i]), 8, int(t - travel))
        if stay <= 0: continue
        g = float(ts.beta(a[i], b[i])) if mode == "thompson" else a[i] / (a[i] + b[i])
        val = stay * (g * env.MEAL - env.TESTCOST)                 # fair total-value objective (no rate)
        if mode == "evpi":
            val += epi_w * evpi(a[i], b[i], tau, 8) * max(days_left - 1, 0)   # info pays on FUTURE days
        elif mode == "betavar":
            val += epi_w * FE.belief_uncertainty(a[i], b[i]) * stay * days_left
        if val > best: best, bi = val, (i, stay)
    return bi


def run_day(env, belief, mode, epi_w, days_left, tau, rng, ts):
    pos = -1; t = env.T; total = 0.0; pois = 0
    while t > 0:
        ch = plan(env, belief, pos, t, mode, epi_w, days_left, tau, ts)
        if ch is None: break
        i, stay = ch
        tt = env.travel_time(pos, i)
        if tt + 1 > t: break
        t -= tt
        r, used, p = FE.forage_patch(env, i, belief, t, rng, stay)
        total += r; pois += p; t -= used; pos = i
    return total, pois


def evaluate(mode, epi_w=0.0, tau=0.5, n_land=60, days=6, structured=False):
    tot = pois = 0.0; ts = np.random.default_rng(7)
    for L in range(n_land):
        env = FE.Landscape(seed=L, structured=structured); belief = FE.init_belief(env.P)
        rng = np.random.default_rng(1000 + L)
        for d in range(days):
            env.abund = FE.Landscape(seed=L, structured=structured).abund.copy()
            v, p = run_day(env, belief, mode, epi_w, days - d, tau, rng, ts)
            tot += v; pois += p
    return tot / (n_land * days), pois / (n_land * days)


def regime(name, structured):
    print(f"\n== {name} ==")
    print(f"  {'epistemic mode':<36} | {'value/day':>9} | {'poison/day':>10}")
    bv, bp = evaluate("none", structured=structured)
    print(f"  {'NONE (pragmatic / mean belief)':<36} | {bv:>9.2f} | {bp:>10.3f}")
    for label, mode, w in [("Beta-variance (the broken term)", "betavar", 6.0),
                           ("Thompson sampling", "thompson", 0.0),
                           ("EVPI (proper value-of-information)", "evpi", 1.0)]:
        v, p = evaluate(mode, epi_w=w, structured=structured)
        flag = "  <- HELPS" if v > bv + 0.2 else ("  <- hurts" if v < bv - 0.2 else "  ~neutral")
        print(f"  {label:<36} | {v:>9.2f} | {p:>10.3f}{flag}")


if __name__ == "__main__":
    print("PROPER value-of-information epistemic term -- fair FLAT planner, WITH vs WITHOUT (no hierarchy claims)")
    regime("GENERIC random landscapes (learning can pay)", False)
    regime("STRUCTURED (rich-far / decoy-near)", True)
    print("\nablation = does EVPI beat NONE robustly (and beat the broken Beta-variance term)?")
    print("if not, Active Inference's epistemic term is INERT on this problem -- and we report THAT (step 3).")
