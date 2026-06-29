"""(B) The HWM-shaped problem + a GENUINE Active-Inference push: a long-horizon foraging EXPEDITION.

This is the version where hierarchy/planning MUST do real work (unlike the inert hwm_forager). The
agent has a day's time budget on a landscape of spatial patches with UNKNOWN edible-fractions. It must
plan a ROUTE (which patches, in what order -- combinatorial, long-horizon) while a shared low-level
policy handles per-mushroom safety. Patches DEPLETE as they are foraged, so camping a single rich spot
is impossible -- value comes from the route. Greedy routing fails in the structured regime (myopic: it
grabs nearer decent patches and won't pay for the long trip to the rich far CLUSTER that one planned
route amortizes over many shallow harvests). The eat/don't decision is the LOW-LEVEL primitive.

Active Inference pushed for real: the high-level minimizes EXPECTED FREE ENERGY over the horizon =
  pragmatic (expected edible harvest)  +  EPISTEMIC (info gain about uncertain patches -- the term we
  never implemented)  -  travel cost.   Beliefs (Beta over each patch's edible-fraction) persist and
update ACROSS days, so the epistemic term drives explore-then-exploit.

ANTI-RIGGING DISCIPLINE (the lesson from the rigged hwm_forager):
  - all planners share the SAME world model / beliefs / low-level policy (no representational asymmetry)
  - PRIMARY OUTPUT is the ablation: deleting the high-level planner MUST degrade value, and removing
    the epistemic term MUST hurt -- if they don't, the hierarchy/AI is inert and we report that.
  - the data-generating g_i are NOT shown to any planner; everyone plans from the same Beta beliefs.
SAFETY: research vehicle; the per-mushroom safety is the shared low-level, not the point of this demo.
"""
import numpy as np


class Landscape:
    def __init__(self, P=10, T=70.0, seed=0, structured=False):
        r = np.random.default_rng(seed)
        self.start = np.array([0.5, 0.5])
        self.MEAL, self.POISON, self.TESTCOST = 1.0, 30.0, 0.15
        if structured:
            # A genuine CLUSTERED-ORIENTEERING instance (the P arg is ignored -- a fixed designed map).
            # Patches DEPLETE as foraged (forage_patch decrements abund), so camping is impossible and
            # value must come from ROUTING. The trap for greedy: the rich patches sit together in ONE
            # far corner but are SHALLOW, so a single long trip is amortized over many small harvests.
            # Greedy scores each patch ALONE, so the first cluster patch looks worse-per-time than a
            # near isolated medium -- greedy grabs the mediums and never pays for the trip; the planner
            # evaluates the whole route and commits. (This is why greedy fails ONLY once depletion is
            # enforced: without it, both planners just camp the single best patch and the gap is fake.)
            P = 12
            self.T, self.TRAVEL = 66.0, 26.0            # tight budget + costly travel: the trip must pay off
            self.pos   = r.uniform(0.42, 0.58, (P, 2))  # near DECOYS crowd the start (0.5,0.5)...
            self.g     = r.uniform(0.12, 0.35, P)       # ...mediocre and shallow
            self.abund = r.integers(4, 7, P)
            idx = r.permutation(P)
            cl = idx[:5]                                 # 5 RICH but SHALLOW patches packed in a far corner
            self.pos[cl] = np.array([0.92, 0.92]) + r.uniform(-0.05, 0.05, (5, 2))
            self.g[cl] = 0.90; self.abund[cl] = 3        # each drains in ~one visit -> no camping, must route
            md = idx[5:8]                                # 3 isolated MEDIUMS mid-range: higher solo rate (greedy bait)
            for k, m in enumerate(md):
                ang = 2 * np.pi * k / 3
                self.pos[m] = np.clip(self.start + 0.34 * np.array([np.cos(ang), np.sin(ang)]), 0.02, 0.98)
            self.g[md] = 0.55; self.abund[md] = 6
            self.pos = np.clip(self.pos, 0.0, 1.0)
        else:                                            # GENERIC random landscape (expected greedy-solvable)
            self.T, self.TRAVEL = T, 14.0               # time = TRAVEL * distance
            self.pos   = r.random((P, 2))               # patch locations in the unit square
            self.g     = r.uniform(0.10, 0.90, P)       # TRUE edible-fraction (hidden from planners)
            self.abund = r.integers(6, 18, P)           # forage-steps before a patch is exhausted (now ENFORCED)
        self.P = P

    def travel_time(self, a, b):
        pa = self.pos[a] if a >= 0 else self.start
        pb = self.pos[b] if b >= 0 else self.start
        return self.TRAVEL * float(np.linalg.norm(pa - pb))


def low_level_forage(true_g, rng):
    """Shared, fair per-mushroom step: a mushroom is edible w.p. g. Low-level keeps it safe via a
    test when the visual is ambiguous (cost), eats edibles (+MEAL), avoids poison (rare residual)."""
    edible = rng.random() < true_g
    vis_ok = rng.random() < (0.85 if edible else 0.15)     # 0.85-reliable visual cue
    reward, cost, ate_poison = 0.0, 0.0, False
    if vis_ok:                                              # looks edible -> confirm then eat
        cost += 0.15                                        # TESTCOST: independent confirm
        confirm = rng.random() < (0.97 if edible else 0.03)
        if confirm:
            if edible: reward += 1.0
            else: reward -= 30.0; ate_poison = True         # residual: test missed it (~3%)
    return reward, cost, edible, ate_poison


# ---------------- world model: Beta beliefs over each patch's edible-fraction ----------------
def init_belief(P): return np.ones(P), np.ones(P)          # alpha, beta (uniform prior)
def belief_mean(a, b): return a / (a + b)
def belief_uncertainty(a, b):                              # Beta variance: a tractable uncertainty proxy
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


def forage_patch(env, i, belief, t_left, rng, max_steps):
    """Spend up to max_steps foraging patch i; DEPLETE the patch as steps are consumed -- abundance
    persists across visits within a day (replenished each morning in evaluate), so a re-visited patch
    yields less and an exhausted one (abund<=0) yields nothing. Update belief; return
    (reward, time_used, poison)."""
    a, b = belief
    reward = tused = 0.0; pois = 0; steps = 0
    while steps < max_steps and tused < t_left and env.abund[i] > 0:
        r, c, edible, ap = low_level_forage(env.g[i], rng)
        reward += r - c; tused += 1.0; pois += ap
        a[i] += edible; b[i] += (1 - edible)                # belief update from what we observe
        env.abund[i] -= 1                                   # the patch depletes as we harvest it
        steps += 1
    return reward, tused, pois


# ---------------- exploration modes: the genuine Active-Inference push ----------------
VMODE = "mean"          # 'mean' (pragmatic) | 'thompson' (sample posterior) | 'infogain' (EFE epistemic)
EPI_W = 0.0             # info-gain weight
DAYS_LEFT = 1           # remaining horizon -> info worth more early (horizon-weighted value-of-information)
TS = np.random.default_rng(0)


def g_estimate(belief):                                    # per-patch edible estimate, sampled ONCE per call
    a, b = belief
    return TS.beta(a, b) if VMODE == "thompson" else a / (a + b)


def info_gain_value(belief, i, steps):                     # EFE epistemic term: horizon-weighted value-of-info
    if VMODE != "infogain": return 0.0
    return EPI_W * DAYS_LEFT * belief_uncertainty(belief[0][i], belief[1][i]) * steps


# ---------------- planners (all share env + belief + modes; differ ONLY in routing) ----------------
def patch_value(env, i, gv, belief, stay):
    return stay * (gv[i] * env.MEAL - env.TESTCOST) + info_gain_value(belief, i, stay)


def run_day(env, belief, planner, rng, days_left=1):
    global DAYS_LEFT; DAYS_LEFT = days_left
    pos = -1; t = env.T; total = 0.0; pois = 0; visited = set()
    while t > 0:
        choice = planner(env, belief, pos, t, visited)
        if choice is None: break
        i, stay = choice
        tt = env.travel_time(pos, i)
        if tt + 1 > t: break
        t -= tt
        r, used, p = forage_patch(env, i, belief, t, rng, stay)
        total += r; pois += p; t -= used; pos = i; visited.add(i)
    return total, pois


def greedy(env, belief, pos, t, visited):
    """FLAT/myopic: best immediate value-per-time (incl. travel), ignore the horizon."""
    gv = g_estimate(belief); best, bi = -1e9, None
    for i in range(env.P):
        if i in visited and env.abund[i] <= 0: continue
        stay = min(int(env.abund[i]), 8)
        tt = env.travel_time(pos, i)
        if tt + 1 > t: continue
        rate = patch_value(env, i, gv, belief, stay) / (tt + stay + 1e-9)
        if rate > best: best, bi = rate, (i, stay)
    return bi


def efe_planner(env, belief, pos, t, visited):
    """HIERARCHICAL (orienteering): route maximizing TOTAL value within the budget (the non-greedy
    objective; greedy-by-rate provably fails). Exploration enters via shared g_estimate / info_gain."""
    import itertools
    gv = g_estimate(belief)
    cands = [i for i in range(env.P) if not (i in visited and env.abund[i] <= 0)]
    top = sorted(cands, key=lambda i: patch_value(env, i, gv, belief, min(int(env.abund[i]), 8)),
                 reverse=True)[:6]                          # prune to the 6 most promising patches

    def total_harvest(route):                              # total expected value the route earns within budget
        v = 0.0; p = pos; tt = t
        for i in route:
            travel = env.travel_time(p, i)
            if travel + 1 > tt: break
            steps = min(int(env.abund[i]), 8, int(tt - travel))
            if steps <= 0: break
            v += patch_value(env, i, gv, belief, steps)
            tt -= travel + steps; p = i
        return v

    best, bi = -1e9, None
    for ln in (1, 2, 3):                                   # search short routes (orienteering)
        for route in itertools.permutations(top, ln):
            v = total_harvest(route)
            if v > best: best, bi = v, route[0]
    return None if bi is None else (bi, min(int(env.abund[bi]), 8))


# ---------------- evaluate over many landscapes x days, with the ablations ----------------
def evaluate(planner, mode="mean", epi_w=0.0, n_land=40, days=6, structured=False):
    global VMODE, EPI_W, TS
    VMODE = mode; EPI_W = epi_w; TS = np.random.default_rng(123)
    tot = pois = 0.0
    for L in range(n_land):
        env = Landscape(seed=L, structured=structured); belief = init_belief(env.P)
        rng = np.random.default_rng(1000 + L)
        for d in range(days):
            env.abund = Landscape(seed=L, structured=structured).abund.copy()   # replenish each day
            v, p = run_day(env, belief, planner, rng, days_left=days - d)
            tot += v; pois += p
    return tot / (n_land * days), pois / (n_land * days)


def regime(name, structured):
    print(f"\n========== {name} ==========")
    print(f"  {'planner':<50} | {'value/day':>9} | {'poison/day':>10}")
    g_v, g_p = evaluate(greedy, mode="mean", structured=structured)
    print(f"  {'GREEDY routing (flat / myopic)':<50} | {g_v:>9.2f} | {g_p:>10.3f}")
    p_v, _ = evaluate(efe_planner, mode="mean", structured=structured)
    print(f"  {'HIERARCHICAL, pragmatic-only (mean belief)':<50} | {p_v:>9.2f} |")
    th_v, th_p = evaluate(efe_planner, mode="thompson", structured=structured)
    print(f"  {'HIERARCHICAL + Thompson (posterior-sample explore)':<50} | {th_v:>9.2f} | {th_p:>10.3f}")
    ig_v, ig_p = evaluate(efe_planner, mode="infogain", epi_w=0.5, structured=structured)
    print(f"  {'HIERARCHICAL + info-gain EFE (horizon-weighted)':<50} | {ig_v:>9.2f} | {ig_p:>10.3f}")
    print(f"  ablation -> hierarchy:   greedy {g_v:.2f} vs hier {p_v:.2f}  "
          f"[{'HIERARCHY HELPS' if p_v > g_v + 0.3 else 'inert/worse'}]")
    best_ex = max(th_v, ig_v)
    print(f"  ablation -> exploration: pragmatic {p_v:.2f} vs best-explore {best_ex:.2f}  "
          f"[{'EXPLORATION HELPS = active inference earns its keep' if best_ex > p_v + 0.2 else 'exploration inert/hurts (honest negative)'}]")


if __name__ == "__main__":
    print("foraging expedition -- value/day (higher=better); fair shared world model; ablation = honesty test")
    print("(6-day horizon, beliefs persist across days -> genuine room for explore-then-exploit)")
    regime("GENERIC random landscapes (expected greedy-solvable)", structured=False)
    regime("NON-GREEDY clustered orienteering (rich-far CLUSTER / near decoys)", structured=True)
    print("\nSAFETY: research vehicle; shared low-level safety, not the point. Honest ablations are the point.")
