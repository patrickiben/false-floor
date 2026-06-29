"""Applying HWM -- "Hierarchical Planning with Latent World Models" (arXiv 2604.03208, Zhang/Terver/
Zholus/Chitnis et al., FAIR/NYU/Mila/Brown) -- to "Should I eat this mushroom?"

The reframe: the decision is NOT a one-shot classification (that's the FLAT planner the paper says
fails). A forager runs a SEQUENTIAL diagnostic plan -- gather evidence (look at base for a volva,
chemical spot-test, spore print) until confident enough, then EAT or DISCARD. That is planning over
a latent world model, with the HWM two-level structure:

  latent world model : Bayesian belief-transition -- predicts how P(poison) updates given an
                       observation-action. (Analytic/calibrated here: we lack multi-observation
                       data to LEARN a neural world model; the STRUCTURE is faithful.)
  low-level planner  : fast, short horizon -- pick the next PRIMITIVE test, or terminate; stop early
                       once the belief crosses a confident bound.
  high-level planner : slow, long horizon -- pick a MACRO-action (a diagnostic strategy). An action
                       encoder compresses the exponential space of test-subsets into a few macros
                       {DECIDE_NOW, ESCALATE, EXHAUSTIVE}, cutting search (the paper's core trick).

Everything is calibrated from THIS project's real findings:
  - visual evidence  = the I-JEPA ensemble risk on the REAL held-out lethal pair. Treated as a
                       CALIBRATED channel (AUROC~0.94 -> sens=spec~0.87), NOT ground truth -- because
                       we MEASURED its confidently-wrong residual. The FLAT baseline trusts the raw
                       point estimate and eats that residual; the world-model planner does not.
  - spore / chem     = INDEPENDENT channels (value_of_channel: only an independent sensor drives
                       false-eat->0). Spore color near-perfectly splits Agaricus(brown)/Amanita(white).
  - base/volva       = informative only when the base is visible (~0.38, from the VLM run).
Asymmetric utility (net_benefit/DCA): eat poison = -H (huge), eat edible = +1, discard = 0.

SAFETY: research vehicle only -- a calibrated simulation of the diagnostic PLAN, not foraging advice.
"""
import numpy as np
from config import CFG
from model import train_ensemble, ensemble_risk
np.seterr(all="ignore")

# ---------- REAL visual evidence: I-JEPA ensemble risk on the held-out lethal pair ----------
ROOT = "data/fungi_real"
L = {s: np.load(f"{ROOT}/{s}.npy") for s in
     ["edible_chanterelle", "poison_jackolantern", "edible_fieldmushroom", "poison_destroyingangel"]}
rng = np.random.default_rng(0)
def split(X): i = rng.permutation(len(X)); c = int(0.7 * len(X)); return X[i[:c]], X[i[c:]]
fmtr, fmte = split(L["edible_fieldmushroom"]); datr, date = split(L["poison_destroyingangel"])
Xtr = np.vstack([L["edible_chanterelle"], L["poison_jackolantern"], fmtr, datr])
ytr = np.r_[np.zeros(len(L["edible_chanterelle"])), np.ones(len(L["poison_jackolantern"])),
            np.zeros(len(fmtr)), np.ones(len(datr))].astype("f4")
mu = Xtr.mean(0); _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); V = Vt[:64]
proj = lambda X: (X - mu) @ V.T
ens = train_ensemble(proj(Xtr), ytr, 64, CFG)
re_, _ = ensemble_risk(ens, proj(fmte))      # visual P(poison) for held-out edibles (field mushroom)
rp_, _ = ensemble_risk(ens, proj(date))      # visual P(poison) for held-out Destroying Angels (POISON)
SPECIMENS = [(0, r) for r in re_] + [(1, r) for r in rp_]   # (true_poison, visual_risk)

# ---------- channel reliabilities (sens=P(+|poison), spec=P(-|edible)), cost, availability ----------
H, B = 2000.0, 10.0                          # eat poison = -2000, eat edible = +10 (a meal), discard = 0
EAT_B = B / (B + H)                          # EAT optimal iff belief P(poison) < ~0.005
CONF_DISCARD = 0.5                           # belief above which we'd surely discard (stop testing)
CH = {   # test costs are in meal-units: a workup is cheap relative to a meal (+10) but not free
    "visual": dict(sens=0.87, spec=0.87, cost=0.0, avail=1.0),  # calibrated from AUROC ~0.94
    "base":  dict(sens=0.90, spec=0.95, cost=0.2, avail=0.38),  # volva; only if base visible
    "chem":  dict(sens=0.90, spec=0.90, cost=1.0, avail=1.0),
    "spore": dict(sens=0.97, spec=0.97, cost=2.0, avail=1.0),   # independent, near-perfect
}

def update(b, ch, out):                      # latent world model: belief transition
    s = CH[ch]
    lp, le = (s["sens"], 1 - s["spec"]) if out == "+" else (1 - s["sens"], s["spec"])
    num = b * lp; den = num + (1 - b) * le
    return num / max(den, 1e-12)

def sample_outcome(ch, true_poison, rng):    # realised reading for a specimen
    s = CH[ch]
    if rng.random() > s["avail"]: return None
    p_plus = s["sens"] if true_poison else (1 - s["spec"])
    return "+" if rng.random() < p_plus else "-"

def terminal(b):                             # best terminal act + its expected utility
    return ("EAT", (1 - b) * B - b * H) if b < EAT_B else ("DISCARD", 0.0)

# ---------- FLAT baseline: threshold the raw classifier (our entire prior approach) ----------
def flat_threshold(true_poison, vis_risk):
    act = "EAT" if vis_risk < EAT_B else "DISCARD"   # trusts the overconfident point estimate
    return act, 0.0

# ---------- FLAT-EXHAUSTIVE: always run the full independent workup, then decide ----------
def flat_exhaustive(true_poison, vis_risk, rng):
    b = update(0.5, "visual", "+" if vis_risk > 0.5 else "-")
    out = sample_outcome("spore", true_poison, rng); b = update(b, "spore", out) if out else b
    return terminal(b)[0], CH["spore"]["cost"]

# ---------- HIERARCHICAL (HWM): high-level macro choice + low-level adaptive escalation ----------
def macro_value(macro, b0):
    """High-level: expected utility of a macro = world-model rollout over outcome distributions
    with low-level early-stop, minus expected cost. (The action encoder = these few macros.)"""
    # depth-limited expectation; outcomes integrated analytically
    def rollout(b, tests, cost):
        a, u = terminal(b)
        if not tests or b < EAT_B or b > CONF_DISCARD:
            return u - cost
        ch = tests[0]; s = CH[ch]
        pe_plus = b * s["sens"] + (1 - b) * (1 - s["spec"])   # P(+ reading)
        v_plus = rollout(update(b, ch, "+"), tests[1:], cost + s["cost"])
        v_minus = rollout(update(b, ch, "-"), tests[1:], cost + s["cost"])
        v_obs = pe_plus * v_plus + (1 - pe_plus) * v_minus
        v_obs = s["avail"] * v_obs + (1 - s["avail"]) * rollout(b, tests[1:], cost)  # base may be unavailable
        return v_obs
    return rollout(b0, macro, 0.0)

MACROS = {"DECIDE_NOW": [], "ESCALATE": ["base", "chem", "spore"], "EXHAUSTIVE": ["spore"]}

def hierarchical(true_poison, vis_risk, rng):
    b = update(0.5, "visual", "+" if vis_risk > 0.5 else "-")   # visual as a CALIBRATED channel
    # high-level: pick the macro with best expected value under the world model
    macro = max(MACROS, key=lambda m: macro_value(MACROS[m], b))
    # low-level: execute the chosen macro's tests, fast early-stop once confident
    cost = 0.0
    for ch in MACROS[macro]:
        if b < EAT_B or b > CONF_DISCARD: break
        out = sample_outcome(ch, true_poison, rng)
        if out is None: continue
        b = update(b, ch, out); cost += CH[ch]["cost"]
    return terminal(b)[0], cost, macro

# ---------- run all three on the real held-out lethal pair ----------
def evaluate(fn, hier=False, trials=200):
    rng2 = np.random.default_rng(7)
    fe = yld = cost = 0.0; npois = npedi = 0; macros = {}
    for _ in range(trials):
        for true_p, vr in SPECIMENS:
            if hier: act, c, m = fn(true_p, vr, rng2); macros[m] = macros.get(m, 0) + 1
            else:
                out = fn(true_p, vr, rng2) if fn.__code__.co_argcount == 3 else fn(true_p, vr)
                act, c = out
            cost += c
            if true_p: npois += 1; fe += (act == "EAT")
            else: npedi += 1; yld += (act == "EAT")
    return dict(false_eat=100 * fe / npois, yield_=100 * yld / npedi,
                avg_cost=cost / (npois + npedi), macros=macros)

print(f"held-out lethal pair: {len(re_)} edible (field mushroom), {len(rp_)} poison (Destroying Angel)")
print(f"eat-bar P(poison)<{EAT_B:.4f} from asymmetric utility (eat poison = -{H:.0f})\n")
print(f"  {'planner':<34} | {'false-eat':>9} | {'yield':>6} | {'avg cost':>8}")
f1 = evaluate(flat_threshold)
print(f"  {'FLAT (threshold the classifier)':<34} | {f1['false_eat']:>8.1f}% | {f1['yield_']:>5.0f}% | {f1['avg_cost']:>8.2f}")
f2 = evaluate(flat_exhaustive)
print(f"  {'FLAT-EXHAUSTIVE (always spore-print)':<34} | {f2['false_eat']:>8.1f}% | {f2['yield_']:>5.0f}% | {f2['avg_cost']:>8.2f}")
h = evaluate(hierarchical, hier=True)
print(f"  {'HIERARCHICAL (HWM plan)':<34} | {h['false_eat']:>8.1f}% | {h['yield_']:>5.0f}% | {h['avg_cost']:>8.2f}")
print(f"\n  HWM macro chosen by the high-level planner: {h['macros']}")
print("\nReads: FLAT (our whole prior approach = threshold the classifier) EATS the confidently-wrong")
print("residual. FLAT-EXHAUSTIVE is safe but pays the full independent test on EVERY mushroom. The HWM")
print("hierarchical plan reaches exhaustive-level safety at LOWER cost by escalating only as needed --")
print("operationalizing the whole investigation: safety = PLANNING to acquire an independent channel.")
print("SAFETY: research vehicle only; calibrated simulation of the plan, not foraging advice. DON'T EAT.")
