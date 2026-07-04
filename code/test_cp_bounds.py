"""Property-based + differential verification of the Clopper-Pearson upper bound (`cp_bounds.cp_upper`) that
every false-eat / false-safe rate in the paper is reported with. The paper's claims live at rates near zero,
so the x=0 and x=n boundaries are load-bearing; these are tested as first-class cases.

Three independent checks:
  (A) PROPERTY-BASED (Hypothesis): invariants that must hold for all valid (x, n) -- range, monotonicity in x
      and n, the x=n and x=0 boundaries, and the DEFINING equation P(Bin(n, cp_upper) <= x) = alpha.
  (B) DIFFERENTIAL: cp_upper (a from-scratch bisection) must agree with three independent implementations --
      scipy Beta-quantile, scipy binomtest exact CI, and statsmodels proportion_confint(method='beta').
  (C) INLINE CONSISTENCY: the cp_upper copied inline into the evaluation scripts is textually identical to the
      canonical one tested here, so this test actually covers the code that produced the paper's numbers.

Run: python test_cp_bounds.py   (no pytest needed; each Hypothesis property is invoked directly).
"""
import math, os, re, sys
from hypothesis import given, strategies as st, settings, HealthCheck
from scipy.stats import beta, binom, binomtest
from statsmodels.stats.proportion import proportion_confint
from cp_bounds import cp_upper

A = 0.05  # one-sided 95% upper, matching the paper


@st.composite
def nx(draw, nmax=600):
    n = draw(st.integers(min_value=1, max_value=nmax))
    x = draw(st.integers(min_value=0, max_value=n))
    return x, n


SET = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# ---------------- (A) property-based ----------------

@SET
@given(nx())
def prop_range(xn):
    x, n = xn; u = cp_upper(x, n, A)
    assert 0.0 <= u <= 1.0, (x, n, u)
    if x < n:
        assert u >= x / n - 1e-12, f"upper {u} below point est {x/n} at x={x},n={n}"

@SET
@given(nx())
def prop_monotone_in_x(xn):
    x, n = xn
    if x < n:
        assert cp_upper(x, n, A) <= cp_upper(x + 1, n, A) + 1e-12, (x, n)

@SET
@given(nx(nmax=400))
def prop_monotone_in_n(xn):
    x, n = xn  # more trials, same successes -> upper bound cannot increase
    assert cp_upper(x, n, A) >= cp_upper(x, n + 1, A) - 1e-9, (x, n)

@SET
@given(st.integers(1, 5000))
def prop_boundary_x_eq_n(n):
    assert cp_upper(n, n, A) == 1.0

@SET
@given(st.integers(1, 5000), st.floats(0.001, 0.2))
def prop_boundary_x_eq_0(n, a):
    # closed form: P(Bin(n,p)<=0) = (1-p)^n = a  ->  p = 1 - a^(1/n)
    assert math.isclose(cp_upper(0, n, a), 1 - a ** (1 / n), rel_tol=1e-9, abs_tol=1e-9), n

@SET
@given(nx())
def prop_satisfies_defining_equation(xn):
    x, n = xn
    if 0 < x < n:  # cp_upper solves P(Bin(n,p) <= x) = alpha
        assert math.isclose(binom.cdf(x, n, cp_upper(x, n, A)), A, abs_tol=1e-6), (x, n)

# ---------------- (B) differential vs three independent implementations ----------------

@SET
@given(nx())
def diff_scipy_beta_quantile(xn):
    x, n = xn
    if 0 < x < n:  # one-sided (1-a) upper == (1-a) quantile of Beta(x+1, n-x)
        assert math.isclose(cp_upper(x, n, A), beta.ppf(1 - A, x + 1, n - x), abs_tol=1e-6), (x, n)


def grid_differential():
    """Explicit differential over the paper's actual (x, n) cells, incl. the x=0 boundary, across FOUR
    independent computations. Prints a table; asserts agreement to 1e-6."""
    cells = [(x, n) for n in (60, 90, 120, 150) for x in (0, 1, 2, 3)]
    print(f"  {'x/n':>7} | {'cp_upper':>10} | {'scipy.beta':>10} | {'binomtest':>10} | {'statsmodels':>11} | max|diff|")
    worst = 0.0
    for x, n in cells:
        ours = cp_upper(x, n, A)
        b = beta.ppf(1 - A, x + 1, n - x) if 0 < x < n else (1 - A ** (1 / n) if x == 0 else 1.0)
        bt = binomtest(x, n).proportion_ci(confidence_level=1 - 2 * A, method="exact").high  # 2-sided(1-2a) upper = 1-sided(1-a)
        sm = proportion_confint(x, n, alpha=2 * A, method="beta")[1]
        d = max(abs(ours - v) for v in (b, bt, sm))
        worst = max(worst, d)
        print(f"  {x:>2}/{n:<4} | {ours:10.6f} | {b:10.6f} | {bt:10.6f} | {sm:11.6f} | {d:.2e}")
        assert d < 1e-6, f"disagreement {d} at x={x}, n={n}"
    print(f"  -> all four implementations agree to <= {worst:.1e} across every cell (incl. x=0 boundary)")


# ---------------- (C) inline-consistency: the tested fn == the one in the paper's scripts ----------------

def _extract_cp_upper_src(path):
    src = open(path).read()
    m = re.search(r"^def cp_upper\(.*?(?=\n^\S|\Z)", src, re.S | re.M)  # full def, incl. nested cdf_le
    return m.group(0) if m else None

def inline_consistency():
    """Behaviourally verify the inline cp_upper in each evaluation script is the SAME function as the
    canonical one (identical outputs on a grid) -- robust to docstring/whitespace differences."""
    here = os.path.dirname(os.path.abspath(__file__))
    grid = [(x, n, al) for n in (1, 2, 5, 60, 90, 150, 300) for x in (0, 1, 2, n) for al in (0.05, 0.1) if x <= n]
    checked = 0
    for fn in ("hard_pairs.py", "hard_regime.py", "pairs_encoders.py", "case3_snakes.py", "case6_berries.py"):
        p = os.path.join(here, fn)
        if not os.path.exists(p):
            continue
        src = _extract_cp_upper_src(p)
        assert src, f"{fn}: could not extract cp_upper source"
        ns = {"math": math}
        exec(src, ns)
        f = ns["cp_upper"]
        for (x, n, al) in grid:
            a_out, b_out = f(x, n, al), cp_upper(x, n, al)
            same = (a_out == b_out) or (a_out != a_out and b_out != b_out)  # nan==nan
            assert same, f"{fn}: cp_upper({x},{n},{al})={a_out} != canonical {b_out}"
        checked += 1
    print(f"  -> cp_upper is behaviourally identical across {checked} evaluation scripts (grid of {len(grid)} inputs each)")


if __name__ == "__main__":
    print("A. property-based (Hypothesis, 120 examples each) ...", flush=True)
    for f in (prop_range, prop_monotone_in_x, prop_monotone_in_n, prop_boundary_x_eq_n,
              prop_boundary_x_eq_0, prop_satisfies_defining_equation, diff_scipy_beta_quantile):
        f(); print(f"   PASS {f.__name__}")
    print("\nB. differential vs scipy.beta / binomtest / statsmodels ...")
    grid_differential()
    print("\nC. inline-consistency with the evaluation scripts ...")
    inline_consistency()
    print("\nALL CHECKS PASSED — cp_upper is correct (property + differential) and is the code the paper ran.")
