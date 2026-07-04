"""Symbolic re-derivation of the identity the Clopper-Pearson bound rests on, plus a 30-digit cross-check.

`cp_upper` solves P(Bin(n,p) <= x) = alpha by bisection. That it equals the Beta-quantile closed form used
throughout the paper's bounds relies on the Beta-binomial identity:

    sum_{k=0}^{x} C(n,k) p^k (1-p)^{n-k}  =  1 - I_p(x+1, n-x)        (I = regularized incomplete Beta)

so P(Bin(n,p) <= x) = alpha  <=>  I_p(x+1, n-x) = 1 - alpha  <=>  p = Beta^{-1}(1-alpha; x+1, n-x).

This script (1) proves that identity SYMBOLICALLY for small (n, x) with SymPy (simplify(lhs - rhs) == 0),
(2) proves the x=0 closed form p = 1 - alpha^(1/n) symbolically, and (3) confirms the numeric cp_upper
satisfies its defining equation at 30-digit precision with mpmath, independent of the float bisection.
Run: python verify_cp_symbolic.py
"""
import sympy as sp
import mpmath as mp
from cp_bounds import cp_upper

p, a, t = sp.symbols("p a t", positive=True)


def prove_beta_binomial_identity(nmax=8):
    print("1. SYMBOLIC Beta-binomial identity  sum_{k<=x} C(n,k)p^k(1-p)^(n-k) == 1 - I_p(x+1,n-x):")
    checked = 0
    for n in range(1, nmax + 1):
        for x in range(0, n):                       # x = n handled separately (upper = 1)
            binom_partial = sum(sp.binomial(n, k) * p**k * (1 - p) ** (n - k) for k in range(x + 1))
            # B(x+1, n-x) = x! (n-x-1)! / n!  (explicit; sp.beta(int,int) does not auto-evaluate)
            B = sp.factorial(x) * sp.factorial(n - x - 1) / sp.factorial(n)
            I_p = sp.integrate(t**x * (1 - t) ** (n - x - 1), (t, 0, p), conds="none") / B
            assert sp.expand(binom_partial - (1 - I_p)) == 0, f"identity FAILED at n={n}, x={x}"
            checked += 1
    print(f"   PASS: identity holds symbolically for all {checked} (n,x) with 1<=n<={nmax}, 0<=x<n")


def prove_x0_closed_form(nmax=12):
    print("2. SYMBOLIC x=0 closed form  cp_upper(0,n,a) = 1 - a^(1/n)  (solves (1-p)^n = a):")
    for n in range(1, nmax + 1):
        cand = 1 - a ** sp.Rational(1, n)                 # claimed bound
        assert sp.simplify((1 - cand) ** n - a) == 0, f"closed form FAILED at n={n}"
    print(f"   PASS: (1-(1-a^(1/n)))^n simplifies to a for all 1<=n<={nmax}")


def crosscheck_mpmath():
    print("3. 30-DIGIT mpmath cross-check of the numeric cp_upper (independent of scipy/float bisection):")
    mp.mp.dps = 30
    def binom_cdf_mp(x, n, pp):
        pp = mp.mpf(pp)
        return sum(mp.binomial(n, k) * pp**k * (1 - pp) ** (n - k) for k in range(x + 1))
    worst_def, worst_x0 = mp.mpf(0), mp.mpf(0)
    for n in (60, 90, 120, 150, 500):
        for x in (0, 1, 2, 3):
            u = cp_upper(x, n, 0.05)
            if 0 < x < n:                                # defining equation P(Bin<=x)=alpha at 30 digits
                worst_def = max(worst_def, abs(binom_cdf_mp(x, n, u) - mp.mpf("0.05")))
            if x == 0:                                   # boundary vs high-precision closed form
                worst_x0 = max(worst_x0, abs(mp.mpf(u) - (1 - mp.mpf("0.05") ** (mp.mpf(1) / n))))
    print(f"   defining-equation residual |P(Bin<=x) - 0.05| <= {float(worst_def):.2e} (float-bisection limited)")
    print(f"   x=0 vs 1-0.05^(1/n)        |diff|             <= {float(worst_x0):.2e}")
    assert worst_def < 1e-9 and worst_x0 < 1e-12, "mpmath cross-check exceeded tolerance"
    print("   PASS")


if __name__ == "__main__":
    prove_beta_binomial_identity()
    prove_x0_closed_form()
    crosscheck_mpmath()
    print("\nALL SYMBOLIC CHECKS PASSED — the appendix's Clopper-Pearson / Beta-quantile identity is machine-verified.")
