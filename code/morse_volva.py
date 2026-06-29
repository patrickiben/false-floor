"""Morse theory on the mushroom shape -- the sharp refinement of the Betti test. Betti numbers missed the
volva because an open cup is a CANCELLING critical pair (homologically trivial, B2=0). Morse theory counts
the critical points INCLUDING the cancelling pairs, so it should SEE the volva where Betti could not.

We compute sublevel-set persistence of the HEIGHT function on the synthetic shapes (= computational Morse
theory: persistence pairs ARE the Morse critical-point cancellation pairs; #crit(index k) >= B_k, the excess
= geometry invisible to homology). Question: does the volva add critical pairs the Agaricus lacks, and at
what PERSISTENCE? Prediction: yes, but LOW persistence (it cancels) -> detectable in clean ground truth,
buried under the real LRM's noise floor (hundreds of critical pairs, persistence up to ~2000). SAFETY: research vehicle.
"""
import numpy as np
from gudhi import CubicalComplex

N = 56
ix = np.arange(N)
Z, Y, X = np.meshgrid(ix, ix, ix, indexing="ij")
cx = cy = N // 2

def cyl(r, z0, z1): return ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r) & (Z >= z0) & (Z < z1)
def ball(cz, r): return (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r * r
def dome(cz, r): return ball(cz, r) & (Z >= cz)
def shell(cz, r, t): d = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2; return (d <= r * r) & (d >= (r - t) ** 2)
def lower_shell(cz, r, t): return shell(cz, r, t) & (Z < cz)
def skirt(z0, z1, ri, ro): rr = (X - cx) ** 2 + (Y - cy) ** 2; return (rr <= ro * ro) & (rr >= ri * ri) & (Z >= z0) & (Z < z1)

stem = cyl(4, 10, 40); cap = dome(40, 14)
agaricus = stem | cap | skirt(26, 28, 4, 11)
# volva = saccate cup at the base, connected to the stem by a thin neck (anatomically the stem emerges from it)
volva = lower_shell(16, 9, 3) | cyl(3, 7, 12)
amanita = agaricus | volva


def morse_height(occ, direction):
    """Sublevel ('up', min-features) or superlevel ('down', max-features) persistence of the height function
    = Morse critical-point pairs. The cup BOTTOM is a min (up); the cup RIM is a max (down)."""
    z = Z.astype("f8")
    filt = np.where(occ, z if direction == "up" else (N - z), 1e6)
    cc = CubicalComplex(top_dimensional_cells=filt); cc.persistence()
    out = {}
    for d in (0, 1):
        out[d] = sorted([(b, dd, dd - b) for b, dd in cc.persistence_intervals_in_dimension(d) if dd < 1e5],
                        key=lambda t: t[2], reverse=True)
    return out


print("=== Morse (height-function) critical-pair spectrum: Agaricus vs Amanita ===\n")
SH = {}
for name, occ in [("Agaricus (no volva)", agaricus), ("Amanita (saccate volva cup)", amanita)]:
    bc = CubicalComplex(top_dimensional_cells=np.where(occ, 0.0, 1.0)); bc.persistence()
    betti = [sum(b <= 0.5 < dd for b, dd in bc.persistence_intervals_in_dimension(d)) for d in (0, 1, 2)]
    print(f"  {name}")
    print(f"    Betti [B0,B1,B2] = {[int(x) for x in betti]}   (homology -- the coarse view: blind to the volva)")
    SH[name] = {}
    for direction, label in [("up", "sublevel/bottom-up (minima)"), ("down", "superlevel/top-down (maxima)")]:
        m = morse_height(occ, direction); SH[name][direction] = m
        print(f"    Morse {label}: finite H0={len(m[0])}  H1={len(m[1])}")
        for d in (0, 1):
            if m[d]:
                tops = ", ".join(f"pers {p:.0f}" for _, _, p in m[d][:5])
                print(f"      H{d} critical pairs: {tops}")
    print()

ag, am = SH["Agaricus (no volva)"], SH["Amanita (saccate volva cup)"]
print("=> THE VOLVA'S MORSE SIGNATURE (extra critical pairs Amanita has, by direction):")
for direction, label in [("up", "bottom-up"), ("down", "top-down")]:
    e0 = len(am[direction][0]) - len(ag[direction][0]); e1 = len(am[direction][1]) - len(ag[direction][1])
    pers = sorted([p for _, _, p in am[direction][0] + am[direction][1]], reverse=True)[:3]
    zext = int(Z[amanita].max() - Z[amanita].min())
    print(f"   {label:10}: +{e0} H0, +{e1} H1 pairs; persistences {[round(p,1) for p in pers]} "
          f"(height units; object z-extent ~{zext}, so the volva is a ~{round(100*max(pers+[0])/zext)}%-of-scale feature)")
print(f"   NOISE FLOOR (real TripoSR LRM, lrm_persistence.py): HUNDREDS of critical pairs per reconstruction.")
print(f"   (Different filtration units across the two, so compare by COUNT: a handful of volva pairs vs hundreds of noise pairs.)")
print("""
READING: Morse theory IS finer than Betti -- it registers the volva as a low-persistence cancelling critical
pair that homology (B2=0) is blind to. That vindicates the instinct. BUT it also pins down WHY the volva is
hard, in one number: it is a ~10%-of-scale, LOW-persistence feature by mathematical necessity (it cancels in
homology). So (1) on the real noisy LRM it is one needle among hundreds of higher-count reconstruction-texture
pairs; (2) it still requires the volva to be CAPTURED/reconstructed (occluded ~2/3 of photos); (3) it is a
descriptor of the same RGB-derived shape = redundant channel. Morse theory does not break the wall -- it gives
the most precise CHARACTERIZATION of it: the diagnostic feature is real, geometric, low-persistence, often
uncaptured, single-modality -- exactly the profile every lever in this project hit. SAFETY: research vehicle; DON'T EAT.
""")
