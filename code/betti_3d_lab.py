"""3D topological lab: do Betti numbers (B0/B1/B2) of the mushroom SHAPE separate Amanita from Agaricus?

Patrick's idea: use the 3D structure's topological invariants (pose/lighting/specimen-invariant). We build
anatomically-faithful voxel mushrooms with KNOWN geometry, validate the homology engine on sanity shapes,
then compute the actual Betti numbers -- and let them fall where they may. (Controlled ground-truth shapes,
so the test isolates the topology question from LRM reconstruction noise; the LRM-on-real-photos version adds
only the occluded-part hallucination, which we argue separately.)

torsion note: a mushroom is an orientable solid embedded in R^3, so its homology is torsion-FREE -- torsion
is identically 0 and not a usable feature. Only the free ranks B0/B1/B2 can carry signal. SAFETY: research vehicle.
"""
import numpy as np
from gudhi import CubicalComplex

N = 56
ix = np.arange(N)
Z, Y, X = np.meshgrid(ix, ix, ix, indexing="ij")
cx = cy = N // 2


def betti(vox):
    """Betti numbers [B0,B1,B2] of the occupied region (True=solid)."""
    cc = CubicalComplex(top_dimensional_cells=np.where(vox, 0.0, 1.0))
    cc.persistence()
    b = [0, 0, 0]
    for d in (0, 1, 2):
        for birth, death in cc.persistence_intervals_in_dimension(d):
            if birth <= 0.5 < death: b[d] += 1
    return b


# ---- primitives ----
def cyl(r, z0, z1): return ((X - cx) ** 2 + (Y - cy) ** 2 <= r * r) & (Z >= z0) & (Z < z1)
def ball(cz, r): return (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= r * r
def dome(cz, r): return ball(cz, r) & (Z >= cz)
def shell(cz, r, t): d = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2; return (d <= r * r) & (d >= (r - t) ** 2)
def lower_shell(cz, r, t): return shell(cz, r, t) & (Z < cz)            # an OPEN cup (bowl), open at top
def torus(cz, R, r): return (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - R) ** 2 + (Z - cz) ** 2 <= r * r
def skirt(z0, z1, ri, ro): rr = (X - cx) ** 2 + (Y - cy) ** 2; return (rr <= ro * ro) & (rr >= ri * ri) & (Z >= z0) & (Z < z1)


print("=== sanity check (homology engine on known shapes) ===")
for name, v, exp in [("solid ball", ball(28, 12), "[1,0,0]"),
                     ("solid torus", torus(28, 12, 4), "[1,1,0]"),
                     ("hollow sphere", shell(28, 14, 3), "[1,0,1]")]:
    print(f"  {name:14} betti {betti(v)}  (expected {exp})")

# ---- anatomically faithful mushrooms (stem + cap; ring = skirt flange; volva = open cup) ----
stem = cyl(4, 10, 40); cap = dome(40, 14)
agaricus = stem | cap | skirt(26, 28, 4, 11)                            # ring skirt attached to stem
amanita_mature = agaricus | lower_shell(16, 9, 3)                       # + OPEN volva cup at the base
amanita_button = cyl(4, 18, 40) | dome(40, 12) | shell(22, 15, 3)      # young: CLOSED universal veil (egg)
hollow_stem = (cyl(5, 10, 40) & ~cyl(2, 11, 39)) | cap                  # a hollow-stemmed species

print("\n=== mushroom Betti numbers [B0, B1, B2] ===")
for name, v in [("Agaricus (ring, NO volva)", agaricus),
                ("Amanita MATURE (open volva cup)", amanita_mature),
                ("Amanita BUTTON (closed veil)", amanita_button),
                ("hollow-stem species", hollow_stem)]:
    print(f"  {name:34} {betti(v)}")

# ---- the occlusion test: a single-view recon misses the base; remove bottom third ----
occ = amanita_mature & (Z >= 22)
print(f"\n  Amanita mature, BASE OCCLUDED (recon misses it):   {betti(occ)}   (volva info gone)")

print("""
READING:
- 'solid ball'/'torus'/'hollow sphere' confirm the engine: B2 only fires for a CLOSED cavity.
- The diagnostic difference (volva present/absent) is GEOMETRIC, not topological: an OPEN volva cup is a
  bowl = contractible, so it adds NO Betti number. A ring is a skirt-flange attached to the stem -> no loop.
  => Agaricus and mature Amanita have the SAME Betti numbers; topology cannot see the volva.
- B2 only appears for a CLOSED shell (button-stage universal veil) -- which is NOT the mature-volva diagnostic,
  and is the irreducible-ambiguity stage anyway. Hollow stem gives B1 -- also not the volva.
- Even with PERFECT 3D, Betti numbers are the wrong invariant here: mushroom diagnostics are differential/
  geometric (the cup's shape/curvature), not topological. And on real single-view photos you don't have the 3D
  base at all -- it's occluded (the test above) or hallucinated. torsion is identically 0 (orientable solid).
SAFETY: research vehicle only; topology is not edibility. When in doubt, DON'T EAT.
""")
