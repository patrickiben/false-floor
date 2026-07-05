"""G2 numbers-gate (partial): recompute every Clopper-Pearson upper bound stated in
MANUSCRIPT_CAUTIONARY.md against the reported (point-rate, n) and flag any mismatch.
Convention (from the paper): UCB is one-sided 95% CP from x = round(mean%/100 * n).
Point estimates themselves need the image pipeline; the BOUNDS are checkable here.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from cp_bounds import cp_upper

# (label, point_rate_%, n, stated_UCB_%)
CELLS = [
    # main 5-pair pooled table (§3), n=720
    ("§3 pooled I-JEPA", 1.3, 720, 2.2), ("§3 pooled DINOv2", 0.1, 720, 0.4), ("§3 pooled CLIP", 0.0, 720, 0.4),
    # hard-pairs table (§3): n=150 control+galerina rows, n=120 sulphur/earthball
    ("hard control DINOv2", 0.3, 150, 2.0), ("hard control CLIP", 0.1, 150, 2.0),
    ("hard honey/Galerina DINOv2", 0.0, 150, 2.0), ("hard honey/Galerina CLIP", 0.1, 150, 2.0),
    ("hard woodtuft/Galerina DINOv2", 0.7, 150, 3.1), ("hard woodtuft/Galerina CLIP", 1.9, 150, 5.1),
    ("hard honey/sulphur DINOv2", 0.4, 120, 2.5), ("hard honey/sulphur CLIP", 1.4, 120, 5.2),
    ("hard puffball/earthball DINOv2", 0.0, 120, 2.5), ("hard puffball/earthball CLIP", 0.4, 120, 2.5),
    # §3.1 Apiaceae, n=90
    ("§3.1 pHemlock/carrot I-JEPA", 9.0, 90, 15.5), ("§3.1 pHemlock/carrot DINOv2", 0.2, 90, 3.3), ("§3.1 pHemlock/carrot CLIP", 0.2, 90, 3.3),
    ("§3.1 wHemlock/carrot I-JEPA", 5.8, 90, 11.3), ("§3.1 wHemlock/carrot DINOv2", 0.0, 90, 3.3), ("§3.1 wHemlock/carrot CLIP", 0.2, 90, 3.3),
    ("§3.1 pHemlock/parsley I-JEPA", 16.2, 90, 24.5), ("§3.1 pHemlock/parsley DINOv2", 0.8, 90, 5.2), ("§3.1 pHemlock/parsley CLIP", 1.4, 90, 5.2),
    # App C.1 Case 3 coral snakes, n=90
    ("coral/kingsnake I-JEPA", 13.8, 90, 20.7), ("coral/kingsnake DINOv2", 0.0, 90, 3.3), ("coral/kingsnake CLIP", 0.7, 90, 5.2),
    ("coral/milksnake I-JEPA", 4.2, 90, 9.9), ("coral/milksnake DINOv2", 0.0, 90, 3.3), ("coral/milksnake CLIP", 0.0, 90, 3.3),
    ("coral/scarletsnake I-JEPA", 22.4, 90, 30.6), ("coral/scarletsnake DINOv2", 0.0, 90, 3.3), ("coral/scarletsnake CLIP", 0.1, 90, 3.3),
    # App C.1 Case 4 recluse, n=90
    ("recluse/house I-JEPA", 48.9, 90, 58.0), ("recluse/house DINOv2", 2.0, 90, 6.8), ("recluse/house CLIP", 18.9, 90, 27.0),
    ("recluse/wolf I-JEPA", 10.3, 90, 16.8), ("recluse/wolf DINOv2", 0.0, 90, 3.3), ("recluse/wolf CLIP", 0.0, 90, 3.3),
    ("recluse/hobo I-JEPA", 26.0, 90, 34.2), ("recluse/hobo DINOv2", 0.0, 90, 3.3), ("recluse/hobo CLIP", 4.8, 90, 9.9),
    # App C.1 Case 5 frogs, mixed n
    ("frog variabilis/imitator I-JEPA", 18.3, 30, 35.7), ("frog variabilis/imitator DINOv2", 4.7, 30, 14.9), ("frog variabilis/imitator CLIP", 10.3, 30, 23.9),
    ("frog fantastica/imitator I-JEPA", 43.3, 30, 59.8), ("frog fantastica/imitator DINOv2", 0.3, 30, 9.5), ("frog fantastica/imitator CLIP", 2.7, 30, 14.9),
    ("frog Allobates/bilinguis I-JEPA", 12.7, 90, 19.4), ("frog Allobates/bilinguis DINOv2", 0.7, 90, 5.2), ("frog Allobates/bilinguis CLIP", 0.9, 90, 5.2),
    ("frog Allobates/parvula I-JEPA", 33.6, 22, 51.5), ("frog Allobates/parvula DINOv2", 10.0, 22, 25.9), ("frog Allobates/parvula CLIP", 16.4, 22, 36.9),
    # App C.1 Case 6 berries, n=90
    ("elderberry/pokeweed I-JEPA", 19.9, 90, 28.2), ("elderberry/pokeweed DINOv2", 0.0, 90, 3.3), ("elderberry/pokeweed CLIP", 0.0, 90, 3.3),
    ("wildgrape/moonseed I-JEPA", 10.8, 90, 18.1), ("wildgrape/moonseed DINOv2", 0.0, 90, 3.3), ("wildgrape/moonseed CLIP", 0.3, 90, 3.3),
    ("bilberry/nightshade I-JEPA", 0.2, 90, 3.3), ("bilberry/nightshade DINOv2", 0.0, 90, 3.3), ("bilberry/nightshade CLIP", 0.0, 90, 3.3),
    # §3.2/§3 control cell restated
    ("§3.2 control DINOv2", 0.3, 150, 2.0), ("§3.2 control CLIP", 0.1, 150, 2.0),
]

TOL = 0.15  # percentage points; stated bounds are rounded to 0.1%
mism = []
print(f"{'cell':40s} {'pt%':>5} {'n':>4} {'stated':>7} {'recomp':>7} {'x':>3}  ok?")
for label, pt, n, stated in CELLS:
    x = round(pt / 100.0 * n)
    recomp = cp_upper(x, n, 0.05) * 100.0
    ok = abs(recomp - stated) <= TOL
    if not ok:
        mism.append((label, pt, n, stated, round(recomp, 2), x))
    print(f"{label:40s} {pt:5.1f} {n:4d} {stated:6.1f}% {recomp:6.2f}% {x:3d}  {'OK' if ok else '<<<MISMATCH'}")

print("=" * 78)
if mism:
    print(f"G2 CP-BOUND MISMATCHES: {len(mism)}/{len(CELLS)}")
    for m in mism:
        print(f"  {m[0]}: stated <={m[3]}% but cp_upper(x={m[5]}, n={m[2]})={m[4]}%")
    sys.exit(1)
print(f"G2 PASS: all {len(CELLS)} stated Clopper-Pearson upper bounds match cp_upper(round(pt/100*n), n) to <= {TOL}pp")
