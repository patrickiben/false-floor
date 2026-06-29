"""EMPIRICAL version of the topology test: a real single-view LRM (TripoSR) reconstructs actual mushroom
photos -> persistent homology (Betti B0/B1/B2) on the reconstruction. Patrick asked for the real LRM, not
the synthetic lab. Question: does a real LRM reconstruction of a real Amanita (volva) differ topologically
from an Agaricus (no volva)?

We bypass the torchmcubes build entirely: TripoSR computes a DENSITY FIELD (renderer.query_triplane ->
density_act) before marching cubes; we query that field on a regular grid, threshold it into a voxel
occupancy, and run the SAME gudhi cubical homology validated in betti_3d_lab.py. (torchmcubes is stubbed so
the import succeeds; we never call marching cubes.)

Honest prior (synthetic lab already proved an open volva cup is contractible, B2=0; and an LRM blobs fine
occluded structure): expect low-fidelity blobs, [1,0,0]-ish for BOTH classes, no topological separation --
i.e. the real LRM confirms the synthetic result, for two compounding reasons. SAFETY: research vehicle; DON'T EAT.
"""
import sys, types, glob, warnings, time
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
# stub torchmcubes (unbuilt C++ ext, we skip marching cubes) and rembg (bg removal, we feed RGB directly)
_mc = types.ModuleType("torchmcubes"); _mc.marching_cubes = lambda *a, **k: None
sys.modules["torchmcubes"] = _mc
_rb = types.ModuleType("rembg"); _rb.remove = lambda *a, **k: None; _rb.new_session = lambda *a, **k: None
sys.modules["rembg"] = _rb
sys.path.insert(0, "/tmp/TripoSR_src")
import torch
from PIL import Image
from gudhi import CubicalComplex
from tsr.system import TSR
from tsr.utils import scale_tensor

R = 96; THRESH = 25.0; N_PER = 5
DATA = "data/fungi_real"
GROUPS = {"Amanita (volva)": ["poison_destroyingangel", "heldout_deathcap"],
          "Agaricus (no volva)": ["edible_fieldmushroom"]}
dev = "cpu"   # query_triplane grid_sample is stable on CPU; avoids MPS edge cases

print("loading TripoSR (downloads ~1.6GB on first run) ...", flush=True)
t0 = time.time()
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192); model.to(dev).eval()
model.set_marching_cubes_resolution(R)
grid = scale_tensor(model.isosurface_helper.grid_vertices, model.isosurface_helper.points_range,
                    (-model.renderer.cfg.radius, model.renderer.cfg.radius)).to(dev)
print(f"  loaded in {time.time()-t0:.0f}s; density grid {R}^3", flush=True)


def betti(occ):
    cc = CubicalComplex(top_dimensional_cells=np.where(occ, 0.0, 1.0)); cc.persistence()
    b = [0, 0, 0]
    for d in (0, 1, 2):
        for bd, dd in cc.persistence_intervals_in_dimension(d):
            if bd <= 0.5 < dd: b[d] += 1
    return b


@torch.no_grad()
def recon_betti(path):
    im = Image.open(path).convert("RGB")
    scene_codes = model([im], device=dev)
    density = model.renderer.query_triplane(model.decoder, grid, scene_codes[0])["density_act"]
    occ = (density.reshape(R, R, R) > THRESH).cpu().numpy()
    if occ.sum() < 20: return None, 0.0           # empty/failed reconstruction
    return betti(occ), float(occ.mean())


for gname, slugs in GROUPS.items():
    paths = []
    for s in slugs: paths += sorted(glob.glob(f"{DATA}/{s}/*.jpg"))[:N_PER]
    paths = paths[:N_PER]
    print(f"\n=== {gname} ===", flush=True)
    rows = []
    for p in paths:
        t = time.time()
        try:
            b, frac = recon_betti(p)
        except Exception as e:
            print(f"  {p.split('/')[-1]:28} FAILED {type(e).__name__}: {str(e)[:60]}", flush=True); continue
        if b is None:
            print(f"  {p.split('/')[-1]:28} empty reconstruction", flush=True); continue
        rows.append(b); print(f"  {p.split('/')[-1]:28} betti {b}  occ {frac*100:4.1f}%  ({time.time()-t:.0f}s)", flush=True)
    if rows:
        a = np.array(rows)
        print(f"  -> mean Betti [B0,B1,B2] = [{a[:,0].mean():.2f}, {a[:,1].mean():.2f}, {a[:,2].mean():.2f}]   (n={len(rows)})")

print("""
READING: if both classes reconstruct to ~[1,0,0] blobs with no B2, the real LRM CONFIRMS the synthetic lab
on real photos -- for two compounding reasons: (1) TripoSR blobs the fine/occluded volva (single-view detail
limit), and (2) even a perfectly reconstructed OPEN volva cup is contractible (B2=0; proven in betti_3d_lab).
Topology cannot separate Amanita from Agaricus on real reconstructions. SAFETY: research vehicle; DON'T EAT.
""")
