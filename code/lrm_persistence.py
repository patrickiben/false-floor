"""PROPER persistent homology on the TripoSR reconstruction (fixing lrm_betti.py's two flaws: raw
single-threshold Betti counts noise, and no background handling). Here we run a GRADED cubical filtration
on the continuous density field (superlevel-set), so each H1/H2 feature has a real PERSISTENCE (death-birth)
and noise (low-persistence) can be separated from genuine structure (high-persistence). This is 'persistent
homology on the LRM output' as asked.

Question: does any HIGH-PERSISTENCE H2 cavity (a reconstructed volva) appear more in Amanita than Agaricus?
Also report the largest-connected-component fraction (fragmentation) to show reconstruction quality. A crude
central foreground crop stands in for rembg (curated photos are roughly centered). SAFETY: research vehicle; DON'T EAT.
"""
import sys, types, glob, warnings, time
import numpy as np
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
_mc = types.ModuleType("torchmcubes"); _mc.marching_cubes = lambda *a, **k: None
sys.modules["torchmcubes"] = _mc
_rb = types.ModuleType("rembg"); _rb.remove = lambda *a, **k: None; _rb.new_session = lambda *a, **k: None
sys.modules["rembg"] = _rb
sys.path.insert(0, "/tmp/TripoSR_src")
import torch
from PIL import Image
from gudhi import CubicalComplex
from scipy import ndimage
from tsr.system import TSR
from tsr.utils import scale_tensor

R = 96; SURF = 25.0; N_PER = 8
DATA = "data/fungi_real"
GROUPS = {"Amanita (volva)": ["poison_destroyingangel", "heldout_deathcap"],
          "Agaricus (no volva)": ["edible_fieldmushroom"]}
dev = "cpu"

print("loading TripoSR ...", flush=True)
model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.renderer.set_chunk_size(8192); model.to(dev).eval()
model.set_marching_cubes_resolution(R)
grid = scale_tensor(model.isosurface_helper.grid_vertices, model.isosurface_helper.points_range,
                    (-model.renderer.cfg.radius, model.renderer.cfg.radius)).to(dev)


def crop_center(im, frac=0.8):
    w, h = im.size; s = int(min(w, h) * frac)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


@torch.no_grad()
def recon_density(path):
    im = crop_center(Image.open(path).convert("RGB"))
    sc = model([im], device=dev)
    d = model.renderer.query_triplane(model.decoder, grid, sc[0])["density_act"]
    return d.reshape(R, R, R).cpu().numpy().astype("f4")


def analyze(dens):
    # graded persistence on -density (sublevel of -dens = superlevel of dens); persistence in density units
    cc = CubicalComplex(top_dimensional_cells=(-dens)); cc.persistence()
    out = {}
    for dim in (1, 2):
        pers = sorted([dd - bd for bd, dd in cc.persistence_intervals_in_dimension(dim) if np.isfinite(dd)], reverse=True)
        out[f"max{dim}"] = pers[0] if pers else 0.0
        out[f"npersist{dim}"] = int(sum(p > SURF for p in pers))   # features surviving a SURF-sized density swing
    occ = dens > SURF
    lbl, n = ndimage.label(occ)
    out["frag"] = n                                                 # connected components at the surface threshold
    out["lcc"] = float((lbl == (np.bincount(lbl.flat)[1:].argmax() + 1)).sum() / max(occ.sum(), 1)) if n else 0.0
    return out


for gname, slugs in GROUPS.items():
    paths = []
    for s in slugs: paths += sorted(glob.glob(f"{DATA}/{s}/*.jpg"))[:N_PER]
    paths = paths[:N_PER]
    print(f"\n=== {gname} ===", flush=True)
    print(f"  {'image':16} | persistH2 (count, max) | persistH1 cnt | frags | largest-CC frac", flush=True)
    rows = []
    for p in paths:
        try:
            a = analyze(recon_density(p))
        except Exception as e:
            print(f"  {p.split('/')[-1]:16} FAILED {type(e).__name__}: {str(e)[:50]}", flush=True); continue
        rows.append(a)
        print(f"  {p.split('/')[-1]:16} |   {a['npersist2']:2d}, {a['max2']:7.1f}     |     {a['npersist1']:3d}      |  {a['frag']:3d}  |   {a['lcc']*100:4.1f}%", flush=True)
    if rows:
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"  -> mean: persistent-H2 cavities {m('npersist2'):.1f} (max pers {m('max2'):.0f}) | "
              f"persistent-H1 {m('npersist1'):.1f} | frags {m('frag'):.0f} | largest-CC {m('lcc')*100:.0f}%   (n={len(rows)})")

print("""
READING: 'persistent-H2' counts cavities that survive a density swing >= the surface threshold (real, not
noise). If both classes have ~0 persistent H2 and high fragmentation / low largest-CC, the LRM reconstructions
are noisy non-solid blobs with NO genuine anatomical cavity, and topology does not separate volva from no-volva
-- the real LRM confirms the synthetic lab (open volva cup is contractible anyway, B2=0). If Amanita showed a
persistent H2 the others lack, that would be the one positive signal. SAFETY: research vehicle; DON'T EAT.
""")
