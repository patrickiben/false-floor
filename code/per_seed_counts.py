"""Per-seed integer counts for the tables: replicates the scripts' evaluation loop (70/30 split, 48-PC projection,
12-member MLP ensemble, 80th-percentile-of-edible threshold) on the archived __full feature caches and records, per seed,
the integer number of held-out poisons (and open-set images) cleared. Run from the repo root: FULL=1 python code/per_seed_counts.py
with data/fungi_big/ present. Output: JSON at $OUT (default per_seed_counts.json) plus one summary line per cell."""
import os, re, ast, sys, json, numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); os.environ["FULL"]="1"; sys.path.insert(0, HERE); torch.set_num_threads(2)
os.environ.setdefault("OUT","per_seed_counts.json")
from model import train_ensemble, ensemble_risk
from config import CFG
ROOT="data/fungi_big"
def block(src, name):
    i=src.index(name+" = ["); start=i+len(name)+3; d=0
    for j in range(start, len(src)):
        d += (src[j]=="[") - (src[j]=="]")
        if d==0: return ast.literal_eval(src[start:j+1])
PE=open(os.path.join(HERE,"pairs_encoders.py")).read(); HR=open(os.path.join(HERE,"hard_regime.py")).read(); HP=open(os.path.join(HERE,"hard_pairs.py")).read()
PAIRS=block(PE,"PAIRS"); OPENSET=block(PE,"OPENSET"); LOOK=block(HR,"LOOKALIKE_OPENSET"); HPAIRS=block(HP,"PAIRS")
ENC=["I-JEPA-H14","DINOv2-L","CLIP-L"]
def load(s, ename, mode=None):
    for c in ([f"{ROOT}/{s}__{ename}__{mode}__full.npy"] if mode else []) + [f"{ROOT}/{s}__{ename}__full.npy"]:
        if os.path.exists(c): return np.load(c)
    raise FileNotFoundError(f"{s} {ename} {mode}")
def per_seed(Xe, Xp, Xo=None):
    out=[]
    for seed in range(10):
        rng=np.random.default_rng(seed)
        def sp(n): i=rng.permutation(n); c=int(.7*n); return i[:c], i[c:]
        etr,ete=sp(len(Xe)); ptr,pte=sp(len(Xp))
        Xtr=np.vstack([Xe[etr],Xp[ptr]]); ytr=np.r_[np.zeros(len(etr)),np.ones(len(ptr))].astype("f4")
        mu=Xtr.mean(0); _,_,Vt=np.linalg.svd(Xtr-mu,full_matrices=False); V=Vt[:48]; pj=lambda A:((A-mu)@V.T).astype("f4")
        ens=train_ensemble(pj(Xtr),ytr,48,CFG)
        re_=ensemble_risk(ens,pj(Xe[ete]))[0]; rp=ensemble_risk(ens,pj(Xp[pte]))[0]; thr=float(np.quantile(re_,.8))
        rec=dict(seed=seed,n_p=int(len(pte)),eaten=int((rp<thr).sum()))
        if Xo is not None: ro=ensemble_risk(ens,pj(Xo))[0]; rec.update(n_o=int(len(Xo)),open_eaten=int((ro<thr).sum()))
        out.append(rec)
    return out
R={"env":f"torch {torch.__version__}, numpy {np.__version__}", "cells":{}}
def run(key, Xe, Xp, Xo=None):
    r=per_seed(Xe,Xp,Xo); R["cells"][key]=r
    e=[x["eaten"] for x in r]; s=f"{key:58s} n_p={r[0]['n_p']:4d} eaten/seed={e} sum={sum(e)} max={max(e)} mean%={100*np.mean(e)/r[0]['n_p']:.2f}"
    if Xo is not None: o=[x["open_eaten"] for x in r]; s+=f" | n_o={r[0]['n_o']} open/seed={o} sum={sum(o)} mean%={100*np.mean(o)/r[0]['n_o']:.3f}"
    print(s, flush=True); json.dump(R, open(os.environ["OUT"],"w"), indent=1)
# (a) the 1151 open set on the Agaricus/Destroying-Angel DINOv2 classifier (hard_regime curated caches)
Xo1151=np.vstack([load(s,"DINOv2-L","curated") for s in LOOK])
run("hard_regime: Agaricus/A.bisporigera DINOv2 + LOOKALIKE_OPENSET(1151)", load("Agaricus_campestris","DINOv2-L","curated"), load("Amanita_bisporigera","DINOv2-L","curated"), Xo1151)
# (b) five ordinary pairs x 3 encoders with the 906 open set (pairs_encoders caches)
for en in ENC:
    Xo906=np.vstack([load(s,en) for s in OPENSET])
    for lab,e,p in PAIRS: run(f"pairs_encoders: {lab} [{en}]", load(e,en), load(p,en), Xo906)
# (c) hard pairs x 3 encoders (hard_pairs caches)
for en in ENC:
    for lab,e,p in HPAIRS: run(f"hard_pairs: {lab} [{en}]", load(e,en), load(p,en))
print("DONE", flush=True)
