"""LEVER #3 (third suspected viability): adapt the REPRESENTATION, not a head on frozen latents.

Hand-rolled minimal LoRA (no peft dependency) on I-JEPA's query/value projections in the last 8 of
32 blocks, plus a linear edible/poison head. Trained on the lethal-pair (+ chanterelle/jacko) TRAIN
split only -- with the SAME rng(0) split sec_decision_lethal uses, so the held-out Destroying Angels
the decision is scored on are NEVER seen in fine-tuning (no leakage). Then re-encodes ALL species
with the adapted encoder and runs the SAME battery (separability / keeper / decision / open-set).

Prior: the frozen metric-head already showed raising aggregate separability does NOT fix the residual
and can worsen it; fine-tuning risks the same decoupling + overfitting on ~hundreds of imgs + open-set
degradation. Expect: separability up (partly circular for the trained pair), residual likely persists.
SAFETY: research vehicle only; latent separability is not edibility. When in doubt, DON'T EAT.
"""
from __future__ import annotations
import glob, warnings
import numpy as np
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

import real_ijepa_test as R   # reuse SPECIES + section functions

ROOT = "data/fungi_real"
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
TRAINED = ["edible_chanterelle", "poison_jackolantern", "edible_fieldmushroom", "poison_destroyingangel"]
LABEL = {"edible_chanterelle": 0, "poison_jackolantern": 1, "edible_fieldmushroom": 0, "poison_destroyingangel": 1}


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r)); self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def set_module(root, name, new):
    parts = name.split("."); obj = root
    for p in parts[:-1]: obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


def imgs_of(slug):
    return sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))


def train_split_paths():
    """Replicate sec_decision_lethal's rng(0) split; return LoRA-train (image_path, label), NO leakage."""
    rng = np.random.default_rng(0)
    pe = []  # paths
    # chanterelle + jacko used entirely as train there
    for s in ("edible_chanterelle", "poison_jackolantern"):
        for p in imgs_of(s): pe.append((p, LABEL[s]))
    # field mushroom then destroying angel: draw in the SAME order as sec_decision_lethal
    for s in ("edible_fieldmushroom", "poison_destroyingangel"):
        ip = imgs_of(s); idx = rng.permutation(len(ip)); c = int(0.7 * len(ip))
        for j in idx[:c]:                      # train 70% only
            pe.append((ip[j], LABEL[s]))
    return pe


def main():
    proc = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
    model = AutoModel.from_pretrained("facebook/ijepa_vith14_1k").to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    # inject LoRA on q,v of the last 8 blocks
    for i in range(24, 32):
        for w in ("query", "value"):
            name = f"encoder.layer.{i}.attention.attention.{w}"
            base = dict(model.named_modules())[name]
            set_module(model, name, LoRALinear(base).to(DEV))
    head = nn.Linear(1280, 2).to(DEV)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    print(f"LoRA params trainable: {sum(p.numel() for p in params):,}  device {DEV}", flush=True)

    data = train_split_paths()
    rng = np.random.default_rng(1)
    EP, BS = 3, 8
    model.train()
    for ep in range(EP):
        order = rng.permutation(len(data)); tot = 0.0
        for b in range(0, len(order), BS):
            batch = [data[j] for j in order[b:b + BS]]
            imgs = [Image.open(p).convert("RGB") for p, _ in batch]
            y = torch.tensor([lab for _, lab in batch]).to(DEV)
            inp = proc(images=imgs, return_tensors="pt").to(DEV)
            z = model(**inp).last_hidden_state.mean(1)
            loss = nn.functional.cross_entropy(head(z), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{EP}  loss {tot/(len(order)//BS+1):.3f}", flush=True)

    # re-encode ALL species with the adapted encoder
    model.eval(); lat = {}
    for slug in R.SPECIES:
        ip = imgs_of(slug)
        if not ip: continue
        out = []
        for k in range(0, len(ip), 16):
            imgs = [Image.open(p).convert("RGB") for p in ip[k:k + 16]]
            with torch.no_grad():
                z = model(**proc(images=imgs, return_tensors="pt").to(DEV)).last_hidden_state.mean(1)
            out.append(z.cpu().numpy())
        lat[slug] = np.concatenate(out).astype("f4"); np.save(f"{ROOT}/{slug}_lora.npy", lat[slug])
        print(f"  re-encoded {slug}: {lat[slug].shape}", flush=True)

    print("\n################  LoRA-ADAPTED battery  ################")
    print("(separability for the TRAINED pair is partly circular now; the load-bearing metric is the")
    print(" held-out DECISION false-eat and the open-set AUROC.)")
    R.sec_separability(lat, tag=" [LoRA]")
    R.sec_keeper(lat)
    R.sec_decision_lethal(lat)
    R.sec_openset_toxic(lat)
    print("\nCompare to FROZEN raw: sep 0.94 | false-eat 7/81@k0..k3 | open-set 0.56.")
    print("SAFETY: research vehicle only; when in doubt, DON'T EAT.")


if __name__ == "__main__":
    main()
