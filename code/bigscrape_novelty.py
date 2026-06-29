"""Firming-up the trained-encoder result: (a) MULTI-SEED the ResNet (CI on the single-seed numbers) and
(b) a TRUE-NOVELTY open-set -- unrelated-genus killers (Gyromitra/gyromitrin, Cortinarius/orellanine,
Conocybe) held out entirely -- vs the SISTER Amanita (A. virosa) that the single-seed run lumped together.

Question: was the 1.6% open-set real, or a sister-Amanita artifact (the model trained on 5 Amanitas
recognizing a 6th)? Prediction: NOVEL >> SISTER, i.e. training helps in-CATEGORY discrimination, not
genuine novelty. SAFETY: research vehicle; when in doubt, DON'T EAT.
"""
import glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from gate import auroc

ROOT = "data/fungi_big"
TRAIN_SPECIES = {
    "Agaricus campestris": 0, "Agaricus arvensis": 0, "Cantharellus cibarius": 0, "Morchella esculenta": 0,
    "Pleurotus ostreatus": 0, "Macrolepiota procera": 0, "Hydnum repandum": 0, "Lycoperdon perlatum": 0,
    "Amanita phalloides": 1, "Amanita bisporigera": 1, "Amanita muscaria": 1, "Galerina marginata": 1,
    "Omphalotus illudens": 1, "Clitocybe dealbata": 1,
}
SISTER = ["Amanita virosa"]                                       # sister of 5 trained Amanitas
NOVEL = ["Conocybe filaris", "Gyromitra esculenta", "Cortinarius rubellus"]   # unrelated-genus killers
LE, LP = "Agaricus_campestris", "Amanita_bisporigera"
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TR = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM])
TF_TE = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
imgs = lambda slug: sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))


class DS(Dataset):
    def __init__(s, p, y): s.p, s.y = p, y
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        try: im = Image.open(s.p[i]).convert("RGB")
        except Exception: im = Image.new("RGB", (224, 224))
        return TF_TR(im), np.float32(s.y[i])


@torch.no_grad()
def prob(net, paths):
    net.eval(); out = []
    for k in range(0, len(paths), 64):
        xb = torch.stack([TF_TE(Image.open(p).convert("RGB")) for p in paths[k:k + 64]]).to(dev)
        out.append(torch.sigmoid(net(xb).squeeze(-1)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_eval(seed):
    rng = np.random.default_rng(seed)
    tr_p, tr_y, lte, ltp = [], [], [], []
    for latin, edib in TRAIN_SPECIES.items():
        slug = latin.replace(" ", "_"); ps = imgs(slug)
        if not ps: continue
        i = rng.permutation(len(ps)); c = int(0.7 * len(ps))
        for j in i[:c]: tr_p.append(ps[j]); tr_y.append(edib)
        for j in i[c:]:
            if slug == LE: lte.append(ps[j])
            if slug == LP: ltp.append(ps[j])
    net = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Linear(net.fc.in_features, 1); net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=1e-4)
    dl = DataLoader(DS(tr_p, tr_y), batch_size=32, shuffle=True, num_workers=0)
    net.train()
    for ep in range(3):
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)
            loss = nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step()
    re, rp = prob(net, lte), prob(net, ltp)
    thr = np.quantile(re, .6)
    sister = np.concatenate([prob(net, imgs(s.replace(" ", "_"))) for s in SISTER])
    novel = np.concatenate([prob(net, imgs(s.replace(" ", "_"))) for s in NOVEL])
    print(f"  seed {seed} done", flush=True)
    return [float((rp < thr).mean()) * 100, float((rp < np.quantile(re, .8)).mean()) * 100,
            float((sister < thr).mean()) * 100, float((novel < thr).mean()) * 100,
            auroc(np.r_[re, rp], np.r_[np.zeros(len(re)), np.ones(len(rp))])]


print("multi-seed ResNet + true-novelty open-set (3 trainings) ...", flush=True)
R = np.array([train_eval(s) for s in range(3)]); m, sd = R.mean(0), R.std(0)
print(f"\n=== TRAINED ResNet-50, 3-seed mean +/- sd ===")
print(f"  lethal-pair false-eat @60%: {m[0]:.1f} +/- {sd[0]:.1f}%   @80%: {m[1]:.1f} +/- {sd[1]:.1f}%   (lethal AUROC {m[4]:.3f})")
print(f"  OPEN-SET  sister Amanita (A. virosa)        eaten: {m[2]:.1f} +/- {sd[2]:.1f}%")
print(f"  OPEN-SET  NOVEL unrelated genera            eaten: {m[3]:.1f} +/- {sd[3]:.1f}%   (Conocybe / Gyromitra / Cortinarius)")
print(f"\n  frozen-I-JEPA control (same data): lethal 3.1/8.1%, open-set 7.1%")
print("  => if NOVEL >> SISTER, the open-set 'win' was in-genus generalization, not novelty (predicted).")
print("SAFETY: research vehicle only; when in doubt, DON'T EAT.")
