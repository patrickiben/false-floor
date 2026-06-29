"""BIG-SCRAPE + REAL-ENCODER-TRAINING experiment (closing the 'capped at 300, never trained' gap).

Everything before this ran on FROZEN, general I-JEPA over ~1,600 RGB images. Here we (1) scrape a much
larger, broader fungal open-set from GBIF and (2) actually FINE-TUNE a real encoder (ResNet-50, ImageNet
-pretrained) on edible-vs-poison, then re-measure the LOAD-BEARING safety number -- the confidently-wrong
false-eat on held-out Destroying Angels + an open-set deadly species never trained -- AT SCALE WITH A
LEARNED REPRESENTATION, vs the frozen-I-JEPA baseline (~5%).

Honest expectation (FungiCLEF: 356K expert images, trained models, STILL nonzero poison cost): aggregate
accuracy rises, the confidently-wrong residual + open-set persist. We report whatever it shows, with the
same anti-rigging discipline (held-out species for open-set; fair matched-yield false-eat).
SAFETY: research vehicle only; latent/learned separability is not edibility. When in doubt, DON'T EAT.

Usage:  python bigscrape_train.py --stage scrape --n 500     # long pole, resumable
        python bigscrape_train.py --stage train
"""
import argparse, glob, os, socket, json, urllib.request, urllib.parse, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "data/fungi_big"
# latin name: (edibility 0=edible / 1=poison, role)  -- 'openset' species are held out ENTIRELY
SPECIES = {
    "Agaricus campestris": (0, "train"),   "Agaricus arvensis": (0, "train"),
    "Cantharellus cibarius": (0, "train"), "Morchella esculenta": (0, "train"),
    "Pleurotus ostreatus": (0, "train"),   "Macrolepiota procera": (0, "train"),
    "Hydnum repandum": (0, "train"),       "Lycoperdon perlatum": (0, "train"),
    "Amanita phalloides": (1, "train"),    "Amanita bisporigera": (1, "train"),     # <- the lethal-pair poison
    "Amanita muscaria": (1, "train"),      "Galerina marginata": (1, "train"),
    "Omphalotus illudens": (1, "train"),   "Clitocybe dealbata": (1, "train"),
    "Amanita virosa": (1, "openset"),      "Conocybe filaris": (1, "openset"),      # deadly, NEVER trained
}
LETHAL_E, LETHAL_P = "Agaricus_campestris", "Amanita_bisporigera"   # the comparable lethal-pair test


def fetch(n):
    socket.setdefaulttimeout(20)
    g = lambda u: json.load(urllib.request.urlopen(u, timeout=25))
    for latin, _ in SPECIES.items():
        slug = latin.replace(" ", "_"); d = f"{ROOT}/{slug}"; os.makedirs(d, exist_ok=True)
        got = len(glob.glob(f"{d}/*.jpg"))
        if got >= n:
            print(f"  {slug:24} {got} (enough)", flush=True); continue
        try:
            key = g("https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(latin)).get("usageKey")
        except Exception as e:
            print(f"  {slug}: match error {e}"); continue
        offset = 0
        while got < n and offset < n * 10:
            try:
                res = g(f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}"
                        f"&mediaType=StillImage&limit=300&offset={offset}")
            except Exception:
                break
            for r in res.get("results", []):
                if got >= n: break
                url = (r.get("media") or [{}])[0].get("identifier")
                if not url: continue
                try:
                    urllib.request.urlretrieve(url, f"{d}/{got:05d}.jpg"); got += 1
                except Exception:
                    pass
            offset += 300
            if res.get("endOfRecords"): break
        print(f"  {slug:24} {got}/{n}", flush=True)


def train_and_eval():
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision as tv
    from torchvision import transforms
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tf_tr = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_te = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # build splits: train species 70/30; lethal-pair test = held-out 30% of the two lethal species;
    # open-set = the 'openset' species, entirely held out
    rng = np.random.default_rng(0)
    train_paths, train_y, test_paths, test_y, ltest, lty, ostest = [], [], [], [], [], [], []
    for latin, (edib, role) in SPECIES.items():
        slug = latin.replace(" ", "_"); ps = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
        if not ps: continue
        if role == "openset":
            ostest += [(p, edib) for p in ps]; continue
        i = rng.permutation(len(ps)); c = int(0.7 * len(ps))
        for j in i[:c]: train_paths.append(ps[j]); train_y.append(edib)
        for j in i[c:]:
            test_paths.append(ps[j]); test_y.append(edib)
            if slug in (LETHAL_E, LETHAL_P): ltest.append(ps[j]); lty.append(edib)
    print(f"train {len(train_paths)} | test {len(test_paths)} | lethal-pair test {len(ltest)} | open-set {len(ostest)}", flush=True)

    class DS(Dataset):
        def __init__(s, paths, ys, tf): s.p, s.y, s.tf = paths, ys, tf
        def __len__(s): return len(s.p)
        def __getitem__(s, i):
            try: img = Image.open(s.p[i]).convert("RGB")
            except Exception: img = Image.new("RGB", (224, 224))
            return s.tf(img), float(s.y[i])

    net = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Linear(net.fc.in_features, 1); net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=1e-4)
    dl = DataLoader(DS(train_paths, train_y, tf_tr), batch_size=32, shuffle=True, num_workers=0)
    print("fine-tuning ResNet-50 (edible vs poison) ...", flush=True)
    for ep in range(4):
        net.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.float().to(dev)
            loss = nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"  epoch {ep+1}/4  loss {tot/len(dl):.3f}", flush=True)

    @torch.no_grad()
    def prob(paths):
        net.eval(); out = []
        for k in range(0, len(paths), 64):
            xb = torch.stack([tf_te(Image.open(p).convert("RGB")) for p in paths[k:k+64]]).to(dev)
            out.append(torch.sigmoid(net(xb).squeeze(-1)).cpu().numpy())
        return np.concatenate(out)

    from gate import auroc
    def at_yield(re, rp, y): return float((rp < np.quantile(re, y)).mean())
    yt = np.array(test_y); rt = prob(test_paths)
    print(f"\n=== TRAINED ResNet-50 @ scale (train {len(train_paths)} imgs, {len([s for s in SPECIES if SPECIES[s][1]=='train'])} species) ===")
    print(f"  aggregate held-out AUROC (edible vs poison): {auroc(rt, yt):.3f}")
    rl = prob(ltest); lty_a = np.array(lty)
    rle, rlp = rl[lty_a == 0], rl[lty_a == 1]
    print(f"  LETHAL pair (Agaricus campestris vs Amanita bisporigera), poison-eaten @60% yield: "
          f"{at_yield(rle, rlp, .6)*100:.1f}%   @80%: {at_yield(rle, rlp, .8)*100:.1f}%   (frozen I-JEPA baseline ~5%)")
    if ostest:
        osp = prob([p for p, _ in ostest])
        # open-set deadly species, never trained: at the lethal-pair 60%-yield threshold, are they eaten?
        thr = np.quantile(rle, .6)
        print(f"  OPEN-SET deadly ({len([1 for _ in ostest])} imgs, never trained) eaten at that threshold: "
              f"{float((osp < thr).mean())*100:.1f}%   (frozen I-JEPA open-set ~chance)")
    print("\nHonest read: did a TRAINED encoder on MORE data break the confidently-wrong floor, or (per")
    print("FungiCLEF) raise the ceiling while the residual + open-set persist? SAFETY: research vehicle; DON'T EAT.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    if a.stage == "scrape": fetch(a.n)
    elif a.stage == "train": train_and_eval()
