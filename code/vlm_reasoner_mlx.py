"""LEVER #1 (besides-JEPA), MLX edition: a current VLM (Qwen3-VL-32B-8bit) as a feature-DEMANDING
reasoner with a VISIBILITY GATE + abstention. Runs on the SAME held-out lethal-pair specimens JEPA
was scored on (rng(0) split), directly comparable.

Default model = mlx-community/Qwen3-VL-32B-Instruct-8bit: the most capable Qwen-VL that fits ~51GB
(dense 32B at 8-bit ~34GB resident; 235B doesn't fit, 30B-A3B MoE is weaker). MLX-native for Apple Silicon.

Decision rules:
  NAIVE:    eat iff the VLM says "safe to eat".
  CAUTIOUS: eat iff safe AND base visible AND no volva AND gills NOT white-and-free. Else abstain.

SETUP:  pip install mlx-vlm        RUN:  python vlm_reasoner_mlx.py   (--model ... --cap N)
SAFETY: research vehicle only; VLMs hallucinate and the feature must be VISIBLE. When in doubt DON'T EAT.
"""
import argparse, glob
import numpy as np
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ROOT = "data/fungi_real"
Q_BASE = "Is the BASE of this mushroom's stem (where it meets the ground) clearly visible in the image? Answer one word: YES or NO."
Q_VOLVA = "At the base of the stem, is there a cup, sac, bag, or swollen bulb (a volva)? Answer one word: YES, NO, or UNCLEAR."
Q_GILLS = "Are the gills white AND free (not attached to the stem)? Answer one word: YES, NO, or UNCLEAR."
Q_SAFE = "Based only on what is visible, is this exact wild mushroom safe for a human to eat? Answer one word: EAT, AVOID, or UNSURE."


def split_test(cap):
    rng = np.random.default_rng(0); out = {}
    for slug in ("edible_fieldmushroom", "poison_destroyingangel"):
        ip = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
        i = rng.permutation(len(ip)); c = int(0.7 * len(ip))
        out[slug] = [ip[j] for j in i[c:]][:cap]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-32B-Instruct-8bit")
    ap.add_argument("--cap", type=int, default=80)
    a = ap.parse_args()
    print(f"loading {a.model} (MLX) ...", flush=True)
    model, processor = load(a.model)
    config = load_config(a.model)
    print("model loaded; scoring now (progress prints every 10 images) ...", flush=True)

    def ask(path, q):
        prompt = apply_chat_template(processor, config, q, num_images=1)
        try:
            out = generate(model, processor, prompt, image=[path], max_tokens=16, temp=0.0, verbose=False)
        except TypeError:
            out = generate(model, processor, prompt, image=[path], max_tokens=16, verbose=False)
        out = out if isinstance(out, str) else getattr(out, "text", str(out))
        return out.strip().lower()

    def kwd(ans, opts):
        for o in opts:
            if o in ans: return o
        return opts[-1]

    test = split_test(a.cap)
    LAB = {"edible_fieldmushroom": "edible", "poison_destroyingangel": "POISON"}
    res = {}
    for slug, paths in test.items():
        rows = []
        for k, p in enumerate(paths):
            try:
                rows.append((kwd(ask(p, Q_BASE), ["yes", "no"]),
                             kwd(ask(p, Q_VOLVA), ["yes", "no", "unclear"]),
                             kwd(ask(p, Q_GILLS), ["yes", "no", "unclear"]),
                             kwd(ask(p, Q_SAFE), ["eat", "avoid", "unsure"])))
            except Exception as e:
                print("   (skip", p, type(e).__name__, str(e)[:80], ")")
            if (k + 1) % 10 == 0:
                print(f"    {slug}: {k+1}/{len(paths)} ...", flush=True)
        res[slug] = rows
        print(f"  {slug} ({LAB[slug]}): {len(rows)} scored", flush=True)

    print(f"\n=== VLM-as-reasoner [{a.model}] on held-out lethal pair (compare JEPA false-eat ~1-7/81; SmolVLM naive 92%) ===")
    for slug in ("poison_destroyingangel", "edible_fieldmushroom"):
        rows = res[slug]; n = max(1, len(rows))
        naive = sum(1 for b, v, g, s in rows if s == "eat")
        caut = sum(1 for b, v, g, s in rows if s == "eat" and b == "yes" and v == "no" and not (g == "yes"))
        base_vis = sum(1 for b, v, g, s in rows if b == "yes")
        volva_y = sum(1 for b, v, g, s in rows if v == "yes")
        gills_y = sum(1 for b, v, g, s in rows if g == "yes")
        tag = "false-eat" if slug == "poison_destroyingangel" else "yield"
        print(f"\n  {LAB[slug]} (n={len(rows)}):")
        print(f"    NAIVE  'safe to eat?'        -> EAT {naive}/{n} ({100*naive/n:.0f}%)  [{tag}]")
        print(f"    CAUTIOUS (safe & base-vis & no-volva & not white-free gills) -> EAT {caut}/{n} ({100*caut/n:.0f}%)  [{tag}]")
        print(f"    base visible {base_vis}/{n} | volva YES {volva_y}/{n} | white-free gills {gills_y}/{n}")
    print("\nKEY: vs SmolVLM (volva recall 4/60, naive 92% false-eat) -- does the 32B model detect the volva")
    print("far more and ABSTAIN when the base isn't visible, driving CAUTIOUS false-eat toward 0?")
    print("SAFETY: research vehicle only; VLMs hallucinate, feature must be visible. DON'T EAT.")


if __name__ == "__main__":
    main()
