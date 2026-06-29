"""LEVER #1 (besides-JEPA): a VLM as a feature-DEMANDING reasoner with a VISIBILITY GATE + abstention.

Runs a capable VLM on the SAME held-out lethal-pair specimens JEPA was scored on (rng(0) split), so
results are directly comparable. Improvements over the SmolVLM pilot:
  - VISIBILITY GATE: first ask if the stem base is even visible; ABSTAIN when it isn't (you cannot
    rule out an Amanita volva you can't see). The pilot's false-safety came from skipping this.
  - capable model: default Qwen2.5-VL-7B (use --model for 3B / 32B-4bit / Llama-Vision / SmolVLM).

Decision rules:
  NAIVE:    eat iff the VLM says "safe to eat".
  CAUTIOUS: eat iff safe AND base visible AND no volva AND gills are NOT white-and-free. Else abstain.
            (Destroying Angels = white free gills + volva at base -> the cautious rule should catch them.)

SETUP on the Mac Studio (one time):
  source ~/Documents/Pattern1_LangGoal_Planner/.venv/bin/activate
  pip install torchvision qwen-vl-utils accelerate          # torchvision is REQUIRED for Qwen2.5-VL
RUN:
  cd ~/Documents/EatThisMushroom
  python vlm_reasoner.py                                     # Qwen2.5-VL-7B, 80/class
  python vlm_reasoner.py --model Qwen/Qwen2.5-VL-32B-Instruct --load4bit   # max capability (needs bitsandbytes)
  python vlm_reasoner.py --model HuggingFaceTB/SmolVLM-Instruct            # the small pilot baseline

SAFETY: research vehicle only; VLMs hallucinate and the feature must be VISIBLE. When in doubt, DON'T EAT.
"""
import argparse, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image

ROOT = "data/fungi_real"
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

Q_BASE = "Is the BASE of this mushroom's stem (where it meets the ground) visible in the image? Answer one word: YES or NO."
Q_VOLVA = ("At the base of the stem, is there a cup, sac, bag, or swollen bulb (a volva)? "
           "Answer one word: YES, NO, or UNCLEAR.")
Q_GILLS = "Are the gills white AND free (not attached to the stem)? Answer one word: YES, NO, or UNCLEAR."
Q_SAFE = "Based only on what is visible, is this exact wild mushroom safe for a human to eat? Answer one word: EAT, AVOID, or UNSURE."


def split_test(cap):
    """Replicate sec_decision_lethal rng(0): field-mushroom perm THEN destroying-angel perm."""
    rng = np.random.default_rng(0); out = {}
    for slug in ("edible_fieldmushroom", "poison_destroyingangel"):
        ip = sorted(glob.glob(f"{ROOT}/{slug}/*.jpg"))
        i = rng.permutation(len(ip)); c = int(0.7 * len(ip))
        out[slug] = [ip[j] for j in i[c:]][:cap]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--load4bit", action="store_true")
    a = ap.parse_args()
    is_qwen = "qwen" in a.model.lower()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"loading {a.model} on {dev} ...", flush=True)
    proc = AutoProcessor.from_pretrained(a.model)
    kw = dict(dtype=torch.float16)
    if a.load4bit:
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    if not a.load4bit: model = model.to(dev)
    model.eval()

    def ask(img, q):
        if is_qwen:
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            try:
                from qwen_vl_utils import process_vision_info
                imgs, _ = process_vision_info(msgs)
            except Exception:
                imgs = [img]
            inp = proc(text=[text], images=imgs, return_tensors="pt")
        else:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=text, images=[img], return_tensors="pt")
        inp = inp.to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=12, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()

    def kwd(ans, opts):
        for o in opts:
            if o in ans: return o
        return opts[-1]

    test = split_test(a.cap)
    LAB = {"edible_fieldmushroom": "edible", "poison_destroyingangel": "POISON"}
    res = {}
    for slug, paths in test.items():
        rows = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                rows.append((kwd(ask(img, Q_BASE), ["yes", "no"]),
                             kwd(ask(img, Q_VOLVA), ["yes", "no", "unclear"]),
                             kwd(ask(img, Q_GILLS), ["yes", "no", "unclear"]),
                             kwd(ask(img, Q_SAFE), ["eat", "avoid", "unsure"])))
            except Exception:
                continue
        res[slug] = rows
        print(f"  {slug} ({LAB[slug]}): {len(rows)} scored", flush=True)

    print(f"\n=== VLM-as-reasoner [{a.model}] on held-out lethal pair (compare JEPA false-eat ~1-7/81) ===")
    for slug in ("poison_destroyingangel", "edible_fieldmushroom"):
        rows = res[slug]; n = max(1, len(rows))
        naive = sum(1 for b, v, g, s in rows if s == "eat")
        caut = sum(1 for b, v, g, s in rows if s == "eat" and b == "yes" and v == "no" and not (g == "yes"))
        base_vis = sum(1 for b, v, g, s in rows if b == "yes")
        volva_y = sum(1 for b, v, g, s in rows if v == "yes")
        gills_y = sum(1 for b, v, g, s in rows if g == "yes")
        tag = "false-eat" if slug == "poison_destroyingangel" else "yield"
        print(f"\n  {LAB[slug]} (n={len(rows)}):")
        print(f"    NAIVE  'safe to eat?'                  -> EAT {naive}/{n} ({100*naive/n:.0f}%)  [{tag}]")
        print(f"    CAUTIOUS (safe & base-visible & no-volva & not-white-free-gills) -> EAT {caut}/{n} ({100*caut/n:.0f}%)  [{tag}]")
        print(f"    base visible {base_vis}/{n} | volva YES {volva_y}/{n} | white-free gills {gills_y}/{n}")
    print("\nWHAT TO LOOK FOR: does the bigger model (a) detect the volva on Destroying Angels far more")
    print("than SmolVLM's 4/60, and (b) ABSTAIN when the base isn't visible -> driving CAUTIOUS false-eat")
    print("toward 0?  If naive stays high but cautious drops, the feature-demand+visibility-gate is the win.")
    print("SAFETY: research vehicle only; a VLM hallucinates, feature must be visible. DON'T EAT.")


if __name__ == "__main__":
    main()
