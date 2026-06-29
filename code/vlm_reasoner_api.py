"""LEVER #1 via GGUF / LM Studio: VLM-as-feature-demanding reasoner over an OpenAI-compatible local
server (llama.cpp Metal backend -- no MLX shared events, the thing that was hanging). Same held-out
lethal-pair specimens, same visibility-gate protocol, directly comparable to JEPA + SmolVLM.

USAGE:
  1) In LM Studio: download a Qwen3-VL GGUF (e.g. 32B Q8_0) + its mmproj, load it, then start the
     local server  (`lms server start`  or the GUI "Developer" tab). Default port 1234.
  2) python vlm_reasoner_api.py            # auto-detects the loaded model
     python vlm_reasoner_api.py --cap 30   # quick pass

Decision rules:
  NAIVE:    eat iff the VLM says "safe to eat".
  CAUTIOUS: eat iff safe AND base visible AND no volva AND gills NOT white-and-free. Else abstain.
SAFETY: research vehicle only; VLMs hallucinate and the feature must be VISIBLE. When in doubt DON'T EAT.
"""
import argparse, glob, base64, json, urllib.request
import numpy as np

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
    ap.add_argument("--base", default="http://localhost:1234/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cap", type=int, default=80)
    a = ap.parse_args()

    def api(path, payload=None):
        url = a.base + path
        if payload is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))

    model = a.model
    if not model:
        try:
            model = api("/models")["data"][0]["id"]
        except Exception as e:
            print(f"could not reach server at {a.base} (is the LM Studio server started?):", e); return
    print(f"server {a.base} | model {model}", flush=True)

    def ask(path, q):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        payload = {"model": model, "max_tokens": 16, "temperature": 0,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": q},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
        return api("/chat/completions", payload)["choices"][0]["message"]["content"].strip().lower()

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

    print(f"\n=== VLM-as-reasoner [GGUF: {model}] on held-out lethal pair (compare JEPA ~1-7/81; SmolVLM naive 92%) ===")
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
    print("\nKEY: does the 32B GGUF detect the volva far more than SmolVLM's 4/60 and ABSTAIN when the base")
    print("isn't visible -> driving CAUTIOUS false-eat toward 0?  SAFETY: research vehicle only. DON'T EAT.")


if __name__ == "__main__":
    main()
