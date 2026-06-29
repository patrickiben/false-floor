"""Smoke test: can we load a capable VLM here and ask a diagnostic question about a real mushroom?
Tries a few small open VLMs in order; prints which loaded + its answer on one Destroying Angel image.
"""
import glob, warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
img_path = sorted(glob.glob("data/fungi_real/poison_destroyingangel/*.jpg"))[0]
img = Image.open(img_path).convert("RGB")
Q = ("Look at the base of this mushroom's stem. Is there a cup-like sac, bag, or swollen bulb "
     "(a volva) at the very bottom? Answer strictly one of: YES / NO / NOT VISIBLE, then one short reason.")

CANDIDATES = ["Qwen/Qwen2.5-VL-3B-Instruct", "HuggingFaceTB/SmolVLM-Instruct"]

def try_model(mid):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.float16).to(dev).eval()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=text, images=[img], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return ans

for mid in CANDIDATES:
    try:
        print(f"trying {mid} on {dev} ...", flush=True)
        ans = try_model(mid)
        print(f"\nLOADED: {mid}\nimage: {img_path}\nQ: volva at base?\nA: {ans.strip()}")
        print("\nSMOKE OK")
        break
    except Exception as e:
        print(f"  {mid} FAILED: {type(e).__name__}: {str(e)[:200]}")
else:
    print("\nNo VLM loaded — will provide the scorer as a Mac script instead.")
