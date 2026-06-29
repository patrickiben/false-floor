"""Scrape the GENUINELY-coincident, expert-acknowledged 'requires a microscope' edible<->deadly confusions
(the regime the curated grid lacked). New species (the rest we already have: Galerina_marginata,
Lycoperdon_perlatum):
  Armillaria mellea       -- honey mushroom (EDIBLE); the most common FATAL Galerina confusion
  Kuehneromyces mutabilis -- sheathed woodtuft (EDIBLE); the canonical Galerina look-alike separable only by spores
  Hypholoma fasciculare   -- sulphur tuft (TOXIC LBM); clustered-wood look-alike of honey mushroom
  Scleroderma citrinum    -- common earthball (TOXIC); the puffball look-alike
Same GBIF paged fetch as bigscrape_train.py. SAFETY: research vehicle; DON'T EAT.
"""
import glob, os, socket, json, urllib.request, urllib.parse
ROOT = "data/fungi_big"; N = 400
NEW = ["Armillaria mellea", "Kuehneromyces mutabilis", "Hypholoma fasciculare", "Scleroderma citrinum"]
socket.setdefaulttimeout(20)
g = lambda u: json.load(urllib.request.urlopen(u, timeout=25))

for latin in NEW:
    slug = latin.replace(" ", "_"); d = f"{ROOT}/{slug}"; os.makedirs(d, exist_ok=True)
    got = len(glob.glob(f"{d}/*.jpg"))
    if got >= N:
        print(f"  {slug:24} {got} (enough)", flush=True); continue
    try:
        key = g("https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(latin)).get("usageKey")
    except Exception as e:
        print(f"  {slug}: match error {e}", flush=True); continue
    offset = 0
    while got < N and offset < N * 10:
        try:
            res = g(f"https://api.gbif.org/v1/occurrence/search?taxonKey={key}&mediaType=StillImage&limit=300&offset={offset}")
        except Exception:
            break
        for r in res.get("results", []):
            if got >= N: break
            url = (r.get("media") or [{}])[0].get("identifier")
            if not url: continue
            try:
                urllib.request.urlretrieve(url, f"{d}/{got:05d}.jpg"); got += 1
            except Exception:
                pass
        offset += 300
        if res.get("endOfRecords"): break
    print(f"  {slug:24} {got}/{N}", flush=True)
print("done", flush=True)
