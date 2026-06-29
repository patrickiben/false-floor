"""CASE 6 (berries): the famous deadly berry foraging confusions (a new plant ORGAN -- fruit). Documented
toxicology/field-guide hazards:
  Pokeweed (Phytolacca americana, toxic) vs Elderberry (Sambucus, edible) -- dark berry clusters
  Moonseed (Menispermum canadense, toxic) vs Wild grape (Vitis, edible)   -- "the deadly grape look-alike"
  Deadly nightshade (Atropa belladonna, deadly) vs Bilberry (Vaccinium, edible) -- iconic toxic berry
Same GBIF fetch. Caveat (stated in the paper): GBIF images are often whole-plant, so this tests species
separability, not isolated-berry-cluster confusion; the encoder-dependence comparison holds regardless.
SAFETY: research vehicle; DON'T EAT.
"""
import glob, os, socket, json, urllib.request, urllib.parse
ROOT = "data/fungi_big"; N = 300
NEW = ["Phytolacca americana", "Sambucus nigra", "Menispermum canadense",
       "Vitis riparia", "Atropa belladonna", "Vaccinium myrtillus"]
socket.setdefaulttimeout(20)
g = lambda u: json.load(urllib.request.urlopen(u, timeout=25))

for latin in NEW:
    slug = latin.replace(" ", "_"); d = f"{ROOT}/{slug}"; os.makedirs(d, exist_ok=True)
    got = len(glob.glob(f"{d}/*.jpg"))
    if got >= N:
        print(f"  {slug:26} {got} (enough)", flush=True); continue
    try:
        key = g("https://api.gbif.org/v1/species/match?name=" + urllib.parse.quote(latin)).get("usageKey")
    except Exception as e:
        print(f"  {slug}: match error {e}", flush=True); continue
    offset = 0
    while got < N and offset < N * 12:
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
    print(f"  {slug:26} {got}/{N}", flush=True)
print("done", flush=True)
