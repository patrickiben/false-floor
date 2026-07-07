"""MORE external-anchored cases (spiders + poison frogs). Two more kingdoms with citable look-alike
claims:
  SPIDERS -- brown recluse (Loxosceles reclusa, medically significant) vs the harmless house/wolf spiders
    constantly mistaken for it. External claim: the recluse-MISIDENTIFICATION literature (Vetter 2008,
    'Arachnophobic entomologists...'; Vetter & Bush) -- non-experts cannot visually distinguish them, recluses
    are massively over-reported. Decision: 'is this a dangerous recluse?'
  FROGS -- poison-frog mimicry. Mimic poison frog (Ranitomeya imitator) vs its near-identical model species
    (Symula et al., Nature 2001 -- the textbook vertebrate Mullerian mimic); plus the Batesian case where a
    NON-toxic frog (Allobates zaparo) mimics a TOXIC model (Ameerega) -- Darst & Cummings, Nature 2006.
Same GBIF fetch; the test scripts skip any species with <40 images. SAFETY: research vehicle; if unsure, leave it alone.
"""
import glob, os, socket, json, urllib.request, urllib.parse
ROOT = "data/fungi_big"; N = 300
NEW = [
    # spiders
    "Loxosceles reclusa", "Kukulcania hibernalis", "Rabidosa rabida", "Eratigena agrestis",
    # frogs
    "Ranitomeya imitator", "Ranitomeya variabilis", "Ranitomeya fantastica",
    "Allobates zaparo", "Ameerega bilinguis", "Ameerega parvula",
]
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
