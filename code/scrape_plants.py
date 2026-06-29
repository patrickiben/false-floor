"""SECOND CASE STUDY (kills the n=1 objection): the deadliest plant look-alike confusions, a non-fungal
safety decision with the same structure. Tests whether the representation-invariance protocol (§5.1) is
mushroom-specific or general. Apiaceae (carrot family) -- the classic fatal umbellifer confusions:
  Conium maculatum  -- POISON HEMLOCK (deadly; killed Socrates)
  Cicuta maculata   -- WATER HEMLOCK (deadly; most violently toxic plant in N. America)
  Daucus carota     -- WILD CARROT / Queen Anne's lace (EDIBLE) -- THE look-alike
  Anthriscus sylvestris -- COW PARSLEY (edible-ish umbellifer) -- a second edible / open-set
Same GBIF paged fetch. SAFETY: research vehicle; DON'T EAT.
"""
import glob, os, socket, json, urllib.request, urllib.parse
ROOT = "data/fungi_big"; N = 400      # reuse the same image store
NEW = ["Conium maculatum", "Cicuta maculata", "Daucus carota", "Anthriscus sylvestris"]
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
