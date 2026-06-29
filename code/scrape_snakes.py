"""THIRD case study, anchored to an EXTERNAL published claim: coral-snake mimicry. The venomous coral snake
(Micrurus) and its harmless mimics (scarlet kingsnake / milk snake, Lampropeltis) are a textbook Batesian
mimicry pair whose VISUAL SIMILARITY is documented in the peer-reviewed literature (Pfennig, Harper & Brodie,
Nature 2001; Brodie 1993) -- the "red touches yellow / red touches black" folk rhyme exists precisely because
they are near-indistinguishable to a quick look. We re-examine that external claim under the
representation-invariance test: does a strong frozen encoder separate what biology calls near-identical mimics?
A lethal decision (coral-snake envenomation can kill), third kingdom (reptiles). Same GBIF fetch. SAFETY:
research vehicle; never a field guide; if you cannot identify a snake with certainty, leave it alone.
"""
import glob, os, socket, json, urllib.request, urllib.parse
ROOT = "data/fungi_big"; N = 300
NEW = ["Micrurus fulvius", "Lampropeltis elapsoides", "Lampropeltis triangulum", "Cemophora coccinea"]
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
    print(f"  {slug:24} {got}/{N}", flush=True)
print("done", flush=True)
