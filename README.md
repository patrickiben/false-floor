# A False Floor

Code and manuscript for *"A False Floor: How a Weak Baseline and a Curated Benchmark Produced an 'Information Limit' on a Lethal Recognition Task that a Stronger Baseline then Refuted."*

A cautionary methods study. An investigation concluded that "should I eat this mushroom?" is **information-limited** — a strong aggregate separability coexisting with an irreducible confidently-wrong residual no method removed. Swapping the frozen, mean-pooled I-JEPA backbone for a stronger frozen encoder (DINOv2) refuted that conclusion. The same pattern reproduces across **six lethal look-alike decisions in five biological groups** (fungi, plants, reptiles, arachnids, amphibians), three of them re-examinations of externally-published indistinguishability claims (coral-snake mimicry, brown-recluse misidentification, poison-frog mimicry). The lesson: a limitation claim is only as strong as the baseline that fails to beat it, and a verification discipline built to catch false positives is blind to false negatives.

> **Safety.** This is a research vehicle for studying ML evaluation, not foraging guidance. Nothing here endorses identifying wild mushrooms, plants, snakes, spiders, or frogs with a model. When in doubt, do not eat / do not handle / consult an expert.

## Layout
- `paper/` — TMLR LaTeX source (`main.tex`, `refs.bib`, `tmlr.sty`, `build_tmlr.sh`), the source `MANUSCRIPT_CAUTIONARY.md`, and the compiled `main.pdf`.
- `code/` — all experiment scripts (encoders, the six case studies, scrapers, the objective-vs-aggregation ablation, the original-investigation method sweep).
- `companion/` — `MANUSCRIPT.md` (the original investigation; Appendix B of the paper) and `FINDINGS.md`.
- `DATA_MANIFEST.txt` — the GBIF species and image counts used.

## Reproduce
1. `pip install -r requirements.txt`
2. Scrape the images from GBIF: `python code/scrape_hard.py`, `code/scrape_plants.py`, `code/scrape_snakes.py`, `code/scrape_more.py`, `code/scrape_berries.py` (writes to `data/fungi_big/`). Images are **not** redistributed here (GBIF licensing, ~29 GB); the scrapers reproduce them via the GBIF `species/match` + `occurrence/search` API.
3. Run the case studies, e.g. `python code/pairs_encoders.py` (five-pair grid), `code/hard_pairs.py` (needs-a-microscope pairs), `code/case3_snakes.py` … `code/case6_berries.py`, `code/ablation_agg.py`. Each prints AUROC / false-eat with one-sided 95% Clopper-Pearson bounds. Set `FULL=1` for full-image runs; `CROP=1` for the substrate-strip control.
4. Build the PDF: `cd paper && ./build_tmlr.sh` (needs pandoc + a LaTeX distribution).

Encoders: frozen `facebook/ijepa_vith14_1k`, `facebook/dinov2-large`, `openai/clip-vit-large-patch14` (Hugging Face). MPS/CUDA/CPU auto-selected.
