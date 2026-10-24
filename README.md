# A False Floor

**Current paper and reproducibility records (23 October 2026):**
*A False Floor: Refuting Our Own Information-Limit Claim in a Lethal Identification Task*
was submitted to PLOS ONE on 22 October 2026. Read the
[current preprint](https://doi.org/10.5281/zenodo.23578222) and
[reproducibility records](https://doi.org/10.5281/zenodo.23569075).

This private repository preserves an earlier TMLR-era source snapshot. Its
manuscript, titles, experiment framing, and validation ledger are historical;
they are not the current submitted paper. The six previously local commits
through 2 September are retained on
[`archive/local-2026-09-02`](https://github.com/patrickiben/false-floor/tree/archive/local-2026-09-02),
including per-seed counts and validation fixes. They have not been promoted to
the current manuscript. Use the DOI records above for the current scope and
archived inputs.

## Educational companions

Choose the [concise study guide](education/STUDY_GUIDE.md) ([PDF](education/STUDY_GUIDE.pdf))
for the overview, or the [complete passage-by-passage companion](education/COMPANION.md)
([PDF](education/COMPANION.pdf)) for the full current-paper and S1 teaching route.
The [education notes](education/README.md) and [coverage map](education/COMPLETE_COVERAGE.md)
give exact source provenance and review limits. This repository remains private.

## Historical snapshot

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

## Verify the statistics

Every false-eat / false-safe rate in the paper is reported with a one-sided 95% Clopper-Pearson upper bound, and the paper's claims live at rates near zero, so the `x=0` boundary of that bound is load-bearing. The from-scratch implementation (`code/cp_bounds.py`, copied inline into each evaluation script) is machine-checked three ways, no image data needed:

- `python code/test_cp_bounds.py` — **property-based** (Hypothesis: range, monotonicity, the `x=0`/`x=n` boundaries, and the defining equation `P(Bin(n,p) <= x) = alpha`) and **differential** (agreement to `< 1e-6` with `scipy.stats.beta.ppf`, `scipy.stats.binomtest`, and `statsmodels` `proportion_confint(method='beta')` on every `(x, n)` cell in the paper). It also confirms the `cp_upper` inlined in the evaluation scripts is behaviourally identical to the canonical one, so the test covers the code that produced the numbers.
- `python code/verify_cp_symbolic.py` — **symbolic re-derivation** (SymPy proves the Beta-binomial identity the bound rests on, and the `x=0` closed form `1 - alpha^(1/n)`) plus a **30-digit** mpmath cross-check independent of the float bisection.

For the full pre-submission harness — a deterministic citation-integrity gate (`tools/check_citations.py`), an AI-artifact / hidden-text scrub (`tools/scrub_ai_artifacts.py`), an absolute-path guard (`tools/check_paths.sh`), plus the two checks above — run the one-command gate:

```bash
bash tools/presubmit.sh      # -> RESULT: ALL CLEAR
```

The offline subset (path guard + both CP checks + source scrub) also runs in CI on every push (`.github/workflows/verify-stats.yml`, Python 3.11 + 3.12). The reproducible record of the whole validation arc is in [`VALIDATION_WORKFLOW.md`](VALIDATION_WORKFLOW.md).
