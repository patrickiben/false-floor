# Reproducible pre-submission validation workflow

Everything done from the moment *"look into methods/workflows/agents/harnesses ... to be confident before
submission"* was asked, in order, with the exact scripts and commands so it can be re-run. This is the third
in a lineage of sibling runs — `~/Documents/Neuro_Atlas/VALIDATION_WORKFLOW.md` (consilience / GigaScience)
and `~/Documents/directed-asymmetry-contagion/VALIDATION_WORKFLOW.md` (DY connectedness) — applied to the
cautionary methods paper **"A False Floor."** Environments referenced:

- **python (day-to-day):** `~/Documents/Pattern1_LangGoal_Planner/.venv/bin/python` (3.9.6; numpy, scipy 1.13,
  hypothesis, statsmodels, sympy, mpmath). Any interpreter with `requirements.txt` works.
- **python (3.12+ path):** system `python3` (3.14.6) — used to reproduce a CPython-version-dependent bug (below).
- **repo / package:** `~/Documents/false-floor/` (public at github.com/patrickiben/false-floor). CP core:
  `code/cp_bounds.py`.
- **manuscript source (author working copy):** `~/Documents/EatThisMushroom/MANUSCRIPT_CAUTIONARY.md`
  (mirrored into `paper/`, which now builds standalone).
- **deposit:** OSF project `d3ft6` (anonymized view-only link in Appendix C); TMLR double-blind via OpenReview.

---

## 0. The reusable pattern (the transferable part)

1. **Research** the current method landscape; *web-verify* every tool is real and current before citing it.
2. **Tier** by return-on-effort (Tier 0 mechanical / Tier 2 code-correctness / Tier 4 AI-panel / packaging).
3. **Implement** the gates as small deterministic checks that emit a pass/fail (and JSON where useful).
4. **Verify** each against the *real* manuscript and code (run it; assert numbers; run from a cold clone).
5. **Integrate**: one-command gate + clean-clone CI; adversarially verify the soft parts; fix; deposit + push.

The single organizing idea, matching the siblings' "make the one fragile thing demonstrably stable": here the
fragile objects are **the near-zero-rate Clopper–Pearson bound at the `x=0` boundary** (the whole thesis is
"a stronger frozen encoder drives false-eat to ≈0%", so the bound at zero events is load-bearing), **the
cross-platform reproducibility of that bound**, and **the integrity and completeness of the reference list**.
Every gate below makes one of those demonstrably handled.

---

## Phase A — Research the methods

Lighter than the sibling runs' formal multi-agent research Workflow: a `WebSearch` sweep of the current
applied-math verification toolbox (property-based testing, differential testing, symbolic re-derivation,
SMT, metamorphic testing, method-of-manufactured-solutions, multiverse/specification-curve, statcheck/GRIM,
Lean/Mathlib, Snakemake, uv/pixi, Zenodo + Software Heritage, PaperBench/CORE-Bench), **with every source
resolved before it was cited** (this caught several fabrications in the draft reading list — a phantom
coauthor, a misattributed method, an offline tool). Output: a tiered playbook Artifact
(https://claude.ai/code/artifact/21339d1a-208c-4e11-b05b-71be42cf40fe) — tiers by effort, a gap analysis for
this paper, a hype-watch, and a fabrications ledger. From it, the two highest-ROI code checks (Phase C) and the
Tier-0 gates (Phase B) were selected. *Not yet done: the formal 7-facet research Workflow with a hostile
verifier the siblings ran; the playbook Artifact is the analogue.*

---

## Phase B — Tier 0: deterministic gates (never an LLM)

Ported from `directed-asymmetry-contagion/tools/` (the house Tier-0 set), adapted to this repo's layout.

### B1. Deterministic citation-integrity gate
**Script:** `tools/check_citations.py` (stdlib urllib only). Resolves each `refs.bib` entry via CrossRef /
OpenAlex / arXiv and scores by **containment of the record title within the full citation text**. Catches
fabrication, DOI hijacking (DOI → different paper), dead DOIs, retractions, year drift.
**Run:**
```bash
python3 tools/check_citations.py paper/refs.bib --json _reports/citation_report.json
```
**Result:** **14 OK / 2 REVIEW / 0 FLAG.** The 2 REVIEW are real papers CrossRef indexes poorly
(Bouthillier/MLSys, DINOv2/TMLR) — correctly separated from "does not exist", and independently confirmed real
by the Phase E adversarial workflow.
**Three robustness fixes were needed vs. the ported original** (the sibling uses inline `\bibitem`, not a
`.bib` file, so these never bit it — worth back-porting):
- **brace-matched `.bib` parser** — the original regex required each entry's closing brace on its own line;
  this repo's `refs.bib` is one entry per line, so it found *zero* entries. Now brace-counts (handles both
  styles + nested braces like `{WILDS}`).
- **DOI-suffix-as-year** — `symula2001`'s DOI `10.1098/rspb.2001.1812` made the extractor read "1812" as the
  year. Fixed by stripping the DOI before harvesting years.
- **DOI-less ML conference venues** (PMLR/ICML/NeurIPS/MLSys/TMLR/CVPR…) are not CrossRef-title-indexed →
  now classified REVIEW, never a hard FLAG (a gate that flags CLIP/WILDS as "possible fabrication" is wrong).

### B2. AI-artifact / hidden-text scrub
**Script:** `tools/scrub_ai_artifacts.py` — deterministic grep for assistant boilerplate / unfilled
placeholders + `pdftotext` hidden-text screen of the PDF. **Extended** with a pattern for *leaked tool-call
scaffolding* (`</content>`, `</invoke>`, `<parameter …>`) — the exact class of cruft that had leaked into the
end of `MANUSCRIPT_CAUTIONARY.md` and was removed by hand this arc.
```bash
python3 tools/scrub_ai_artifacts.py paper/MANUSCRIPT_CAUTIONARY.md paper/main.tex paper/_body.tex paper/_abstract.tex --pdf paper/main.pdf
```
**Result:** clean (0 source hits; PDF clean).

### B3. Absolute-path guard
**Script:** `tools/check_paths.sh` — fails on `/Users/`, `/home/<user>/`, `C:\`, etc. in executable code.
```bash
bash tools/check_paths.sh
```
**Result:** clean. **Related fix:** `paper/build_tmlr.sh` used to `cd ~/Documents/EatThisMushroom`, so a
reviewer's fresh clone could not rebuild the PDF. It is now **repo-relative** (derives its own directory,
`mktemp` temps, builds in place) — the repo builds standalone.

---

## Phase C — Tier 2: code-correctness (the load-bearing bound)

Every false-eat / false-safe rate is a one-sided 95% Clopper–Pearson upper bound, copy-pasted inline into 8
evaluation scripts. It was extracted to one canonical module and machine-checked two independent ways.

### C1. Property-based + differential + inline-consistency — `code/test_cp_bounds.py`
Hypothesis invariants (range, monotone in x & n, the `x=n`/`x=0` boundaries, defining eqn `P(Bin≤x)=α`);
**differential** agreement with `scipy.stats.beta.ppf`, `scipy.stats.binomtest`, and `statsmodels
proportion_confint(method='beta')` on every paper cell incl `x=0`; and a **behavioural** check that the
`cp_upper` inlined in all 8 scripts equals the canonical one.
```bash
cd code && python test_cp_bounds.py
```
**Result:** four implementations agree to **≤ 3.1e-13** across every cell including the `x=0` boundary; all 8
inline copies match canonical to **≤ 1e-12**.

### C2. Independent symbolic re-derivation (exact oracle) — `code/verify_cp_symbolic.py`
SymPy proves the Beta–binomial identity the bound rests on (36 `(n,x)` cases) and the `x=0` closed form
`1−α^(1/n)` (12 cases); mpmath cross-checks the numeric bound at 30 digits.
```bash
cd code && python verify_cp_symbolic.py
```
**Result:** identity + closed form proven; mpmath defining-equation residual ≤ 9.1e-15, `x=0` diff ≤ 7.4e-18.
**SymPy gotcha:** `sp.beta(int,int)` does **not** auto-evaluate — use the factorial form `x!(n−x−1)!/n!`,
integrate with `conds='none'`, and compare with `sp.expand(...)==0`.

---

## Phase D — clean-clone CI + one-command gate

### D1. CI: `.github/workflows/verify-stats.yml`
A cold `ubuntu-latest` runner on **Python 3.11 AND 3.12** installs only the light stats deps (no torch, no
image data) and runs the path guard, both CP checks, and the source scrub on every push touching the code /
tools / manuscript source. **CI is green.**

> **This gate paid for itself on its first run: it FAILED and caught a real bug the dev machine could not.**
> The inline-consistency check used exact `==`, but on Python 3.12+ `cp_upper` differed by ~1 ULP (2.8e-17):
> CPython 3.12 made `sum()` use compensated (Neumaier) summation, while two scripts accumulate the CDF with a
> manual `s += ` loop. The 3.9 dev venv passed; 3.12 caught it. Reproduced deterministically with system
> `python3` (3.14). Fixed: assert a `< 1e-12` tolerance (never bit-exact FP across platforms), cover all 8
> scripts, correct the false "byte-identical" docstrings.

### D2. One-command pre-submission gate: `tools/presubmit.sh`
Runs B1+B2+B3 + C1+C2 and prints one verdict.
```bash
PYTHON=~/Documents/Pattern1_LangGoal_Planner/.venv/bin/python bash tools/presubmit.sh
# -> RESULT: ALL CLEAR
```

---

## Phase E — Tier 4: adversarial citation resolution (multi-agent Workflow)

The deterministic gate (B1) is conservative and CrossRef-blind to some venues; it is complemented by an
adversarial LLM Workflow that resolves each reference against **DOI resolver / Crossref / PubMed / arXiv /
PMLR / DBLP** and then has a *second, skeptical* agent try to **refute** each resolution.

**Tool:** the `Workflow` harness. Run **wf_cd059b8c-211** (32 agents = 16 resolve → 16 verify;
first attempt wf_56795271-c2e died because `args` did not arrive as a JS array → entries were embedded in the
script instead). Script auto-saved under
`~/.claude/projects/.../workflows/scripts/verify-refs-bib-*.js`.
**Structure:** `pipeline(entries, resolve→schema, verify→schema)`; verdicts OK / MINOR_FIX / MAJOR_FIX /
FABRICATION_SUSPECTED / UNVERIFIABLE with exact corrected fields.
**Result:** **no fabrications**; every present DOI resolves; `kuznetsov2026` (2026, the recent one whose
author list was filled earlier) confirmed correct. One **MAJOR** and several enrichments (Phase F).

**Why both B1 and E:** the deterministic gate scores by title containment, so it **passed `vetter2000`**
(title matched perfectly) despite a wrong year/volume/issue; the adversarial pass **caught** it. Conversely
the adversarial pass confirmed the real papers (CLIP/WILDS/DINOv2/MLSys) that the deterministic gate can only
mark REVIEW. Deterministic = existence / DOI-hijack / dead-DOI; adversarial-LLM = metadata correctness.

---

## Phase F — fixes that fell out (applied)

- **`vetter2000` was factually wrong** → **1999, vol 5, issue 2** (was 2000/6/1; PubMed PMID 10673460 + the
  DOJ URL `DOJvol5num2`). Also fixed in the manuscript's hand-typed reference list.
- **Added verified DOIs** (symula, torralba, dacrema, assran, hendrycks) + confirmed page ranges. **Dropped
  the musgrave DOI** — its Springer suffix has an underscore, and `\doi{}` breaks because the TMLR template
  loads no `url` package (`tmlr.bst` falls back to literal `doi: #1`); the entry is otherwise complete.
- **Restored 3 references cited in the text but silently absent from the bibliography** — `mcgreivy2024`
  (the most on-point prior work), `darst2006`, `musgrave2020`. `build_tmlr.sh` injects citations by matching
  exact prose, and de-AI editing had introduced a line-wrap (Musgrave/Darst) and an en-dash (McGreivy
  `1256–1269`) that broke the regexes. Made those patterns robust (`\s+`, `[^)]*`). **All 16 entries now
  cited; `\bibitem` count 16, up from 13.**
- **Removed leaked `</content></invoke>` scaffolding** from the end of the manuscript source (now caught by B2).
- PDF rebuilt (12 pp, no undefined citations, no LaTeX errors).

---

## File inventory (created / changed in this arc)

| file | phase | purpose |
|---|---|---|
| `code/cp_bounds.py` | C | canonical Clopper–Pearson `cp_upper` |
| `code/test_cp_bounds.py` | C1 | Hypothesis property + differential + inline-consistency |
| `code/verify_cp_symbolic.py` | C2 | SymPy exact oracle + 30-digit mpmath cross-check |
| `tools/check_citations.py` | B1 | deterministic citation gate (ported + brace-parser + 3 fixes) |
| `tools/scrub_ai_artifacts.py` | B2 | AI-artifact / hidden-text scrub (ported + leaked-scaffolding pattern) |
| `tools/check_paths.sh` | B3 | absolute-path guard (ported verbatim) |
| `tools/presubmit.sh` | D2 | one-command pre-submission gate (adapted) |
| `.github/workflows/verify-stats.yml` | D1 | clean-clone CI (3.11+3.12): path guard + CP checks + scrub |
| `paper/build_tmlr.sh` | B3/F | made repo-relative (self-contained standalone build) |
| `paper/refs.bib` | F | Vetter fix + verified DOIs + page ranges |
| `paper/MANUSCRIPT_CAUTIONARY.md` | F | Vetter fix + removed leaked scaffolding |
| `VALIDATION_WORKFLOW.md` | 0 | this reproducible record |
| `README.md` (Verify section), `requirements.txt`, `.gitignore` | B–D | docs / deps / hygiene |

**Commits (in order):** `945c1d6` (CP checks + README + deps) → `8fdc560` (gitignore) → `3457a53` (CI gate)
→ `2c64601` (Neumaier tolerance fix) → `1652971` (references correctness) → *this arc* (Tier-0 tools +
portable build + CI extension + this doc). **Workflow runs:** wf_cd059b8c-211 (citation resolution, 32
agents); wf_56795271-c2e (failed first attempt).

---

## One-command re-run (everything)

```bash
D=~/Documents/false-floor
V=~/Documents/Pattern1_LangGoal_Planner/.venv/bin/python   # any python with the deps below
$V -m pip install numpy scipy hypothesis statsmodels sympy mpmath
cd "$D"
PYTHON=$V bash tools/presubmit.sh          # B+C+E-deterministic: path guard + CP tests + symbolic + citations + scrub -> ALL CLEAR
( cd paper && bash build_tmlr.sh )         # rebuild the PDF from source; prints the \bibitem count (want 16)
# Phase E adversarial workflow: re-invoke Workflow({scriptPath:"<saved verify-refs-bib-*.js>"})
```
The offline subset (path guard + CP checks + scrub) also runs in CI on every push.

---

## Generalizing to any applied-math manuscript (and what this run did NOT do)

Same pattern transfers; swap the domain pieces. Done here: property-based + differential + symbolic
re-derivation of the load-bearing statistic, deterministic Tier-0 gates, clean-clone matrix CI, and a
grounded adversarial citation Workflow.

**Deliberately NOT done (not applicable here, unlike the two sibling runs):**
- **Statistical design analysis (Type-S/M), TOST, FDR, null-calibration** — those siblings had a single
  fragile p-value / VAR effect. This paper's robustness *is* the multi-seed CIs + one-sided CP bounds + the
  encoder-swap that flips the conclusion (that encoder-dependence IS the specification-curve argument, and it
  lives in the paper). There is no lone barely-significant effect to retrodesign.
- **Method-of-manufactured-solutions / convergence order** — no numerical solver; the bound is closed-form.
- **Adversarial referee-panel Workflow (Phase F of the DY run)** — offered, not run in this arc.

**Three lessons this run contributes back to the house harness** (back-ported to the sibling docs):
1. **Run CI on the Python version matrix you ship on**; never assert cross-platform floating-point
   bit-equality (the CPython 3.12 `sum()`→Neumaier change).
2. **Deterministic and adversarial citation checks are complementary** — title containment passes a real
   paper with a wrong year (Vetter); the adversarial pass catches metadata errors; the deterministic pass
   catches fabrication/hijacking. Run both.
3. **When the build injects citations from prose (pandoc/regex), assert a post-build invariant**
   (`\bibitem` count == entries cited) — prose edits silently dropped three references.
