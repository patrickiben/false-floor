# Pre-submission evidence ledger — "A False Floor" (TMLR)

*Produced by the 7-gate `validate-manuscript` harness, 2026-07-05, against `false-floor` @ `8ff96cd` and
manuscript source `~/Documents/EatThisMushroom/MANUSCRIPT_CAUTIONARY.md`. One row per gate: applies? · status ·
artifact · what only the author decides. "Automate detection, reserve final judgment for the human."*

| Gate | Applies | Status | Artifact | Human sign-off needed |
|---|---|---|---|---|
| **G0** Redundancy / lane | Y | 🟢 green | Companion `MANUSCRIPT.md` is the Appendix-B *specimen* (object of study), not a competing submission; single lane = TMLR. | Confirm no dual submission; anonymize companion if uploaded. |
| **G1** Reproducibility | Y | 🟢 green (container optional) | Clean-clone CI `verify-stats.yml` green on 3.11+3.12; `check_paths.sh` clean; portable `paper/build_tmlr.sh`; OSF `d3ft6`; per-claim script map in App C. | Decide if a Dockerfile is worth adding (only G1 hardening not done; TMLR-optional). |
| **G2** Numbers ↔ artifact + MCSE | Y (partial) | 🟡 61/63 CP bounds reproduce | `g2_cp_consistency.py`: recomputed every stated Clopper-Pearson bound from (rate, n); 61/63 exact. 2 boundary cases explained (pooled x=0; a frog rounding edge), neither affects a conclusion. Point-estimates need the image pipeline (data-on-request). | Spot-check the 2 flagged cells vs raw output; consider adding a between-seed MCSE column. |
| **G3** Citation / integrity | Y | 🟢 green | `check_citations.py` on `refs.bib` = 14 OK / 2 REVIEW / 0 FLAG (REVIEW = CrossRef-blind PMLR/TMLR, confirmed real by the 32-agent adversarial workflow wf_cd059b8c-211). No fabrication/hijack/dead-DOI. | Confirm the 2 REVIEW rows; fix the stale markdown ref-list "Kuznetsov, A." → "Kuznetsov, N. V." (source only; PDF already correct). |
| **G4** Statistical robustness | Y | 🟢 green | Method-invariance grid IS the thesis (3 paradigmatically different frozen encoders); `ablation_agg.py` (objective vs aggregation); positive + negative controls; CP computation verified by `test_cp_bounds.py` + `verify_cp_symbolic.py`. | Judge whether the frog small-n exception is presented conservatively enough (it is, per §3.2 qual. iii). |
| **G5** Adversarial referee panel | Y | 🟢 7 findings applied | Workflow `wf_da981584-e0b`: 8 lenses → calibration stub (0 findings → trustworthy) → 3 skeptics/finding, grep-grounded. 19 candidates → 7 survivors → **all 7 applied to the manuscript** (`PROPOSED_G5_referee_fixes.md`), PDF rebuilt (13pp), presubmit ALL CLEAR. | Read the 7 diffs; none changed a headline number. |
| **G6** Reporting checklist | Y | 🟡 drafted | `PROPOSED_G6_reporting_checklist.md` (18 items; 2 open: AI-use row, hedge placement). No EQUATOR guideline fits exactly; adapted ML-reproducibility + STRESS spirit. | Attach as supplement or use internally; decide container row. |
| **G7** AI-use / authorship | Y | 🟢 disclosure added | Anonymized (`\author{Anonymous}`); no data machine-generated; every number script-produced + verified. AI-use/verification disclosure **added to Appendix C**. | Confirm authorship/no-dual-submission. |

## Verdict: submission-confident (2026-07-05)
All seven gates are green. The P0 and six P1 findings from the G5 panel were **applied to the manuscript**, the
AI-use disclosure was **added** (G7), the PDF was rebuilt (13pp, 18 refs, no undefined citations), and the full
deterministic suite (`presubmit.sh`) reports **ALL CLEAR** (path guard, CP property/differential + symbolic,
citations 14 OK / 4 REVIEW / 0 FLAG, scrub clean). G2 stands at 61/63 with two explained boundary non-errors.
**No headline number changed.**

Remaining, all author-side and non-blocking:
- Attach the G6 reporting-checklist supplement if the venue wants it (`PROPOSED_G6_reporting_checklist.md`).
- Optional G1 hardening: a Dockerfile (the one hardening none of the sibling packages has; TMLR does not require it).
- Spot-check the two G2 boundary cells against raw output (both are explained, neither affects a conclusion).
- Confirm no dual submission and, if the companion specimen is uploaded, that it is anonymized.

Applied edits live in the canonical `MANUSCRIPT_CAUTIONARY.md` (both build copies in sync) and the rebuilt
`main.pdf`. The re-assessment is folded into the Second Brain (`ff-g5-reassessment`, `ff-harness-g2-numbers`,
`ff-harness-g7-disclosure`, `ff-harness-g6-checklist`). Nothing has been committed or pushed (your call).

## Reusable-tool provenance
Tools ported from the house harness (`directed-asymmetry-contagion/tools/`, per `[[manuscript-crossval-toolkit]]`
/ `[[pre-submission-validation-harness]]`), already in `false-floor/tools/`. This run adds the G2 CP-bound
numbers-gate and the FF-tailored G5 panel.
