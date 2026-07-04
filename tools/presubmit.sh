#!/usr/bin/env bash
# One-command pre-submission gate for the false-floor package.
#
#   bash tools/presubmit.sh
#
# Runs every deterministic check in one pass and reports a combined verdict:
#   [code ]  absolute-path guard          (reproduction: no machine-specific paths)
#   [code ]  Clopper-Pearson property+differential test  (correctness of every reported rate)
#   [code ]  Clopper-Pearson symbolic re-derivation      (exact oracle for the bound's identity)
#   [paper]  citation-integrity gate      (integrity: every reference resolves; no fabrication/hijack) [network]
#   [paper]  AI-artifact / hidden-text     (integrity: no leftover AI boilerplate/placeholders/scaffolding)
#
# The code checks + the source scrub also run in CI on every push (.github/workflows/verify-stats.yml);
# this bundles them with the network citation gate so you get a single green light before submit.
# Requires: python deps in requirements.txt (numpy scipy hypothesis statsmodels sympy mpmath).
# Exit 0 = all clear.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$REPO/paper"
PY="${PYTHON:-python3}"
fail=0

hr() { printf '%s\n' "------------------------------------------------------------------------------"; }
run() { echo; echo ">> $1"; hr; shift; "$@"; local rc=$?; if [ $rc -ne 0 ]; then fail=1; echo "   [FAILED rc=$rc]"; fi; return 0; }

echo "=============================================================================="
echo " PRE-SUBMISSION GATE   repo: $REPO"
echo "=============================================================================="

cp_test()    { ( cd "$REPO/code" && "$PY" test_cp_bounds.py ); }
cp_symbolic(){ ( cd "$REPO/code" && "$PY" verify_cp_symbolic.py ); }

run "code  · absolute-path guard"               bash "$REPO/tools/check_paths.sh" "$REPO"
run "code  · CP property + differential test"   cp_test
run "code  · CP symbolic re-derivation"         cp_symbolic

mkdir -p "$REPO/_reports"
run "paper · citation-integrity gate [network]" "$PY" "$REPO/tools/check_citations.py" \
      "$PAPER/refs.bib" --json "$REPO/_reports/citation_report.json"

SCRUB=("$PAPER/MANUSCRIPT_CAUTIONARY.md" "$PAPER/main.tex" "$PAPER/_body.tex" "$PAPER/_abstract.tex")
if [ -f "$PAPER/main.pdf" ]; then
  run "paper · AI-artifact / hidden-text"       "$PY" "$REPO/tools/scrub_ai_artifacts.py" "${SCRUB[@]}" --pdf "$PAPER/main.pdf"
else
  run "paper · AI-artifact scan"                "$PY" "$REPO/tools/scrub_ai_artifacts.py" "${SCRUB[@]}"
fi

echo; echo "=============================================================================="
if [ $fail -eq 0 ]; then
  echo " RESULT: ALL CLEAR — deterministic pre-submission checks pass"
else
  echo " RESULT: ATTENTION NEEDED — one or more checks reported items above"
fi
echo "=============================================================================="
exit $fail
