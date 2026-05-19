#!/usr/bin/env bash
# Prepare Zenodo release archive for IEEE TLT submission.
# Run from repo root after all data collection is complete.
#
# Usage:
#   bash scripts/prepare_zenodo_release.sh [--dry-run]
#
# Output:
#   zenodo_release/
#     README.md                    — release notes
#     data/item_bank/              — 1,500 items (all qtypes)
#     outputs/qa_responses_*.csv  — aggregated P1 responses (no full JSONs)
#     outputs/unlearning_eval.csv — P2 evaluation summary
#     outputs/human_annotations_v2.csv
#     outputs/icc_report_v2.csv
#     outputs/eval_*.csv          — D1–D7 eval outputs
#     prompts/                    — all prompts + format_suffixes + demographic_variants
#     src/                        — all source code
#     reports/analysis_plan.md    — pre-registered analysis plan v2.0
#     reports/tlt_paper_draft.tex — paper draft
#     MODELCARDS.md
#     data/DATASHEET_v2.md
#     CITATION.cff

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[DRY RUN] No files will be copied."
fi

RELEASE_DIR="zenodo_release"

if [ "$DRY_RUN" = false ]; then
  rm -rf "$RELEASE_DIR"
  mkdir -p "$RELEASE_DIR"/{data/item_bank,outputs,src,prompts,reports}
fi

copy_if() {
  local src="$1"; local dst="$2"
  if [ "$DRY_RUN" = true ]; then
    echo "WOULD COPY: $src → $dst"
  elif [ -e "$src" ]; then
    cp -r "$src" "$dst"
  else
    echo "  [SKIP] Not found: $src"
  fi
}

echo "=== Item bank ==="
for qtype in mcq tf fill sa; do
  copy_if "data/item_bank/$qtype/items_with_difficulty.jsonl" \
          "$RELEASE_DIR/data/item_bank/${qtype}_items.jsonl"
done

echo "=== Aggregated P1 responses (CSVs only, no full JSONs) ==="
for csv in outputs/qa_responses_*.csv; do
  copy_if "$csv" "$RELEASE_DIR/outputs/"
done

echo "=== P2 unlearning evaluation ==="
copy_if "outputs/unlearning_eval.csv" "$RELEASE_DIR/outputs/"

echo "=== Human annotations ==="
copy_if "outputs/human_annotations_v2.csv" "$RELEASE_DIR/outputs/"
copy_if "outputs/icc_report_v2.csv"        "$RELEASE_DIR/outputs/"

echo "=== Evaluation outputs ==="
for eval_csv in outputs/eval_d*.csv; do
  copy_if "$eval_csv" "$RELEASE_DIR/outputs/"
done

echo "=== Source code ==="
copy_if "src/" "$RELEASE_DIR/"

echo "=== Prompts ==="
copy_if "prompts/" "$RELEASE_DIR/"

echo "=== Reports ==="
copy_if "reports/analysis_plan.md"       "$RELEASE_DIR/reports/"
copy_if "reports/tlt_paper_draft.tex"    "$RELEASE_DIR/reports/"
copy_if "reports/references_tlt.bib"     "$RELEASE_DIR/reports/"

echo "=== Metadata ==="
copy_if "MODELCARDS.md"        "$RELEASE_DIR/"
copy_if "data/DATASHEET_v2.md" "$RELEASE_DIR/"
copy_if "CITATION.cff"         "$RELEASE_DIR/"

echo "=== Verification ==="
if [ "$DRY_RUN" = false ]; then
  echo "Release directory contents:"
  find "$RELEASE_DIR" -type f | sort
  echo ""
  echo "Total files: $(find "$RELEASE_DIR" -type f | wc -l)"
  echo "Total size:  $(du -sh "$RELEASE_DIR" | cut -f1)"
fi

echo ""
echo "=== NEXT STEPS ==="
echo "1. Upload LoRA adapters to HuggingFace Hub:"
echo "   huggingface-cli upload --repo-type model YOUR_ORG/naive-student-mistral-unlearned \\"
echo "     outputs/unlearning/mistral_7b_instruct_v0_3/"
echo ""
echo "2. Create Zenodo deposit:"
echo "   - Upload zenodo_release/ as zip"
echo "   - Set license: CC-BY-4.0 for data, MIT for code"
echo "   - Copy DOI and insert into:"
echo "     * MODELCARDS.md"
echo "     * data/DATASHEET_v2.md"
echo "     * reports/tlt_paper_draft.tex (\\reproducibility section)"
echo "     * CITATION.cff"
echo ""
echo "3. Tag the release in git:"
echo "   git tag v2.0-tlt-submission"
echo "   git push origin v2.0-tlt-submission"
