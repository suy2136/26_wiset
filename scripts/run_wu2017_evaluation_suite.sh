#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"
case "$MODE" in
  smoke)
    export LIMIT_TEST_SAMPLES="${LIMIT_TEST_SAMPLES:-10}"
    echo "Wu2017 smoke evaluation: first $LIMIT_TEST_SAMPLES samples per case"
    ;;
  full)
    unset LIMIT_TEST_SAMPLES
    echo "Wu2017 full evaluation"
    ;;
  *)
    echo "Usage: bash scripts/run_wu2017_evaluation_suite.sh {smoke|full}"
    exit 2
    ;;
esac

bash scripts/run_rule_based_jin_wu.sh
RULE_RUN_DIR="$(tr -d '\r\n' < viewport_prediction/data/experiment_runs/rule_based_jin_wu_latest.txt)"

bash scripts/run_wu2017_unseen_models.sh
MODEL_RUN_DIR="$(tr -d '\r\n' < viewport_prediction/data/experiment_runs/unseen_wu2017_latest.txt)"

OUTPUT_DIR="$MODEL_RUN_DIR/comparison"
python analysis/plot_wu2017_unseen_report.py \
  --rule-run-dir "$RULE_RUN_DIR" \
  --model-run-dir "$MODEL_RUN_DIR" \
  --output-dir "$OUTPUT_DIR"

printf '{"mode":"%s","status":"complete","rule_run_dir":"%s","model_run_dir":"%s","comparison_dir":"%s"}\n' \
  "$MODE" "$RULE_RUN_DIR" "$MODEL_RUN_DIR" "$OUTPUT_DIR" \
  > "$MODEL_RUN_DIR/suite_status.json"
echo "Wu2017 $MODE suite complete: $MODEL_RUN_DIR"
