#!/bin/bash
# harness/run_all.sh
# Runs the full benchmark protocol: 4 concurrency levels x 5 runs = 20 CSVs
#
# Usage:
#   bash harness/run_all.sh <endpoint> <config-name>
#
# Example:
#   bash harness/run_all.sh https://triton-resnet50-xxxxx-uc.a.run.app cloudrun
#
# Output:
#   results/<config-name>/results_<config>_<concurrency>_<run>.csv
#   20 CSV files total

set -e  # exit immediately if any command fails

# ── Arguments ────────────────────────────────────────────────────────────────
ENDPOINT=$1
CONFIG=$2

if [ -z "$ENDPOINT" ] || [ -z "$CONFIG" ]; then
  echo "Usage: bash harness/run_all.sh <endpoint> <config-name>"
  echo "  endpoint:    Triton server URL (e.g. https://...uc.a.run.app)"
  echo "  config-name: one of: gpu cpu confidential cloudrun cloudfunction"
  exit 1
fi

# ── Settings (match PROTOCOL.md exactly) ─────────────────────────────────────
CONCURRENCY_LEVELS=(1 10 50 100)
RUNS=5
REQUESTS=200
WARMUP=20
OUTPUT_DIR="results/${CONFIG}"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "GCP Inference Benchmark — run_all.sh"
echo "Config:      $CONFIG"
echo "Endpoint:    $ENDPOINT"
echo "Concurrency: ${CONCURRENCY_LEVELS[*]}"
echo "Runs:        $RUNS per concurrency level"
echo "Requests:    $REQUESTS per client ($WARMUP warmup discarded)"
echo "Output:      $OUTPUT_DIR/"
echo "=================================================="
echo ""

# ── Track overall progress ────────────────────────────────────────────────────
TOTAL=$((${#CONCURRENCY_LEVELS[@]} * RUNS))   # 4 x 5 = 20
COMPLETED=0
FAILED_RUNS=()

START_TIME=$(date +%s)

# ── Main loop ─────────────────────────────────────────────────────────────────
for C in "${CONCURRENCY_LEVELS[@]}"; do
  echo ""
  echo "──────────────────────────────────────────────────"
  echo "Concurrency: $C clients"
  echo "──────────────────────────────────────────────────"

  for R in $(seq 1 $RUNS); do
    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "Run $R / $RUNS  (overall: $COMPLETED / $TOTAL)"

    # Run the harness — capture exit code without triggering set -e
    python harness/harness.py \
      --endpoint "$ENDPOINT" \
      --config-name "$CONFIG" \
      --concurrency "$C" \
      --run "$R" \
      --requests "$REQUESTS" \
      --warmup "$WARMUP" \
      --output "$OUTPUT_DIR" \
    && STATUS=0 || STATUS=$?

    if [ $STATUS -ne 0 ]; then
      echo "  ⚠️  Run failed (exit code $STATUS). Logging for retry."
      FAILED_RUNS+=("c${C}_r${R}")
    fi

    # Brief pause between runs to let the server stabilize
    if [ "$R" -lt "$RUNS" ] || [ "$C" -ne "${CONCURRENCY_LEVELS[-1]}" ]; then
      echo "  Pausing 5s before next run..."
      sleep 5
    fi

  done
done

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=================================================="
echo "All runs complete"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "CSVs written to: $OUTPUT_DIR/"
echo ""

CSV_COUNT=$(ls "$OUTPUT_DIR"/results_*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "CSV files found: $CSV_COUNT / $TOTAL expected"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
  echo ""
  echo "⚠️  Failed runs (check error rate — may need re-running):"
  for RUN in "${FAILED_RUNS[@]}"; do
    echo "   $RUN"
  done
  echo ""
  echo "Per protocol: re-run each failed run once."
  echo "Re-run example:"
  echo "  python harness/harness.py --endpoint $ENDPOINT --config-name $CONFIG \\"
  echo "    --concurrency <C> --run <R> --requests $REQUESTS --warmup $WARMUP \\"
  echo "    --output $OUTPUT_DIR"
else
  echo ""
  echo "✅ All $TOTAL runs completed successfully."
fi

echo ""
echo "Next step: commit CSVs to the shared repo"
echo "  git add $OUTPUT_DIR/"
echo "  git commit -m 'Add $CONFIG benchmark results'"
echo "=================================================="