#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TARGET_DIR:?Set TARGET_DIR to the existing Llama-3.3-70B-Instruct directory}"
: "${DRAFT_DIR:?Set DRAFT_DIR to the existing matching SpecForge EAGLE3 directory}"
OUT_ROOT="${OUT_ROOT:-results/llama70b_tp8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MIN_OUTPUT="${MIN_OUTPUT:-256}"
mkdir -p "$OUT_ROOT"
python scripts/build_dataset.py --check
if [[ ! -f .runtime/install_receipt.json ]]; then
    python tools/install_runtime.py
fi
if [[ ! -f "$OUT_ROOT/prepared4000.json" ]]; then
    python main.py prepare --model "$TARGET_DIR" --output "$OUT_ROOT/prepared4000.json"
fi
COMMON=(--model "$TARGET_DIR" --draft "$DRAFT_DIR" --gpus "$GPU_IDS")
python main.py calibrate "${COMMON[@]}" --data "$OUT_ROOT/prepared4000.json" \
    --output "$OUT_ROOT/calibration" --steady-output "$OUT_ROOT/steady4000.json" \
    --minimum-output "$MIN_OUTPUT" --wait-free --resume
for STAGE in smoke timing kernel; do
    python main.py run "${COMMON[@]}" --data "$OUT_ROOT/steady4000.json" \
        --output "$OUT_ROOT/formal" --stage "$STAGE" --wait-free --resume
done
python main.py report --output "$OUT_ROOT/formal"
echo "Completed. Read $OUT_ROOT/formal/REPORT.md"
echo "Return the entire $OUT_ROOT directory, including calibration and raw traces."
