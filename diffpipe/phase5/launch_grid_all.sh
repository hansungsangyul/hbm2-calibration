#!/bin/bash
# Launch all 150 Phase 5 grid runs sequentially on one GPU.
# Usage: bash phase5/launch_grid_all.sh [start_cell] [end_cell]
#   default: cell 0..29, rep 0..4 (all 150 runs)
# Idempotent: each run exits early if results.json already exists.

set -uo pipefail

PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
SCRIPT=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5/run_phase5_grid.py
WORKDIR=/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation

CELL_START=${1:-0}
CELL_END=${2:-29}
EPOCHS=41
BATCH_SIZE=16   # reduced from cfg=32 to fit V100 32GB with lambda_max HVP overhead

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$WORKDIR"

total=0
skipped=0
ran=0

for cell in $(seq $CELL_START $CELL_END); do
    for rep in 0 1 2 3 4; do
        total=$((total+1))
        OUT_DIR="/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5/runs"
        i_sigma=$((cell / 6))
        j_fwhm=$((cell % 6))
        result_path="${OUT_DIR}/sg${i_sigma}_fw${j_fwhm}/rep${rep}/results.json"
        if [ -f "$result_path" ]; then
            echo "[SKIP] cell=$cell rep=$rep — results.json exists"
            skipped=$((skipped+1))
            continue
        fi
        echo "===== cell=$cell rep=$rep ====="
        if $PYTHON "$SCRIPT" --cell_index "$cell" --rep "$rep" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"; then
            ran=$((ran+1))
        else
            echo "[FAILED] cell=$cell rep=$rep (exit $?) — skipping"
        fi
    done
done

echo ""
echo "Done. total=$total  ran=$ran  skipped=$skipped"
