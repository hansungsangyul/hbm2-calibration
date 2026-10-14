#!/bin/bash
# launch_p43.sh — P4.3 절제 실험 런처 (HANDOFF v7.6 §1 지시67)
# 60 runs: AB-1(20) + AB-2(20) + AB-3(20), §1 착수 전 §2 완료 확인 후 실행
# 예상 시간: ~72h
# Usage: tmux 세션 내 실행 권장

PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
SCRIPT=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_p43_ablation.py
LOG_DIR=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p43_logs

mkdir -p "$LOG_DIR"

echo "P4.3 절제 실험 시작 ($(date))"
echo "Python: $PYTHON"
echo "Script: $SCRIPT"

# 함수: 단일 run 실행
run_one() {
    local cond=$1
    local idx=$2
    local logfile="$LOG_DIR/${cond}_run${idx}.log"
    echo "[$(date +%H:%M)] Starting ${cond}/run${idx}"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON $SCRIPT --condition $cond --run_idx $idx \
        > "$logfile" 2>&1
    echo "[$(date +%H:%M)] Done    ${cond}/run${idx}  (exit $?)"
}

# AB-1 (20 runs: fwhm=2.5, prenorm=True, seeds 770000-770019)
echo ""
echo "=== AB-1: fwhm=2.5, prenorm=True, seeds 770000-770019 ==="
for idx in $(seq 0 19); do
    run_one ab1 $idx
done

# AB-2 (20 runs: fwhm=2.0956, prenorm=False, seeds 780000-780019)
echo ""
echo "=== AB-2: fwhm=2.0956, prenorm=False, seeds 780000-780019 ==="
for idx in $(seq 0 19); do
    run_one ab2 $idx
done

# AB-3 (20 runs: fwhm=2.0956, prenorm=False, seeds 790000-790019)
echo ""
echo "=== AB-3: fwhm=2.0956, prenorm=False, seeds 790000-790019 ==="
for idx in $(seq 0 19); do
    run_one ab3 $idx
done

echo ""
echo "=== P4.3 절제 실험 완료 $(date) ==="
echo "결과: /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p43_runs/"
