#!/bin/bash
# launch_v77.sh — HANDOFF v7.7 마스터 체인 런치 스크립트
# tmux 세션 v77_chain 내에서 실행 (터미널 꺼도 유지)
# 총 예상 시간: ~120h (순차, GPU 1개)
#
# 실행 순서:
#   1. 즉시군 병렬: exp7(백그라운드) + exp5 + exp1 + exp2 순차
#   2. report_immediate.py
#   3. exp3b 20 runs 순차
#   4. report_3b.py
#   5. exp3 20 runs 순차
#   6. exp4 20 runs 순차
#   7. report_3_4.py
#   8. exp6 20 runs (4 cells × 5 reps)
#   9. report_6.py
#  10. 완료 메시지

set -e

PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
V77=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77
LOG=$V77/launch_v77.log

mkdir -p "$V77"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== HANDOFF v7.7 실험 체인 시작 ==="
log "Python: $PYTHON"
log "V77: $V77"

# ── 즉시군 ────────────────────────────────────────────────────────────────────
log "[즉시군] exp7 (CPU, 백그라운드 시작)"
PYTHONUNBUFFERED=1 $PYTHON $V77/run_exp7_esf_sensitivity.py >> "$LOG" 2>&1 &
EXP7_PID=$!
log "[즉시군] exp7 PID=$EXP7_PID"

log "[즉시군] exp5 시작"
PYTHONUNBUFFERED=1 $PYTHON $V77/run_exp5_iso_profile.py >> "$LOG" 2>&1
log "[즉시군] exp5 완료"

log "[즉시군] exp1 시작"
PYTHONUNBUFFERED=1 $PYTHON $V77/run_exp1_infer_renorm.py >> "$LOG" 2>&1
log "[즉시군] exp1 완료"

log "[즉시군] exp2 시작"
PYTHONUNBUFFERED=1 $PYTHON $V77/run_exp2_real_fpr.py >> "$LOG" 2>&1
log "[즉시군] exp2 완료"

log "[즉시군] exp7 완료 대기..."
wait $EXP7_PID
log "[즉시군] exp7 완료 (PID=$EXP7_PID)"

log "[즉시군] 완료 → report_immediate"
PYTHONUNBUFFERED=1 $PYTHON $V77/report_immediate.py >> "$LOG" 2>&1
log "[즉시군] report_immediate 완료"

# ── 실험 3b (D_v5e) ───────────────────────────────────────────────────────────
log "[3b] D_v5e 학습 시작 (20 runs)"
for i in $(seq 0 19); do
    log "  [3b] run_idx=$i"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON $V77/run_exp3b_dv5e.py --run_idx "$i" >> "$LOG" 2>&1
done
log "[3b] 완료 → report_3b"
PYTHONUNBUFFERED=1 $PYTHON $V77/report_3b.py >> "$LOG" 2>&1
log "[3b] report_3b 완료"

# ── 실험 3 (sigma_post) ───────────────────────────────────────────────────────
log "[3] sigma_post 학습 시작 (20 runs)"
for i in $(seq 0 19); do
    log "  [3] run_idx=$i"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON $V77/run_exp3_sigma_post.py --run_idx "$i" >> "$LOG" 2>&1
done
log "[3] sigma_post 완료"

# ── 실험 4 (z-score) ─────────────────────────────────────────────────────────
log "[4] z-score 학습 시작 (20 runs)"
for i in $(seq 0 19); do
    log "  [4] run_idx=$i"
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PYTHON $V77/run_exp4_zscore.py --run_idx "$i" >> "$LOG" 2>&1
done
log "[4] z-score 완료"

log "[3+4] 완료 → report_3_4"
PYTHONUNBUFFERED=1 $PYTHON $V77/report_3_4.py >> "$LOG" 2>&1
log "[3+4] report_3_4 완료"

# ── 실험 6 (DeepLabV3+ 아키텍처 전이) ────────────────────────────────────────
log "[6] 아키텍처 전이 시작 (4 cells × 5 reps = 20 runs)"
for cell in 0 1 2 3; do
    for rep in 0 1 2 3 4; do
        log "  [6] cell=$cell rep=$rep"
        PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            $PYTHON $V77/run_exp6_arch.py --cell_id "$cell" --rep "$rep" >> "$LOG" 2>&1
    done
done
log "[6] 완료 → report_6"
PYTHONUNBUFFERED=1 $PYTHON $V77/report_6.py >> "$LOG" 2>&1
log "[6] report_6 완료"

log "=== HANDOFF v7.7 전 실험 완료 ==="
log "결과 위치: $V77"
log "로그: $LOG"
echo ""
echo "========================================="
echo " HANDOFF v7.7 체인 전 실험 완료!"
echo " $(date)"
echo " 보고서:"
echo "   $V77/report_immediate.md"
echo "   $V77/report_3b.md"
echo "   $V77/report_3_4.md"
echo "   $V77/report_6.md"
echo "========================================="
