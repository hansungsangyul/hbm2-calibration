#!/usr/bin/env bash
# HANDOFF v7.8 실행 체인
# §0 완료 (보고서 v2 직접 작성)
# §1 → §2 → §3 (진단) → §3 (본실행, 진단 통과 시)
set -e

PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
V77=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77
LOG=${V77}/launch_v78.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== HANDOFF v7.8 실행 체인 시작 ==="
log "Python: $PYTHON"
log "V77: $V77"

# ── §1: 실험 1 OOD 보완 ──────────────────────────────────────────────────────
log "[§1] exp1 OOD 보완 시작"
$PYTHON ${V77}/run_exp1_ood_supplement.py 2>&1 | tee -a "$LOG"
log "[§1] exp1 OOD 보완 완료"

# ── §2: 실험 2 전처리 수정 재실행 ─────────────────────────────────────────────
log "[§2] exp2 재실행 시작"
$PYTHON ${V77}/run_exp2_real_fpr_v2.py 2>&1 | tee -a "$LOG"
log "[§2] exp2 재실행 완료"

# ── §3: 실험 6 진단 ──────────────────────────────────────────────────────────
log "[§3] exp6 진단 시작 (LR sweep × 6 runs)"
$PYTHON ${V77}/run_exp6_diag.py 2>&1 | tee -a "$LOG"
log "[§3] exp6 진단 완료"

# ── §3: 진단 결과 확인 → 본실행 ──────────────────────────────────────────────
DIAG_JSON=${V77}/exp6_diag_summary.json
VERDICT=$($PYTHON -c "import json; d=json.load(open('${DIAG_JSON}')); print(d['verdict'])")
BEST_LR=$($PYTHON -c "import json; d=json.load(open('${DIAG_JSON}')); print(d['best_lr'] or 'none')")

log "[§3] 진단 판정: ${VERDICT}  best_lr=${BEST_LR}"

if [ "$VERDICT" = "COMPETENT_FOUND" ] && [ "$BEST_LR" != "none" ]; then
    log "[§3] 본실행 시작 (4 cells × 5 reps, lr=${BEST_LR})"
    for cell in 0 1 2 3; do
        for rep in 0 1 2 3 4; do
            log "  [§3] cell=${cell}  rep=${rep}"
            $PYTHON ${V77}/run_exp6_v2.py \
                --cell_id ${cell} --rep ${rep} --lr ${BEST_LR} \
                2>&1 | tee -a "$LOG"
        done
    done
    log "[§3] 본실행 완료 → report_6_v3"
    $PYTHON -c "
import json, numpy as np
from pathlib import Path
V77 = Path('${V77}')
rows = []
for p in sorted((V77/'exp6_v2_runs').glob('*/results.json')):
    d = json.load(open(p))
    rows.append(d)
print(f'exp6_v2: {len(rows)} runs loaded')
" 2>&1 | tee -a "$LOG"
else
    log "[§3] 유능 config 미달 — M8 수용. 본실행 생략."
    log "[§3] 기여3 하향 (report_6_v2.md 판정 INVALID_EXECUTION 유지)"
fi

log "=== v7.8 체인 완료 ==="
