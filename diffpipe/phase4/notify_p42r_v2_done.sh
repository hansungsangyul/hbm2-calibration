#!/bin/bash
# P4.2r v2 완료 감지 → 텔레그램 알림
LOG=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_v2_launch.log
RUNS_DIR=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_v2_runs
NOTIFY=/home/xiangyue/.local/bin/notify_telegram.sh

echo "[$(date)] p42r_v2 완료 대기 중 (log: $LOG) ..."

until grep -q "All 40 runs done" "$LOG" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] 완료 감지 — 결과 수집 중 ..."

STATS=$(python3 - <<'EOF'
import json, glob, os
from pathlib import Path

runs_dir = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42r_v2_runs')
total = 0; div_c = 0; div_m = 0
f1_c = []; f1_m = []
tid_c = []; tid_m = []

for cond in ('calib', 'manual'):
    for rf in sorted((runs_dir / cond).glob('*/results.json')):
        d = json.load(open(rf))
        total += 1
        div = d.get('diverged', False)
        f1 = d.get('final_f1', 0)
        tid = d.get('test_id', {}).get('mean_f1', 0)
        if cond == 'calib':
            div_c += int(div)
            f1_c.append(f1)
            tid_c.append(tid)
        else:
            div_m += int(div)
            f1_m.append(f1)
            tid_m.append(tid)

def fmt(lst):
    if not lst: return '—'
    return f'{sum(lst)/len(lst):.3f} (n={len(lst)})'

print(f"총 완료: {total}/40")
print(f"발산(사후): calib {div_c}/20, manual {div_m}/20")
print(f"val F1 mean: calib {fmt(f1_c)}, manual {fmt(f1_m)}")
print(f"TEST_ID F1: calib {fmt(tid_c)}, manual {fmt(tid_m)}")
EOF
)

MSG="[P4.2r v2 §4-A 완료]
$STATS

조기 종료 제거 + 41ep 완주 + 사후 판정
seeds: calib 750000~, manual 760000~
로그: phase4/p42r_v2_launch.log"

echo "[$(date)] 텔레그램 전송..."
"$NOTIFY" "$MSG"
echo "[$(date)] 전송 완료"
