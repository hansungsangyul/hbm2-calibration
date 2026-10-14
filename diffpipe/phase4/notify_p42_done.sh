#!/bin/bash
# P4.2 완료 감지 → 텔레그램 알림
# PID 3563273 (launch_p42.sh) 종료 후 tost_result.json 읽어 메시지 전송

RESULT_JSON=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42_runs/tost_result.json
TARGET_PID=3563273
NOTIFY=/home/xiangyue/.local/bin/notify_telegram.sh

echo "[$(date)] P4.2 완료 대기 중 (PID=$TARGET_PID) ..."

# PID가 사라질 때까지 대기
while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $TARGET_PID 종료 감지"

# tost_result.json 대기 (최대 10분, TOST 분석 시간)
for i in $(seq 1 20); do
    [ -f "$RESULT_JSON" ] && break
    sleep 30
done

if [ -f "$RESULT_JSON" ]; then
    # 결과 파싱
    EQUIV=$(python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
print('PASS' if d.get('equivalence') else 'FAIL')
" 2>/dev/null || echo "?")
    CALIB_F1=$(python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
print(f\"{d.get('calib_mean','?'):.4f}\")
" 2>/dev/null || echo "?")
    MANUAL_F1=$(python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
print(f\"{d.get('manual_mean','?'):.4f}\")
" 2>/dev/null || echo "?")

    MSG="[P4.2 TOST 완료]
결과: $EQUIV
D_calib F1: $CALIB_F1
D_manual F1: $MANUAL_F1
→ tost_result.json 참고"
else
    MSG="[P4.2 TOST 완료]
tost_result.json 없음 — 분석 확인 필요
launch.log 참조: phase4/p42_runs/launch.log"
fi

echo "[$(date)] 텔레그램 전송 중..."
"$NOTIFY" "$MSG"
echo "[$(date)] 전송 완료"
