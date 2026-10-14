#!/bin/bash
# start_v77.sh — v77_chain tmux 세션 시작
# Usage: bash start_v77.sh
# 확인:  tmux attach -t v77_chain

SESSION=v77_chain
LAUNCH=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/launch_v77.sh

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[WARN] tmux 세션 '$SESSION' 이미 존재합니다."
    echo "  확인: tmux attach -t $SESSION"
    echo "  재시작하려면: tmux kill-session -t $SESSION && bash start_v77.sh"
    exit 1
fi

tmux new-session -d -s "$SESSION" "bash $LAUNCH"
echo "v77_chain tmux 세션 시작됨."
echo "  확인:   tmux attach -t $SESSION"
echo "  로그:   tail -f /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77/launch_v77.log"
