#!/bin/bash
set -euo pipefail
R=/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2
PY=/home/xiangyue/.conda/envs/unet/bin/python
NW=${NW:-1}

mkdir -p "$R/a14/logs"
exec 9>"$R/a14/launch.lock"
flock -n 9 || { echo "A14 launcher already running"; exit 1; }
trap 'echo FAILED > "$R/a14/STATUS"' ERR

[[ "$NW" =~ ^[1-9][0-9]*$ ]]
cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation

CUDA_VISIBLE_DEVICES=0 "$PY" "$R/a14_selftest_diag.py" > "$R/a14/logs/selftest.log" 2>&1
"$PY" -c "import json; assert json.load(open('$R/a14/selftest_diag.json'))['selftest_pass']"

Q="$R/a14/queue.txt"
: > "$Q"
printf 'dman_r 0\nab1_r 0\n' >> "$Q"
for i in $(seq 0 19); do printf 'c1 %s\nc2 %s\n' "$i" "$i" >> "$Q"; done

worker() {
  while true; do
    J=$(flock "$Q.lock" -c "head -1 '$Q'; sed -i '1d' '$Q'")
    [ -z "$J" ] && break
    read -r condition run_idx <<< "$J"
    if ! CUDA_VISIBLE_DEVICES=0 "$PY" "$R/a14_train.py" --condition "$condition" --run_idx "$run_idx" \
      > "$R/a14/logs/${condition}_run${run_idx}.log" 2>&1; then
      echo "FAILED: $condition $run_idx" >&2
      return 1
    fi
  done
}

echo RUNNING > "$R/a14/STATUS"
pids=()
for w in $(seq 1 "$NW"); do worker & pids+=("$!"); sleep 60; done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if (( failed )); then echo FAILED > "$R/a14/STATUS"; exit 1; fi

CUDA_VISIBLE_DEVICES=0 "$PY" "$R/a14_counts.py" > "$R/a14/logs/counts.log" 2>&1
"$PY" "$R/a14_analysis.py" > "$R/a14/logs/analysis.log" 2>&1
echo DONE > "$R/a14/STATUS"
