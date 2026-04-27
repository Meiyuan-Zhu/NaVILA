#!/usr/bin/env bash
set -u

cd /home/lmgroup-intern/workspace/NaVILA/evaluation || exit 1

RUN_LOG="eval_out_quick/candidatec_100_extra50_noforcestop_video/run.log"
RES_JSON="eval_out_quick/candidatec_100_extra50_noforcestop_video/results/navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json"
PYTHON_BIN="$(command -v python3 || command -v python || true)"

echo "== Process =="
pgrep -af "scripts/resume_candidatec_remaining.sh|python run.py --run-type eval --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml" || echo "No matching process"

echo
pid=$(pgrep -f "python run.py --run-type eval --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml" | head -n1 || true)
if [[ -n "$pid" ]]; then
  echo "== /proc/$pid =="
  grep -E "State|VmRSS|Threads|voluntary_ctxt_switches|nonvoluntary_ctxt_switches" /proc/$pid/status || true
fi

echo
if [[ -f "$RUN_LOG" ]]; then
  echo "== run.log tail =="
  tail -n 12 "$RUN_LOG"
else
  echo "run.log not found"
fi

echo
if [[ -f "$RES_JSON" ]]; then
  echo "== JSON entries =="
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "python/python3 not found in current shell"
    exit 0
  fi
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p=Path('eval_out_quick/candidatec_100_extra50_noforcestop_video/results/navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json')
d=json.loads(p.read_text())
keys=sorted(d.keys(), key=lambda x:int(x))
print('entries:', len(d))
print('last_ids:', keys[-10:])
PY
else
  echo "result json not found"
fi
