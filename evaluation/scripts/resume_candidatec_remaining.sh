#!/usr/bin/env bash
set -euo pipefail

source /home/lmgroup-intern/tools/miniconda3/etc/profile.d/conda.sh
conda activate navila-eval

cd /home/lmgroup-intern/workspace/NaVILA/evaluation

# Headless Habitat-Sim uses EGL; keep this consistent with the official eval script.
unset DISPLAY
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-$HOME/nvidia-egl.json}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export EGL_LOG_LEVEL=${EGL_LOG_LEVEL:-error}
export PYTHONFAULTHANDLER=1

GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2n | head -n1 | cut -d, -f1 | tr -d ' ')
if [[ -z "${GPU_ID}" ]]; then
  GPU_ID=0
fi

EPISODES_ALLOWED=$(python - <<'PY'
import json
from pathlib import Path
all_ids=[46,47,48,49,50,51,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
p=Path('eval_out_quick/candidatec_100_extra50_noforcestop_video/results/navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json')
d=json.loads(p.read_text()) if p.exists() else {}
done={int(k) for k in d.keys()}
remaining=[x for x in all_ids if x not in done]
print('[' + ','.join("'{}'".format(x) for x in remaining) + ']')
PY
)

EPISODE_COUNT=$(python - <<'PY'
import json
from pathlib import Path
all_ids=[46,47,48,49,50,51,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
p=Path('eval_out_quick/candidatec_100_extra50_noforcestop_video/results/navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json')
d=json.loads(p.read_text()) if p.exists() else {}
done={int(k) for k in d.keys()}
remaining=[x for x in all_ids if x not in done]
print(len(remaining))
PY
)

if [[ "$EPISODE_COUNT" -le 0 ]]; then
  echo "No remaining episodes. Nothing to run."
  exit 0
fi

echo "Resuming candidate_c with EPISODE_COUNT=$EPISODE_COUNT"
echo "EPISODES_ALLOWED=$EPISODES_ALLOWED"
echo "Using GPU_ID=$GPU_ID"

python run.py --run-type eval --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
  TORCH_GPU_ID "$GPU_ID" \
  SIMULATOR_GPU_IDS "[$GPU_ID]" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID "$GPU_ID" \
  EVAL_CKPT_PATH_DIR /home/lmgroup-intern/workspace/models/navila-llama3-8b-8f \
  EVAL.SPLIT val_unseen \
  EVAL.EPISODE_COUNT "$EPISODE_COUNT" \
  EVAL.SAVE_RESULTS True \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED "$EPISODES_ALLOWED" \
  RESULTS_DIR eval_out_quick/candidatec_100_extra50_noforcestop_video/results \
  VIDEO_DIR eval_out_quick/candidatec_100_extra50_noforcestop_video/videos \
  LOG_FILE eval_out_quick/candidatec_100_extra50_noforcestop_video/run.log \
  VIDEO_OPTION "['disk']" \
  MEMORY.ENABLE True \
  MEMORY.STRATEGY candidate_b_v3 \
  MEMORY.NUM_VIDEO_FRAMES_OVERRIDE 16 \
  MEMORY.DECO_REFINE.ENABLE True \
  MEMORY.DECO_REFINE.INITIAL_WINDOW_SIZE 8 \
  MEMORY.DECO_REFINE.REFINED_HISTORY_SIZE 7 \
  MEMORY.DECO_REFINE.CANDIDATE_POOL_MAX 120 \
  MEMORY.USE_SUBGOAL_PARSER True \
  MEMORY.SUBGOAL_PARSER.USE_LLM True \
  MEMORY.SUBGOAL_PARSER.BACKEND openai_compatible \
  MEMORY.SUBGOAL_PARSER.MODEL qwen-flash \
  MEMORY.SUBGOAL_PARSER.API_BASE_URL https://dashscope.aliyuncs.com/compatible-mode/v1 \
  MEMORY.SUBGOAL_PARSER.API_KEY_ENV DASHSCOPE_API_KEY \
  MEMORY.SUBGOAL_PARSER.TIMEOUT_SECONDS 12 \
  MEMORY.SUBGOAL_PARSER.MAX_SUBGOALS 8 \
  MEMORY.SUBGOAL_PARSER.FALLBACK_TO_RULE True \
  MEMORY.SUBGOAL_PARSER.REQUIRE_LLM_SUCCESS False \
  MEMORY.SUBGOAL_CACHE_DIR data/subgoal_cache_candidatec_100_extra50_noforcestop_video \
  MEMORY.STAGE_TRACKER.ENABLE True \
  MEMORY.STAGE_TRACKER.INTERVAL 10 \
  MEMORY.STAGE_TRACKER.CONFIDENCE_THRESHOLD 0.0 \
  MEMORY.USE_TRAJECTORY_TRACE True \
  MEMORY.ENABLE_TRACE_IN_PROMPT True \
  MEMORY.LOOP_RECOVERY.ENABLE True \
  INFERENCE.FORCE_STOP_MAX_STEPS_ENABLED False
