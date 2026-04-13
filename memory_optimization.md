The NaVILA repository is already available locally. Please inspect it first before proposing changes.

You are an expert research engineering agent. Your task is to modify an existing NaVILA codebase to implement a practical **training-free MVP** of a new memory mechanism called:

**Candidate A: Subgoal-Budgeted Selective Memory with Trajectory Trace**

The goal is to improve NaVILA’s memory construction during inference by replacing its current **uniform historical frame sampling** with a **goal-aware keyframe selection mechanism** under a fixed memory budget.

This is a **stage-2 research prototype**, not a production refactor. Prioritize:
1. correctness
2. minimal changes
3. fast implementation
4. clear logging
5. easy ablation against the original baseline

Do **not** implement experience memory, persistent memory, retrieval across episodes, or full graph memory in this task.

When there is any conflict between the **full Candidate A design** and the **one-day MVP**, always choose the simpler MVP.

# 1. High-level goal

Replace uniform history sampling with an online, goal-aware selector. Optionally add a compact explicit trajectory trace that tracks instruction progress and prevents drift.

The main idea is:
- Keep the overall NaVILA architecture unchanged.
- Keep the high-level VLA/VLM inference pipeline unchanged as much as possible.
- Keep the low-level locomotion controller unchanged.
- Only modify how historical memory frames are selected and packed into the prompt/context.

Instead of the original uniform history sampler, implement a goal-aware heuristic memory manager that chooses the most useful historical frames based on:
- relevance to the current instruction/subgoal
- novelty/diversity relative to already selected frames
- decision-point importance (especially turning/junction-like moments)

**Important:** a short trajectory trace is optional in the MVP. If needed for time, implement the frame selector first and leave trace behind a config flag.


# 2. Important constraints

## 2.1 Hard constraints
- No finetuning
- No LoRA
- No retraining
- No cross-episode retrieval
- No heavy architectural rewrite
- No breaking changes to NaVILA’s original action output format
- No changes to the low-level locomotion controller
- Keep memory budget fixed and small (default: K = 8 historical slots)

## 2.2 Practical constraints
- The implementation must be inference-time only
- The result should be easy to compare against the original NaVILA baseline (R2R)
- The code must be clean enough for a research demo / evaluation
- If the repo structure differs from assumptions below, adapt carefully while preserving the intended algorithm


# 3. Key modules

## 3.1 Subgoal Parser 
- One-shot per instruction/episode
- Decompose instruction into ordered sub-instructions
- Can be LLM/API-based, but must run only once
- If an API-based parser is used, cache results locally
- If API is unavailable, fall back to a rule-based splitter

## 3.2 State Tracker
- Online, lightweight, rule-based
- Maintains rough progress state, e.g.:
  - `current_subgoal_id`
  - `completed_ids`
  - `last_milestone_text`
- Should use VLA output, recent actions, and optional trace if available
- Must not depend on expensive online API calls

## 3.3 Frame Scorer
Score candidate historical frames using:

### Required score terms
- **Relevance**: similarity between frame embedding and current instruction/subgoal embedding
- **Novelty**: penalize similarity to already selected frames
- **Turn/Junction bonus**: reward frames near large yaw changes or likely decision points

### Optional score terms
- **Coverage**: reward temporal spread
- **Uncertainty-based weighting**: optional advanced behavior, not required for MVP

## 3.4 Keyframe Bank
- Fixed-size memory bank (default K=8)
- Slot 0 must always preserve the first frame as an anchor
- Remaining slots should be selected by score, not uniform sampling

### MVP bank behavior
If subgoal buckets are too complicated, use:
- first frame always keep
- current frame handled as usual
- remaining historical slots filled by top-scoring frames

This is acceptable for the MVP.

## 4.5 Trajectory Trace
- Short text field updated every VLA call
- Example:
  "stage=2; recent_actions=[forward,forward,left]; last_cue=junction"
- Keep it deterministic, short, easy to serialize, and easy to disable
- Do not generate long natural-language summaries inside the loop


# 5. Data structures

Use simple structures that match the implemented MVP.

Suggested structure:

- `KeyframeSlot = {t, subgoal_id?, pose_hint?, img_tokens?, img_embed, score_rel, score_nov, score_turn?, score_cov?, total_score}`
- `Trace = {completed_subgoals[], current_subgoal, last_k_actions, last_seen_landmarks}`

If subgoal buckets are implemented, use:
- `Bucket[subgoal_id] = min-heap(KeyframeSlot, by total_score)`

If not, a single ranked bank is acceptable for the MVP.

---

# 6. Required algorithmic behavior

## 6.1 Subgoal parsing
Implement a small module that converts a full navigation instruction into a list of ordered subgoals.

Example:
Instruction:
"Walk out of the bedroom, turn left into the hallway, go past the sofa, and stop by the table."

Desired output:
[
  "walk out of the bedroom",
  "turn left into the hallway",
  "go past the sofa",
  "stop by the table"
]

Implementation guidance:
- Preferred: deterministic LLM/API call with temperature 0 and JSON output
- Must support local caching to disk
- Must support fallback rule-based splitting if API is unavailable
- Run only once per episode, never per step

## 6.2 State tracking
Implement a lightweight online state tracker that maintains something like:
- current_subgoal_id
- completed_subgoal_ids
- last_milestone_text
- recent_actions

A heuristic tracker is acceptable.

Suggested heuristics:
- advance stage when a large turn occurs
- advance stage when a milestone-like cue appears
- advance stage when action pattern changes significantly
- optionally use stop tendency to infer final stage

Do not create a heavy dependency here.

## 6.3 Frame scoring (required)
Each candidate historical frame should receive a score.

### Required MVP formula
Score(i) = w_rel * Rel(i) + w_nov * Nov(i) + w_turn * Turn(i)

### Optional extended formula
Score(i) = w_rel * Rel(i) + w_nov * Nov(i) + w_turn * Turn(i) + w_cov * Cov(i)

Where:

- `Rel(i)`: relevance
  - similarity between frame embedding and current instruction/subgoal embedding

- `Nov(i)`: novelty
  - penalize frames too similar to already selected memory frames
  - suggested:
    `Nov(i) = 1 - max cosine similarity to selected embeddings`

- `Turn(i)`: turn/junction importance
  - add a bonus if this frame is near a large rotation or likely decision point
  - suggested heuristic:
    if `abs(delta_yaw) > threshold`, add turn bonus
  - default threshold: 20 degrees

- `Cov(i)` (optional): temporal coverage
  - reward temporal spread within selected memory

## 6.4 Keyframe bank (required)
Recommended default layout under K=8:
- 1 anchor frame (first frame, always kept)
- 1 recent frame
- 4 score-based selected frames
- 1 previous-stage support frame (optional if stage tracker exists)
- 1 global-diversity frame

If this is too complicated, simplify to:
- first frame always keep
- current frame separate as usual
- remaining historical slots filled by top-scoring frames

This is the default acceptable MVP.

## 6.5 Memory read policy (required)
At each high-level VLA decision step:
- pack current observation as usual
- replace original uniformly sampled history with selected keyframes from the bank
- optionally append a short trajectory trace string

---

# 7. Implementation strategy

## 7.1 Phase 1: inspect codebase
First inspect the repository and identify:
1. where NaVILA loads/constructs historical observation frames
2. where uniform sampling of history happens
3. where current observation and historical observations are packed into model input
4. where instruction text is available
5. where inference loop / evaluation loop runs
6. whether frame embeddings can be reused from existing vision features
7. whether yaw / pose / action history is available during inference

Before changing code, summarize:
- which file(s) implement the current history sampler
- which file(s) need modification
- the safest insertion points for Candidate A

## 7.2 Phase 2: implement minimal modules
Create or add modules such as:
- `subgoal_parser.py`
- `state_tracker.py`
- `frame_scorer.py`
- `memory_manager.py`

Adapt names if the repo has a better structure.

## 7.3 Phase 3: integrate with inference
- Replace the original history selection logic with the new memory manager
- Add config flags to enable/disable the feature
- Preserve original baseline behavior

## 7.4 Phase 4: logging and ablation
Add logging so the user can inspect:
- selected frame indices
- score components per selected frame
- total scores
- stage/subgoal id if available
- whether turn bonus was triggered
- optional trace content

This is very important for debugging and research presentation.

---

# 8. Deliverables

You must provide all of the following:

## 8.1 Code changes
- Implement the Candidate A MVP in the repo

## 8.2 Change summary
After implementation, report:
- which files were changed
- what each change does
- what assumptions were made

## 8.3 Run instructions
Provide exact commands to run:
- baseline
- new method
- optional ablation with trace on/off

## 8.4 Evaluation instructions
Explain how to compare:
- original NaVILA baseline
- Candidate A memory manager

## 8.5 Logging examples
Show example logs of:
- selected history frame indices
- score components
- total score
- stage/subgoal state if available
- optional trace

---

# 9. Acceptance criteria

The implementation is successful if:
- the code runs without retraining
- the original NaVILA path still works behind a config flag
- Candidate A path can be enabled cleanly
- historical memory is no longer uniformly sampled when Candidate A is enabled
- selected keyframes are score-based and logged
- the first frame is preserved as an anchor
- the implementation is small enough to be debugged in one day
- the user can run a baseline vs modified comparison easily

Bonus, if feasible:
- optional short trace text
- optional simple subgoal parsing cache
- optional ablation flags

---

# 10. What NOT to do

Do not:
- redesign the entire project
- add experience memory
- add persistent retrieval
- implement a full topological graph memory
- introduce a complex async system
- require full retraining
- hardcode dataset-specific hacks unless clearly isolated and documented
- over-engineer the state tracker
- spend time polishing non-critical abstractions if they block implementation

---

# 11. Debugging priorities

If something is unclear in the codebase, prioritize in this order:
1. get a minimal score-based keyframe selector working
2. preserve baseline behavior behind a switch
3. add logs
4. add subgoal parsing
5. add trace
6. refine heuristics

If time is tight, a valid minimal version is:
- no API parser
- no trace
- no explicit subgoal buckets
- just replace uniform sampling with:
  - first-frame anchor
  - relevance score
  - novelty penalty
  - turn bonus
  - top-K selection

This is still acceptable as a strong MVP.

Start by inspecting the repository and locating the current historical memory sampling logic.