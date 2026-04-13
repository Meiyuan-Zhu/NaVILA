The NaVILA repository is already available locally. Please inspect it first before proposing changes.

You are an expert research engineering agent. Your task is to stabilize and simplify my current Candidate A memory prototype for NaVILA.

This is NOT a request to build a larger or more complex system.
This is a request to DEBUG and REFACTOR the current Candidate A into a smaller, more stable, inference-time-only MVP.

====================================================
PROJECT CONTEXT
====================================================

We are working on stage 2 of a "Memory Optimization for VLN" research task.
The goal is to improve NaVILA's memory mechanism during inference, under tight time constraints.
The intended design originally came from a "Candidate A" idea:
- selective memory instead of uniform sampling
- optional subgoal-aware routing
- optional trajectory trace
- fixed memory budget

However, the current implementation has become unstable.

Observed failure mode:
- severe oscillation: repeated turn-left / turn-right behavior
- dense adjacent history-frame selection
- memory appears biased toward recent turning frames
- subgoal/stage and trace may be amplifying local misalignment instead of helping
- stop behavior is not yet the main target; first priority is removing oscillation and making memory selection sane

Important: this is a one-day debugging-oriented research prototype.
Favor simplicity, control, and clean ablations over sophistication.

====================================================
CURRENT ISSUES TO ASSUME AS TRUE
====================================================

Please assume the following problems are likely real and should be addressed first:

1. The current scorer implementation is unstable:
   - total_score currently uses only relevance + novelty + coverage
   - turn_bonus exists in code but may not actually affect the final score
   - relevance mixes visual similarity, recency, and text semantic similarity
   - text semantic relevance may inject turn/left/right/junction biases
   - coverage is only a soft term and does not prevent adjacent-frame collapse

2. The current subgoal/stage/trace path is too strong:
   - stage-aware routing may be pushing the model into a later stage too early
   - trace content such as repeated recent turn actions may be reinforcing turning loops
   - subgoal-aware logic should be treated as optional, not core, during stabilization

3. The current memory bank selects too many temporally adjacent frames:
   - we need hard temporal deduplication / minimum frame gap
   - soft coverage alone is not sufficient

4. The first priority is to produce a stable "Candidate A-lite" baseline:
   - fixed budget
   - first-frame anchor
   - score-based historical frame selection
   - no retraining
   - no LoRA
   - no full graph memory
   - no experience memory
   - no heavy architectural rewrite

====================================================
HIGH-LEVEL GOAL
====================================================

Refactor the current Candidate A implementation into a simpler and more stable version called:

Candidate A-lite-v2

Its default behavior should be:

- keep NaVILA architecture unchanged
- keep low-level locomotion unchanged
- keep the current frame pathway unchanged
- replace uniform historical sampling with a stable score-based selector
- use a fixed history budget (default K=8)
- preserve the first frame as an anchor
- use simple visual relevance + novelty as the core scoring rule
- add a hard minimum-frame-gap rule to prevent adjacent-frame collapse
- disable or demote risky components by default:
  - text semantic relevance OFF by default
  - stage-aware routing OFF by default
  - trace-to-prompt OFF by default
  - turn bonus OFF by default
  - coverage OFF by default

Optional components may remain in code behind flags, but they must not drive the default experiment.

====================================================
REQUIRED IMPLEMENTATION PLAN
====================================================

PHASE 1 — REPO INSPECTION
Before editing code, inspect the repository and identify:
1. where historical observation frames are currently sampled
2. where the current Candidate A memory manager is integrated
3. where FrameScorer is instantiated
4. where config values are loaded
5. where prompt/context packing happens
6. where trace is inserted into prompt, if applicable
7. where subgoal/stage influences memory routing, if applicable

Then summarize:
- the relevant files
- the safest insertion points
- what must be changed vs what can be left untouched

Do not rewrite the whole project.

----------------------------------------------------
PHASE 2 — PATCH THE FRAME SCORER
----------------------------------------------------

Modify the scorer to make it numerically sane and easier to interpret.

A. Required changes
1. Make the default score formula:

   Score = w_rel * Rel + w_nov * Nov

2. Set defaults:
   - weight_rel > weight_nov
   - weight_cov = 0 by default
   - weight_turn = 0 by default
   - use_text_semantic_relevance = False by default

3. Make Rel and Nov explicitly bounded to [0, 1].

4. Keep Coverage and TurnBonus available in code as optional features, but not active in the default path.

B. Relevance redesign
Default relevance should be mostly visual.

Preferred default:
- visual_rel = normalized cosine similarity between candidate frame feature and current observation feature, mapped to [0, 1]
- recency may be kept as a very small auxiliary term, or disabled entirely
- text semantic relevance should be disabled by default because it may inject turn/left/right/junction bias

Recommended default:
Rel = 0.9 * visual_rel + 0.1 * recency
Then clamp to [0, 1]

If recency causes instability, simplify further to:
Rel = visual_rel

C. Novelty redesign
Novelty should be:
- based on max similarity to already selected frames
- mapped cleanly to [0, 1]

Recommended:
max_sim = max cosine similarity
map cosine to [0, 1]
Nov = 1 - mapped_max_sim
clamp to [0, 1]

D. Coverage
Keep Coverage implementation in the file if useful, but do not use it in the default total score.
Coverage may remain for later ablations behind a flag.

E. Turn bonus
Keep turn_bonus() in the file for optional future use, but do not use it in the default total score.
If total_score currently ignores turn_bonus, document that bug/fix explicitly.

----------------------------------------------------
PHASE 3 — ADD HARD TEMPORAL DEDUPLICATION
----------------------------------------------------

This is required.

Implement a hard minimum-frame-gap rule in the memory selection logic.

Rule:
- aside from the first-frame anchor, two selected history frames must not be too close in time
- default minimum gap: 4 or 5 steps
- if a candidate frame is too close to an already selected frame, skip it even if its score is high

This rule is more important than coverage for the default stabilization experiment.

----------------------------------------------------
PHASE 4 — REMOVE HIGH-RISK CONTROL LOOPS FROM THE DEFAULT PATH
----------------------------------------------------

The following should be disabled by default in the main experiment:

1. Stage-aware memory routing
   - subgoal/stage may still be computed and logged
   - but do not let current_subgoal_id drive memory bucket assignment in the default path

2. Trace-to-prompt
   - trace may still be maintained for logging
   - but do not feed trace into the VLA prompt by default

3. Text semantic relevance
   - keep behind a config flag
   - OFF by default

4. Coverage in score
   - OFF by default

5. Turn bonus in score
   - OFF by default

This means the default A-lite-v2 experiment should be:
- first-frame anchor
- top-scoring history frames
- score based mainly on visual relevance + novelty
- hard min-frame-gap

----------------------------------------------------
PHASE 5 — KEEP ONLY LIGHTWEIGHT OPTIONAL FEATURES
----------------------------------------------------

Optional features may remain behind config flags:
- subgoal parser
- state tracker
- trace
- coverage
- text semantic relevance
- turn bonus

But these should not be required for the default experiment.

If these optional features are already implemented, do not delete them unless necessary.
Instead:
- make them configurable
- make the simpler path the default
- ensure they cannot silently affect the default run

----------------------------------------------------
PHASE 6 — CONFIG CLEANUP
----------------------------------------------------

Create or update config flags so the default experiment is explicit.

Required flags (or equivalent):
- enable_candidate_a_lite_v2 = True
- memory_budget_k = 8
- weight_rel = 0.75 (or similar)
- weight_nov = 0.25 (or similar)
- weight_cov = 0.0
- weight_turn = 0.0
- use_text_semantic_relevance = False
- enable_trace_in_prompt = False
- enable_stage_aware_routing = False
- min_frame_gap = 4
- enable_cov_in_score = False
- enable_turn_in_score = False

You may choose slightly different default numbers if justified, but keep the defaults simple and conservative.

----------------------------------------------------
PHASE 7 — LOGGING
----------------------------------------------------

Add clean logs for debugging and research reporting.

Required logs:
- selected history indices
- candidate scores for chosen frames
- score breakdown (rel, nov, optionally cov/turn if enabled)
- whether a candidate was rejected due to min_frame_gap
- whether text semantic relevance was enabled
- whether trace was injected into prompt
- whether stage-aware routing was enabled

Optional logs:
- stage and trace values, if they still exist

The goal is to make it very obvious what changed between:
- original baseline
- old unstable Candidate A
- new Candidate A-lite-v2

----------------------------------------------------
PHASE 8 — EVALUATION PLAN
----------------------------------------------------

Provide commands and instructions to run these experiments:

1. Baseline
   - original NaVILA uniform history sampling

2. Old Candidate A (if still runnable)
   - current unstable version for comparison

3. Candidate A-lite-v2 (default stabilized version)
   - visual relevance + novelty
   - no text semantic relevance
   - no stage-aware routing
   - no trace in prompt
   - no coverage
   - no turn bonus
   - min_frame_gap enabled

4. Optional small ablations
   - + coverage
   - + text semantic relevance
   - + trace in prompt
   - + stage-aware routing

The agent should recommend running the stabilized default first, then adding one risky term back at a time.

====================================================
DELIVERABLES
====================================================

You must provide:

1. Code changes
2. A short explanation of what was changed and why
3. Exact run commands
4. A recommended experiment order
5. A short note on expected behavior improvements:
   - fewer adjacent selected frames
   - less turn-left / turn-right oscillation
   - more stable progress through trajectories
6. A list of optional features that were intentionally disabled by default
7. A short note on what remains for future work:
   - trace reintroduction
   - stage-aware routing reintroduction
   - stop-check refinement
   - scorer term ablations
   - more principled text embeddings

====================================================
ACCEPTANCE CRITERIA
====================================================

The implementation is successful if:

- baseline still runs
- a simplified Candidate A-lite-v2 path runs without retraining
- default score no longer depends on text semantic relevance, coverage, or turn bonus
- selected history frames are no longer heavily clustered in adjacent indices
- the system is easier to debug
- logs clearly show score components and rejections due to min_frame_gap
- the default experimental path is smaller, safer, and more stable than the current Candidate A

====================================================
WHAT NOT TO DO
====================================================

Do NOT:
- redesign the whole project
- add graph memory
- add experience memory
- add retrieval across episodes
- add LoRA / retraining
- add a heavy async architecture
- let trace remain in the default prompt path
- let stage-aware routing remain in the default path
- keep unsafe text semantic relevance enabled by default
- spend time polishing nonessential abstractions

====================================================
DEBUGGING PRIORITY ORDER
====================================================

If something is unclear, follow this order:

1. stabilize the scorer
2. add min_frame_gap
3. ensure baseline path still works
4. ensure simplified A-lite-v2 path works
5. add logs
6. only then revisit optional features one-by-one

Start by inspecting the repository and identifying the current history sampling path, current Candidate A integration path, and current scorer integration path. Then propose the minimal patch plan before writing code.