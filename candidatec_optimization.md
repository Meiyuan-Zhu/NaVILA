You are an expert research engineering agent. Your task is to implement a clean, minimal memory optimization for NaVILA, based on the following finalized design:

====================================================
GOAL
====================================================

Implement a stage-agnostic memory refinement mechanism for NaVILA.

Final memory layout:
- Initial anchor memory: first 8 observations of the episode, always preserved
- Refined trajectory memory: 7 selected frames from the remaining historical candidate pool
- Current observation: unchanged

This design is inspired by:
- JanusVLN: preserve an initial window as persistent global anchor memory
- DecoVLN: refine the remaining historical memory using a greedy scoring objective over a candidate pool

Important:
- Do NOT implement stage-aware retrieval in this version
- Do NOT use subgoal-conditioned scoring in this version
- Do NOT modify the core NaVILA architecture
- Do NOT introduce new training or finetuning
- This is an inference-time memory manager only

====================================================
HIGH-LEVEL DESIGN
====================================================

1. Initial window
- Always keep the first 8 observed frames of the episode
- These frames are not part of the candidate pool for refined trajectory memory
- They are treated as persistent global anchor memory

2. Historical candidate pool
- All observed frames after the first 8 belong to a candidate pool
- Maintain a maximum pool size of 120 frames
- If the pool exceeds 120, evict the oldest candidate frames using FIFO
- This pool should be updated online during navigation

3. Refined trajectory memory
- At each history refresh step, select 7 frames from the candidate pool
- Use a DecoVLN-style greedy objective:
  score(f) = λ_R * SimSem(f, I)
             - (1 - λ_R) * [ w_V * SimVis(f, M) + w_T * SimTemp(f, M) ]
- Here:
  - I = full instruction text
  - M = set of already selected refined memory frames
  - SimSem = semantic relevance to the full instruction
  - SimVis = maximum visual similarity to already selected memory frames
  - SimTemp = temporal similarity penalty to already selected memory frames

4. Final memory fed to the model
- 8 initial anchor frames
- 7 refined historical frames
- current observation

====================================================
REQUIRED MATHEMATICAL DEFINITIONS
====================================================

Use the following definitions:

1. Semantic relevance
SimSem(f, I) = cosine(e_f, e_I)

Where:
- e_f = image embedding of candidate frame f
- e_I = text embedding of the full instruction I

2. Visual similarity penalty
SimVis(f, M) = max_{m in M} cosine(e_f, e_m)

3. Temporal similarity penalty
SimTemp(f, M) = 1 / ( min_{m in M} |t_f - t_m| + epsilon )

Where:
- t_f = timestep of candidate frame f
- t_m = timestep of selected memory frame m
- epsilon = small constant, e.g. 1e-6

4. Greedy selection
At each iteration:
f* = argmax over candidate_pool \ selected_memory of score(f)

Then add f* into the selected refined memory set M, and continue until:
- 7 frames are selected
- or candidate pool is exhausted

5. If M is empty
When selecting the first refined frame:
- SimVis = 0
- SimTemp = 0
So the first selection is based only on semantic relevance

====================================================
DEFAULT HYPERPARAMETERS
====================================================

Please use these defaults unless repo constraints require small adjustments:

- INITIAL_WINDOW_SIZE = 8
- REFINED_HISTORY_SIZE = 7
- CANDIDATE_POOL_MAX = 120
- lambda_R = 0.65
- w_V = 0.6
- w_T = 0.4
- epsilon = 1e-6

If the candidate pool has <= 7 frames, use all of them directly.

====================================================
FEATURE EXTRACTION
====================================================

Implement the score using embedding features.

Preferred priority:
1. Reuse an existing image embedding source already available in the project if possible
2. Reuse an existing text encoder if available
3. If needed, add a lightweight image-text embedding module, but keep changes minimal

Requirements:
- Image embeddings for candidate frames must be cached and reused when possible
- Instruction text embedding should be computed once per episode and cached
- Do not recompute everything from scratch at every step if not necessary

====================================================
CANDIDATE POOL UPDATE RULE
====================================================

At each new observation after the initial 8 frames:
- append the new frame into the candidate pool
- if candidate pool size > 120:
  - pop the oldest frame from the candidate pool (FIFO)

Candidate pool contents should therefore always represent:
- all non-initial historical frames if total <= 120
- otherwise the most recent 120 non-initial historical frames

This is a sliding FIFO pool, not a global full-history store.

====================================================
WHEN TO RUN REFINEMENT
====================================================

Run the refined memory selection at the same cadence NaVILA already uses for history refresh / memory repacking.

Do NOT run the greedy selection at every single low-level step unless that is already how history refresh works.

If the current implementation refreshes every N steps, hook this refinement into that same refresh point.

====================================================
PROMPT / INPUT PACKING
====================================================

Update the input organization so the three roles are explicit:

1. Initial anchor memory
- the first 8 observations
- persistent global anchor

2. Refined trajectory memory
- the 7 selected frames from the candidate pool

3. Current observation
- unchanged

If the current prompt text or internal labels only distinguish between:
- historical observations
- current observation

Please adapt them so the memory roles are clearer, e.g.:
- initial anchor memory
- refined trajectory memory
- current observation

Do this carefully and minimally so the model still receives compatible input formatting.

====================================================
IMPLEMENTATION REQUIREMENTS
====================================================

Please inspect the repo first and identify:

1. where historical observations are currently stored
2. where history refresh happens
3. where prompt / image packing happens
4. where frame embeddings can be reused
5. where to insert:
   - initial window persistence
   - candidate pool maintenance
   - refined greedy selection

Then implement the minimal patch.

Do NOT redesign the whole project.

====================================================
LOGGING REQUIREMENTS
====================================================

Add clear logs at each history refresh:

- current step
- initial window indices
- candidate pool size
- candidate pool timestep span
- selected refined memory indices
- selected refined memory scores
- per-frame score breakdown:
  - semantic relevance
  - visual penalty
  - temporal penalty
  - total score
- whether candidate pool eviction happened
- which frames were evicted by FIFO

Example style:
[deco_refine][ep=1][step=30]
initial=[0,1,2,3,4,5,6,7]
candidate_pool_size=57
candidate_pool_range=[8..64]
selected_refined=[12,18,27,36,44,53,61]
score_breakdown=[...]

These logs are important for analysis and reporting.

====================================================
FILES / CODE STYLE
====================================================

Please:
- keep code changes localized
- add a dedicated memory refinement module if appropriate
- avoid scattering logic across too many files
- preserve the baseline path so it remains runnable
- gate the new behavior behind explicit config flags

Suggested config flags:
- MEMORY.STRATEGY = "deco_refine_initial_window"
- MEMORY.INITIAL_WINDOW_SIZE = 8
- MEMORY.REFINED_HISTORY_SIZE = 7
- MEMORY.CANDIDATE_POOL_MAX = 120
- MEMORY.LAMBDA_R = 0.65
- MEMORY.W_VIS = 0.6
- MEMORY.W_TEMP = 0.4

====================================================
EVALUATION
====================================================

After implementation, provide:

1. exact run command(s)
2. which config file(s) changed
3. a short explanation of the memory flow
4. one brief comparison against baseline:
   - baseline uniform history
   - new initial-window + refined-history memory

If possible, recommend a small quick-eval subset first before full val-unseen.

====================================================
ACCEPTANCE CRITERIA
====================================================

The implementation is successful if:

- the first 8 frames are always preserved
- all later history frames go into a capped FIFO candidate pool of size 120
- at each history refresh, 7 refined frames are selected greedily using the Deco-style objective
- baseline path still works
- logs clearly show how refined memory is selected
- no stage-aware retrieval is used in this version

====================================================
WHAT NOT TO DO
====================================================

Do NOT:
- use stage tracker in this version
- use subgoal-conditioned scoring in this version
- add loop recovery in this version
- add corrective finetuning
- add external API calls
- redesign the action policy
- modify low-level locomotion
- add unrelated optimizations

Start by inspecting the repository and identifying the exact insertion points. Then propose the minimal patch plan before coding.