import copy
import gc
import json
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.memory_modules import CandidateAMemoryManager
from vlnce_baselines.memory_modules.stage_tracker import StageTracker

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        padding_frames = num_frames - len(frames)
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    else:
        padding_frames = 0

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


def _format_memory_debug(step_id, episode_id, debug_info, include_scores=True):
    strategy_name = debug_info.get("strategy", "candidate_a")
    selected = debug_info.get("selected_indices", [])
    fallback = debug_info.get("fallback")
    query_source = debug_info.get("query_source", "instruction")
    parser_source = debug_info.get("parser_source", "unknown")
    min_frame_gap = debug_info.get("min_frame_gap", 0)
    min_gap_rejected_count = debug_info.get("min_gap_rejected_count", 0)
    min_gap_rejected_examples = debug_info.get("min_gap_rejected_examples", [])
    text_semantic_enabled = bool(debug_info.get("text_semantic_enabled", False))
    stage_aware_routing_enabled = bool(debug_info.get("stage_aware_routing_enabled", False))
    cov_in_score_enabled = bool(debug_info.get("cov_in_score_enabled", False))
    turn_in_score_enabled = bool(debug_info.get("turn_in_score_enabled", False))
    state = debug_info.get("state", {})
    trace = debug_info.get("trace", "")

    parts = [
        f"[{strategy_name}][ep={episode_id}][step={step_id}]",
        f"selected_history_indices={selected}",
    ]
    if fallback is not None:
        parts.append(f"fallback={fallback}")
    parts.append(f"query_source={query_source}")
    parts.append(f"parser_source={parser_source}")
    parts.append(
        "flags={text_semantic:%s,stage_routing:%s,cov_score:%s,turn_score:%s}" % (
            text_semantic_enabled,
            stage_aware_routing_enabled,
            cov_in_score_enabled,
            turn_in_score_enabled,
        )
    )
    parts.append(
        f"min_frame_gap={int(min_frame_gap)} min_gap_rejected_count={int(min_gap_rejected_count)} "
        f"min_gap_rejected_examples={min_gap_rejected_examples}"
    )

    if state:
        parts.append(
            f"stage={state.get('current_subgoal_id', 0)} recent_actions={state.get('recent_actions', [])} "
            f"last_milestone={state.get('last_milestone_text', '') or 'none'}"
        )
    if trace:
        parts.append(f"trace={trace}")

    if include_scores:
        details = debug_info.get("score_details", {})
        detail_chunks = []
        for idx in selected:
            payload = details.get(idx)
            if payload is None:
                continue
            detail_chunks.append(
                "idx={idx}:rel={rel:.3f},nov={nov:.3f},cov={cov:.3f},turn={turn:.3f},total={total:.3f},anchor={anchor}".format(
                    idx=idx,
                    rel=float(payload.get("rel", 0.0)),
                    nov=float(payload.get("nov", 0.0)),
                    cov=float(payload.get("cov", 0.0)),
                    turn=float(payload.get("turn", 0.0)),
                    total=float(payload.get("total", 0.0)),
                    anchor=bool(payload.get("anchor", False)),
                )
            )
        if detail_chunks:
            parts.append("scores={" + " | ".join(detail_chunks) + "}")

    if "initial_indices" in debug_info:
        parts.append(
            "segments={initial:%s,historical:%s} targets={initial:%s,historical:%s}" % (
                debug_info.get("initial_indices", []),
                debug_info.get("historical_indices", []),
                int(debug_info.get("initial_target", 0)),
                int(debug_info.get("historical_target", 0)),
            )
        )
    return " ".join(parts)


def _extract_distance_to_goal(info):
    if not isinstance(info, dict):
        return None
    val = info.get("distance_to_goal")
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _format_stage_tracker_debug(episode_id, step_id, payload, recent_actions):
    return (
        "[StageTracker][ep={ep}][step={step}] prev_stage={prev} current_stage={cur} "
        "confidence={conf:.3f} validity={valid} allowed=[{amin},{amax}] "
        "recent_actions={actions} evidence={ev}"
    ).format(
        ep=episode_id,
        step=step_id,
        prev=int(payload.get("previous_stage_id", 0)),
        cur=int(payload.get("current_stage_id", 0)),
        conf=float(payload.get("confidence", 0.0)),
        valid=str(payload.get("validity", "unknown")),
        amin=int(payload.get("allowed_min", 0)),
        amax=int(payload.get("allowed_max", 0)),
        actions=list(recent_actions or []),
        ev=str(payload.get("evidence", "")),
    )


def _apply_stop_guard(stop_guard_cfg, raw_action, stop_streak, last_distance_to_goal, memory_manager):
    guard_info = {
        "enabled": bool(stop_guard_cfg.ENABLED),
        "mode": str(stop_guard_cfg.MODE),
        "raw_action": int(raw_action),
        "guarded_action": int(raw_action),
        "consecutive_stop_count": int(stop_streak),
        "distance_to_goal": last_distance_to_goal,
        "require_final_stage": bool(stop_guard_cfg.REQUIRE_FINAL_STAGE),
        "at_final_stage": None,
        "blocked": False,
        "block_reasons": [],
    }

    if not bool(stop_guard_cfg.ENABLED) or int(raw_action) != 0:
        return int(raw_action), guard_info

    mode = str(stop_guard_cfg.MODE).lower()
    checks = []

    if mode in ("consecutive", "fusion"):
        min_count = max(1, int(stop_guard_cfg.CONSECUTIVE_THRESHOLD))
        cond = stop_streak >= min_count
        guard_info["min_consecutive_required"] = min_count
        if mode == "consecutive" or bool(stop_guard_cfg.FUSION_REQUIRE_CONSECUTIVE):
            checks.append(("consecutive", cond))

    if mode in ("distance", "fusion"):
        dist_ok = True
        dist_val = last_distance_to_goal
        if dist_val is None:
            checks.append(("distance_missing", True))
        else:
            threshold = float(stop_guard_cfg.DISTANCE_THRESHOLD_M)
            dist_ok = float(dist_val) <= threshold
            guard_info["distance_threshold_m"] = threshold
            guard_info["distance_ok"] = bool(dist_ok)
            if mode == "distance" or bool(stop_guard_cfg.FUSION_REQUIRE_DISTANCE):
                checks.append(("distance", dist_ok))

    if mode in ("trace", "fusion"):
        at_final_stage = True
        if memory_manager is not None:
            at_final_stage = bool(memory_manager.state_tracker.is_final_stage())
        guard_info["at_final_stage"] = at_final_stage
        if mode == "trace" or bool(stop_guard_cfg.FUSION_REQUIRE_FINAL_STAGE):
            if bool(stop_guard_cfg.REQUIRE_FINAL_STAGE):
                checks.append(("final_stage", at_final_stage))

    should_allow_stop = True
    for check_name, check_ok in checks:
        if not check_ok:
            should_allow_stop = False
            guard_info["block_reasons"].append(check_name)

    if not should_allow_stop:
        guard_info["blocked"] = True
        guard_info["guarded_action"] = 1
        return 1, guard_info

    return 0, guard_info


@baseline_registry.register_trainer(name="navila")
class NaVILATrainer(BaseVLNCETrainer):
    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

        super().__init__(config)

    def _make_dirs(self) -> None:
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        # build model
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint_path, model_name)
        model = model.cuda()

        config = self.config.clone()
        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(
            config.RESULTS_DIR, model_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"{split}_{self.num_chunks}-{self.chunk_idx}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}

        past_rgbs = [[] for _ in range(envs.num_envs)]
        rgb_frames = [[] for _ in range(envs.num_envs)]  # this is for visualization, contains text and map

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_path}]" " [Episodes evaluated: {evaluated}/{total}]" " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        assert envs.num_envs == 1

        queue_actions = []
        memory_manager = None
        memory_debug_once = False
        current_episode_id = None
        memory_cfg = config.MEMORY
        stop_guard_cfg = config.INFERENCE.STOP_GUARD
        force_stop_enabled = bool(getattr(config.INFERENCE, "FORCE_STOP_MAX_STEPS_ENABLED", False))
        force_stop_max_steps = max(1, int(getattr(config.INFERENCE, "FORCE_STOP_MAX_STEPS", 30)))
        episode_step_count = 0
        last_episode_id = None
        use_selective_memory = bool(
            memory_cfg.ENABLE
            and str(memory_cfg.STRATEGY) in {"candidate_a", "candidate_b_v1", "candidate_b_v2", "candidate_b_v3"}
        )
        stage_tracker = None
        stop_streak = 0
        last_distance_to_goal = None

        if use_selective_memory:
            memory_manager = CandidateAMemoryManager(memory_cfg)
            logger.info(f"Selective memory manager enabled. strategy={memory_cfg.STRATEGY}")
            if str(memory_cfg.STRATEGY) == "candidate_b_v3" and bool(getattr(memory_cfg.STAGE_TRACKER, "ENABLE", True)):
                stage_tracker = StageTracker(
                    interval=int(getattr(memory_cfg.STAGE_TRACKER, "INTERVAL", 10)),
                    max_stage_delta=int(getattr(memory_cfg.STAGE_TRACKER, "MAX_STAGE_DELTA", 1)),
                    confidence_threshold=float(getattr(memory_cfg.STAGE_TRACKER, "CONFIDENCE_THRESHOLD", 0.0)),
                    max_evidence_chars=int(getattr(memory_cfg.STAGE_TRACKER, "MAX_EVIDENCE_CHARS", 180)),
                )
                logger.info("Stage tracker enabled for candidate_b_v3.")
        else:
            logger.info("Using baseline uniform memory sampler.")

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:

            current_episodes = envs.current_episodes()
            episode_id = current_episodes[0].episode_id

            if episode_id != last_episode_id:
                episode_step_count = 0
                last_episode_id = episode_id

            force_stop_now = bool(force_stop_enabled and episode_step_count >= force_stop_max_steps)

            if force_stop_now:
                if len(queue_actions) > 0:
                    queue_actions = []
                logger.info(
                    f"[ForcedStop][ep={episode_id}] step_count={episode_step_count} reached max={force_stop_max_steps}; sending STOP."
                )
                outputs = envs.step([0])
                stop_streak += 1
                if memory_manager is not None:
                    memory_manager.update_after_action(action_id=0, turn_deg=0.0)

            elif len(queue_actions) > 0:
                print(f"using queue...{queue_actions[0]}")
                queued_action = queue_actions[0]
                outputs = envs.step([queued_action])
                if memory_manager is not None:
                    turn_deg = 15.0 if queued_action in (2, 3) else 0.0
                    memory_manager.update_after_action(action_id=queued_action, turn_deg=turn_deg)
                queue_actions.pop(0)
                print(f"queue length after using...{len(queue_actions)}")

            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")

                    num_video_frames = model.config.num_video_frames
                    num_video_frames_override = int(getattr(memory_cfg, "NUM_VIDEO_FRAMES_OVERRIDE", -1))
                    if num_video_frames_override > 1:
                        num_video_frames = num_video_frames_override
                    instruction = current_episodes[0].instruction.instruction_text
                    memory_debug = None

                    if memory_manager is not None:
                        if current_episode_id != episode_id:
                            memory_manager.reset_episode(episode_id=episode_id, instruction=instruction)
                            if stage_tracker is not None:
                                stage_tracker.reset(memory_manager.state_tracker.subgoals)
                            subgoals = list(getattr(memory_manager.state_tracker, "subgoals", []) or [])
                            if len(subgoals) > 0:
                                subgoal_text = " | ".join(
                                    [f"{idx + 1}.{sg}" for idx, sg in enumerate(subgoals)]
                                )
                            else:
                                subgoal_text = "none"
                            logger.info(
                                f"[EpisodeSubgoals][ep={episode_id}] source={memory_manager.parser_source} subgoals={subgoal_text}"
                            )
                            current_episode_id = episode_id
                        try:
                            past_and_current_rgbs, memory_debug = memory_manager.select_frames(
                                history_frames=past_rgbs[0],
                                current_frame=curr_rgb,
                                instruction=instruction,
                                num_frames=num_video_frames,
                            )

                            if stage_tracker is not None and stage_tracker.should_infer(memory_manager.step_id):
                                previous_stage_id = int(memory_manager.state_tracker.current_subgoal_id)
                                stage_question = stage_tracker.build_prompt(
                                    previous_stage_id=previous_stage_id,
                                    recent_actions=list(memory_manager.state_tracker.recent_actions),
                                )
                                stage_conv = conv_templates["llama_3"].copy()
                                stage_conv.append_message(
                                    stage_conv.roles[0],
                                    "Current observation <image>\n" + stage_question,
                                )
                                stage_conv.append_message(stage_conv.roles[1], None)
                                stage_prompt = stage_conv.get_prompt()

                                stage_images_tensor = process_images([curr_rgb], image_processor, model.config).to(
                                    model.device, dtype=torch.float16
                                )
                                stage_input_ids = (
                                    tokenizer_image_token(
                                        stage_prompt,
                                        tokenizer,
                                        IMAGE_TOKEN_INDEX,
                                        return_tensors="pt",
                                    )
                                    .unsqueeze(0)
                                    .cuda()
                                )
                                stage_stop_str = (
                                    stage_conv.sep if stage_conv.sep_style != SeparatorStyle.TWO else stage_conv.sep2
                                )
                                stage_stopping_criteria = KeywordsStoppingCriteria(
                                    [stage_stop_str], tokenizer, stage_input_ids
                                )
                                with torch.inference_mode():
                                    stage_output_ids = model.generate(
                                        stage_input_ids,
                                        images=stage_images_tensor.half().cuda(),
                                        do_sample=False,
                                        temperature=0.0,
                                        max_new_tokens=96,
                                        use_cache=True,
                                        stopping_criteria=[stage_stopping_criteria],
                                        pad_token_id=tokenizer.eos_token_id,
                                    )
                                stage_outputs = tokenizer.batch_decode(
                                    stage_output_ids, skip_special_tokens=True
                                )[0].strip()
                                if stage_outputs.endswith(stage_stop_str):
                                    stage_outputs = stage_outputs[: -len(stage_stop_str)]
                                stage_payload = stage_tracker.parse_response(
                                    text=stage_outputs.strip(),
                                    previous_stage_id=previous_stage_id,
                                )
                                memory_manager.state_tracker.set_stage(
                                    stage_id=int(stage_payload["current_stage_id"]),
                                    confidence=float(stage_payload["confidence"]),
                                    evidence=str(stage_payload["evidence"]),
                                )
                                logger.info(
                                    _format_stage_tracker_debug(
                                        episode_id=episode_id,
                                        step_id=memory_manager.step_id,
                                        payload=stage_payload,
                                        recent_actions=list(memory_manager.state_tracker.recent_actions),
                                    )
                                )

                            should_log = bool(memory_cfg.LOG_SELECTED_FRAMES) and (
                                memory_manager.step_id == 1
                                or (memory_manager.step_id % max(1, int(memory_cfg.LOG_INTERVAL)) == 0)
                            )
                            if should_log:
                                logger.info(
                                    _format_memory_debug(
                                        memory_manager.step_id,
                                        episode_id,
                                        memory_debug,
                                        include_scores=bool(memory_cfg.LOG_SCORES),
                                    )
                                )
                        except Exception as memory_error:
                            if not memory_debug_once:
                                logger.warning(f"Candidate A selector failed, falling back to uniform sampler: {memory_error}")
                                memory_debug_once = True
                            past_and_current_rgbs = sample_and_pad_images(
                                past_rgbs[0] + [curr_rgb], num_frames=num_video_frames
                            )
                    else:
                        past_and_current_rgbs = sample_and_pad_images(
                            past_rgbs[0] + [curr_rgb], num_frames=num_video_frames
                        )


                    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

                    frame_length = len(past_and_current_rgbs)
                    print(f"input frame length {frame_length}")

                    strategy_name = str(getattr(memory_cfg, "STRATEGY", ""))
                    if strategy_name in {"candidate_b_v2", "candidate_b_v3"} and memory_debug is not None:
                        history_frame_count = max(0, len(past_and_current_rgbs) - 1)
                        initial_count = int(memory_debug.get("initial_used", min(8, history_frame_count)))
                        initial_count = max(0, min(initial_count, history_frame_count))
                        historical_count = max(0, history_frame_count - initial_count)
                        initial_images = "<image>\n" * initial_count
                        historical_images = "<image>\n" * historical_count
                        question = (
                            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                            f"of initial frames at the beginning of the episode {initial_images}, video of historical observations {historical_images}, and current observation <image>\n. "
                            f'Your assigned task is: "{instruction}" '
                            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                            f"degree, moving forward a certain distance, or stop if the task is completed."
                        )
                    else:
                        question = (
                            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                            f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                            f"degree, moving forward a certain distance, or stop if the task is completed."
                        )
                    if (
                        memory_manager is not None
                        and str(getattr(memory_cfg, "STRATEGY", "")) != "candidate_b_v1"
                        and bool(getattr(memory_cfg, "ENABLE_TRACE_IN_PROMPT", False))
                    ):
                        trace_text = memory_manager.state_tracker.get_trace_text()
                        if trace_text:
                            question += f" Current navigation trace: {trace_text}."

                    conv_mode = "llama_3"
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
                        model.device, dtype=torch.float16
                    )
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_tensor.half().cuda(),
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=32,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()

                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
                    outputs = outputs.strip()
                    print(outputs)
                    logger.info(f"[ActionText][ep={episode_id}] {outputs}")

                    # Define the regex patterns for each action
                    patterns = {
                        0: re.compile(r"\bstop\b", re.IGNORECASE),
                        1: re.compile(r"\bis move forward\b", re.IGNORECASE),
                        2: re.compile(r"\bis turn left\b", re.IGNORECASE),
                        3: re.compile(r"\bis turn right\b", re.IGNORECASE),
                    }

                    # Function to map a string to an action integer
                    def map_string_to_action(s):
                        for action, pattern in patterns.items():
                            if pattern.search(s):
                                return action
                        return None  # Return None if no match is found

                    try:
                        actions = [map_string_to_action(outputs)]
                    except:
                        actions = [1]

                    raw_action = 1 if actions[0] is None else int(actions[0])
                    if raw_action == 0:
                        stop_streak += 1
                    else:
                        stop_streak = 0

                    guarded_action, guard_info = _apply_stop_guard(
                        stop_guard_cfg=stop_guard_cfg,
                        raw_action=raw_action,
                        stop_streak=stop_streak,
                        last_distance_to_goal=last_distance_to_goal,
                        memory_manager=memory_manager,
                    )
                    actions = [guarded_action]
                    print(actions)
                    logger.info(f"[ActionID][ep={episode_id}] {actions}")
                    if bool(stop_guard_cfg.ENABLED):
                        logger.info(
                            "[StopGuard][ep={ep}] raw={raw} guarded={guarded} blocked={blocked} reasons={reasons} "
                            "distance={dist} stage={stage} consecutive_stop_count={count}".format(
                                ep=episode_id,
                                raw=guard_info.get("raw_action"),
                                guarded=guard_info.get("guarded_action"),
                                blocked=guard_info.get("blocked"),
                                reasons=guard_info.get("block_reasons", []),
                                dist=guard_info.get("distance_to_goal"),
                                stage=(
                                    memory_manager.state_tracker.current_subgoal_id
                                    if memory_manager is not None
                                    else -1
                                ),
                                count=guard_info.get("consecutive_stop_count"),
                            )
                        )

                if actions[0] == 1:
                    try:
                        match = re.search(r"move forward (\d+) cm", outputs)
                        distance = int(match.group(1))
                    except:
                        distance = 25
                    if (distance % 25) != 0:
                        distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                    outputs = envs.step([1])
                    if memory_manager is not None:
                        memory_manager.update_after_action(action_id=1, turn_deg=0.0)

                    for _ in range(int(distance // 25) - 1):
                        queue_actions.append(1)

                elif actions[0] == 2:
                    try:
                        match = re.search(r"turn left (\d+) degree", outputs)
                        degree = int(match.group(1))
                    except:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([2])
                    if memory_manager is not None:
                        memory_manager.update_after_action(action_id=2, turn_deg=15.0)

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(2)
                    print(f"queue length: {len(queue_actions)}")

                elif actions[0] == 3:
                    try:
                        match = re.search(r"turn right (\d+) degree", outputs)
                        degree = int(match.group(1))
                    except:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([3])
                    if memory_manager is not None:
                        memory_manager.update_after_action(action_id=3, turn_deg=15.0)

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(3)

                else:  # 0, stop
                    outputs = envs.step(actions)
                    if memory_manager is not None:
                        memory_manager.update_after_action(action_id=0, turn_deg=0.0)

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            episode_step_count += 1
            last_distance_to_goal = _extract_distance_to_goal(infos[0]) if len(infos) > 0 else None

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))

                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i] = []
                episode_step_count = 0
                last_episode_id = None
                if memory_manager is not None and i == 0:
                    current_episode_id = None

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (envs, batch, rgb_frames,) = self._pause_envs(
                envs_to_pause,
                envs,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            batch,
            rgb_frames,
        )

    def eval(self) -> None:
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu")
        )
        if "tensorboard" in self.config.VIDEO_OPTION:
            assert len(self.config.TENSORBOARD_DIR) > 0, "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert len(self.config.VIDEO_DIR) > 0, "Must specify a directory for storing videos on disk"

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            if os.path.isdir(self.config.EVAL_CKPT_PATH_DIR):
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                )
