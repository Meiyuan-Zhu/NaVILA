import copy
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from vlnce_baselines.memory_modules.frame_scorer import FrameScorer
from vlnce_baselines.memory_modules.state_tracker import StateTracker
from vlnce_baselines.memory_modules.subgoal_parser import SubgoalParser


class CandidateAMemoryManager:
    """Subgoal-budgeted selective memory manager (training-free MVP)."""

    def __init__(self, memory_cfg):
        self.cfg = memory_cfg
        parser_cfg = getattr(memory_cfg, "SUBGOAL_PARSER", None)
        self.parser = SubgoalParser(
            cache_dir=str(memory_cfg.SUBGOAL_CACHE_DIR),
            enabled=bool(memory_cfg.USE_SUBGOAL_PARSER),
            use_llm=bool(getattr(parser_cfg, "USE_LLM", False)),
            backend=str(getattr(parser_cfg, "BACKEND", "openai_compatible")),
            model=str(getattr(parser_cfg, "MODEL", "qwen-flash")),
            api_base_url=str(getattr(parser_cfg, "API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")),
            api_key_env=str(getattr(parser_cfg, "API_KEY_ENV", "DASHSCOPE_API_KEY")),
            timeout_seconds=int(getattr(parser_cfg, "TIMEOUT_SECONDS", 8)),
            max_subgoals=int(getattr(parser_cfg, "MAX_SUBGOALS", 8)),
            fallback_to_rule=bool(getattr(parser_cfg, "FALLBACK_TO_RULE", True)),
        )
        self.state_tracker = StateTracker(
            max_actions=int(memory_cfg.STATE.MAX_RECENT_ACTIONS),
            turn_threshold_deg=float(memory_cfg.TURN_THRESHOLD_DEGREES),
            use_trace=bool(memory_cfg.USE_TRAJECTORY_TRACE),
        )
        self.scorer = FrameScorer(
            weight_rel=float(memory_cfg.WEIGHTS.RELEVANCE),
            weight_nov=float(memory_cfg.WEIGHTS.NOVELTY),
            weight_turn=float(memory_cfg.WEIGHTS.TURN_BONUS),
            weight_cov=float(getattr(memory_cfg.WEIGHTS, "COVERAGE", 0.0)),
            turn_threshold_deg=float(memory_cfg.TURN_THRESHOLD_DEGREES),
            enable_cov_in_score=bool(getattr(memory_cfg, "ENABLE_COV_IN_SCORE", False)),
            enable_turn_in_score=bool(getattr(memory_cfg, "ENABLE_TURN_IN_SCORE", False)),
            use_text_semantic_relevance=bool(getattr(memory_cfg, "USE_TEXT_SEMANTIC_RELEVANCE", True)),
            text_embed_dim=int(getattr(memory_cfg, "TEXT_EMBED_DIM", 128)),
        )
        self.min_frame_gap = max(0, int(getattr(memory_cfg, "MIN_FRAME_GAP", 0)))
        self.enable_stage_aware_routing = bool(getattr(memory_cfg, "ENABLE_STAGE_AWARE_ROUTING", False))
        if str(getattr(memory_cfg, "STRATEGY", "candidate_a")) == "candidate_b_v1":
            self.strategy_name = "candidate_b_v1"
        else:
            self.strategy_name = (
                "candidate_a_lite_v2" if bool(getattr(memory_cfg, "ENABLE_CANDIDATE_A_LITE_V2", False)) else "candidate_a"
            )

        self.episode_id: Optional[str] = None
        self.step_id = 0
        self.frame_meta: Deque[Dict[str, float]] = deque(maxlen=int(memory_cfg.BUFFER.MAX_HISTORY_FRAMES))
        self.history_feats: List[np.ndarray] = []
        self._cached_hist_len = 0
        self._fallback_warned = False
        self.parser_source = "rule"

    def reset_episode(self, episode_id: str, instruction: str):
        subgoals = self.parser.parse_one_shot(instruction)
        self.parser_source = str(getattr(self.parser, "last_source", "rule"))
        self.state_tracker.reset(subgoals)
        self.episode_id = episode_id
        self.step_id = 0
        self.frame_meta.clear()
        self.history_feats = []
        self._cached_hist_len = 0

    def update_after_action(self, action_id: int, turn_deg: float = 0.0, yaw_delta: Optional[float] = None):
        turn_value = float(yaw_delta) if yaw_delta is not None else float(turn_deg)
        self.state_tracker.update(action_id=action_id, yaw_delta=turn_value)
        self.frame_meta.append(
            {
                "action": int(action_id),
                "turn_deg": abs(float(turn_value)),
                "yaw_delta": float(turn_value),
            }
        )

    @staticmethod
    def _uniform_sample(images: List[Image.Image], num_frames: int, width: int, height: int) -> List[Image.Image]:
        frames = copy.deepcopy(images)

        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

        latest_frame = frames[-1]
        sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
        sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
        return sampled_frames

    def select_frames(
        self,
        history_frames: List[Image.Image],
        current_frame: Image.Image,
        instruction: str,
        num_frames: int,
        width: int = 512,
        height: int = 512,
    ) -> Tuple[List[Image.Image], Dict[str, object]]:
        self.step_id += 1

        if num_frames is None or num_frames <= 1:
            return [current_frame], {"strategy": self.strategy_name, "selected_indices": []}

        if len(history_frames) == 0:
            frames = self._uniform_sample([current_frame], num_frames=num_frames, width=width, height=height)
            return frames, {
                "strategy": self.strategy_name,
                "fallback": "empty_history",
                "selected_indices": [],
                "state": self.state_tracker.as_dict(),
                "trace": self.state_tracker.get_trace_text(),
            }

        if self.strategy_name == "candidate_b_v1":
            return self._select_frames_candidate_b_v1(
                history_frames=history_frames,
                current_frame=current_frame,
                instruction=instruction,
                num_frames=num_frames,
                width=width,
                height=height,
            )

        history = list(history_frames)
        max_hist_frames = int(self.cfg.BUFFER.MAX_HISTORY_FRAMES)
        truncated = False
        if len(history) > max_hist_frames:
            history = history[-max_hist_frames:]
            truncated = True

        target_hist_slots = num_frames - 1
        current_subgoal_text = self.state_tracker.get_current_subgoal_text()
        use_subgoal_for_rel = bool(
            self.enable_stage_aware_routing and getattr(self.cfg, "USE_SUBGOAL_FOR_RELEVANCE", False)
        )
        use_subgoal_query = bool(use_subgoal_for_rel and current_subgoal_text)
        relevance_query = current_subgoal_text if use_subgoal_query else instruction
        query_source = "subgoal" if use_subgoal_query else "instruction"
        query_text_feat = None
        if self.scorer.use_text_semantic_relevance:
            query_text_feat = self.scorer.compute_text_feature(relevance_query)

        current_feat = self.scorer.compute_feature(current_frame)

        if truncated or len(history) < self._cached_hist_len:
            self.history_feats = [self.scorer.compute_feature(img) for img in history]
            self._cached_hist_len = len(history)
        elif len(history) > self._cached_hist_len:
            new_frames = history[self._cached_hist_len :]
            self.history_feats.extend([self.scorer.compute_feature(img) for img in new_frames])
            self._cached_hist_len = len(history)

        hist_feats = self.history_feats

        selected_indices = [0]
        selected_feats = [hist_feats[0]]
        score_details = {
            0: {
                "rel": 1.0,
                "nov": 0.0,
                "cov": 0.0,
                "turn": 0.0,
                "total": 1.0,
                "anchor": True,
            }
        }

        index_pool = list(range(1, len(history)))
        meta_list = list(self.frame_meta)[-len(history) :]
        min_gap_rejected_indices: List[int] = []

        while len(selected_indices) < min(target_hist_slots, len(history)) and len(index_pool) > 0:
            best_idx = None
            best_score = -1e9
            best_payload = None

            for idx in index_pool:
                selected_non_anchor = [s for s in selected_indices if s > 0]
                if self.min_frame_gap > 0 and any(abs(int(idx) - int(s)) < self.min_frame_gap for s in selected_non_anchor):
                    min_gap_rejected_indices.append(int(idx))
                    continue

                frame_meta = meta_list[idx] if idx < len(meta_list) else None
                rel = self.scorer.relevance(
                    hist_feats[idx],
                    current_feat,
                    idx=idx,
                    num_candidates=len(history),
                    instruction=instruction,
                    current_subgoal_text=relevance_query,
                    query_text_feat=query_text_feat,
                    frame_meta=frame_meta,
                )
                nov = self.scorer.novelty(hist_feats[idx], selected_feats)
                cov = self.scorer.temporal_coverage(
                    idx=idx,
                    selected_indices=selected_indices,
                    num_candidates=len(history),
                )
                turn = self.scorer.turn_bonus(frame_meta)
                total = self.scorer.total_score(rel=rel, nov=nov, cov=cov, turn=turn)

                if total > best_score:
                    best_score = total
                    best_idx = idx
                    best_payload = {"rel": rel, "nov": nov, "cov": cov, "turn": turn, "total": total, "anchor": False}

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            selected_feats.append(hist_feats[best_idx])
            score_details[best_idx] = best_payload
            index_pool.remove(best_idx)

        fallback_reason = None
        if len(selected_indices) < target_hist_slots:
            fallback_reason = "min_gap_relaxed" if self.min_frame_gap > 0 else "insufficient_candidates"
            relaxed_pool = [idx for idx in index_pool if idx not in selected_indices]
            relaxed_ranked: List[Tuple[float, int, Dict[str, float]]] = []
            for idx in relaxed_pool:
                frame_meta = meta_list[idx] if idx < len(meta_list) else None
                rel = self.scorer.relevance(
                    hist_feats[idx],
                    current_feat,
                    idx=idx,
                    num_candidates=len(history),
                    instruction=instruction,
                    current_subgoal_text=relevance_query,
                    query_text_feat=query_text_feat,
                    frame_meta=frame_meta,
                )
                nov = self.scorer.novelty(hist_feats[idx], selected_feats)
                cov = self.scorer.temporal_coverage(
                    idx=idx,
                    selected_indices=selected_indices,
                    num_candidates=len(history),
                )
                turn = self.scorer.turn_bonus(frame_meta)
                total = self.scorer.total_score(rel=rel, nov=nov, cov=cov, turn=turn)
                payload = {"rel": rel, "nov": nov, "cov": cov, "turn": turn, "total": total, "anchor": False}
                relaxed_ranked.append((total, idx, payload))

            relaxed_ranked.sort(key=lambda x: x[0], reverse=True)
            for _, idx, payload in relaxed_ranked:
                if idx in selected_indices:
                    continue
                selected_indices.append(idx)
                score_details[idx] = payload
                selected_feats.append(hist_feats[idx])
                if len(selected_indices) >= target_hist_slots:
                    break

        selected_indices = sorted(selected_indices[:target_hist_slots])
        selected_history = [history[i] for i in selected_indices]

        while len(selected_history) < target_hist_slots:
            selected_history.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

        final_frames = selected_history + [current_frame]
        debug_info = {
            "strategy": self.strategy_name,
            "fallback": fallback_reason,
            "selected_indices": selected_indices,
            "score_details": score_details,
            "query_source": query_source,
            "relevance_query": relevance_query,
            "parser_source": self.parser_source,
            "min_frame_gap": self.min_frame_gap,
            "min_gap_rejected_count": len(min_gap_rejected_indices),
            "min_gap_rejected_examples": sorted(set(min_gap_rejected_indices))[:8],
            "text_semantic_enabled": bool(self.scorer.use_text_semantic_relevance),
            "stage_aware_routing_enabled": bool(self.enable_stage_aware_routing),
            "cov_in_score_enabled": bool(self.scorer.enable_cov_in_score),
            "turn_in_score_enabled": bool(self.scorer.enable_turn_in_score),
            "state": self.state_tracker.as_dict(),
            "trace": self.state_tracker.get_trace_text(),
        }
        return final_frames, debug_info

    def _select_frames_candidate_b_v1(
        self,
        history_frames: List[Image.Image],
        current_frame: Image.Image,
        instruction: str,
        num_frames: int,
        width: int = 512,
        height: int = 512,
    ) -> Tuple[List[Image.Image], Dict[str, object]]:
        history = list(history_frames)
        max_hist_frames = int(self.cfg.BUFFER.MAX_HISTORY_FRAMES)
        if len(history) > max_hist_frames:
            history = history[-max_hist_frames:]

        if num_frames is None or num_frames <= 1:
            return [current_frame], {"strategy": self.strategy_name, "selected_indices": []}

        target_hist_slots = max(1, int(num_frames) - 1)
        desired_uniform = 6
        desired_relevance = 2
        relevance_k = min(desired_relevance, max(0, target_hist_slots - 1))
        uniform_k = max(1, target_hist_slots - relevance_k)

        # Uniform pool always preserves the initial frame (idx=0).
        if len(history) <= uniform_k:
            uniform_indices = list(range(len(history)))
        else:
            uniform_indices = np.linspace(0, len(history) - 1, num=uniform_k, dtype=int).tolist()
            if 0 not in uniform_indices:
                uniform_indices[0] = 0
            uniform_indices = sorted(set(int(i) for i in uniform_indices))
            while len(uniform_indices) < uniform_k:
                for i in range(len(history)):
                    if i not in uniform_indices:
                        uniform_indices.append(i)
                    if len(uniform_indices) >= uniform_k:
                        break
            uniform_indices = sorted(uniform_indices[:uniform_k])

        meta_list = list(self.frame_meta)[-len(history) :]
        query_text_feat = self.scorer.compute_text_feature(instruction)

        remaining = [i for i in range(len(history)) if i not in uniform_indices]
        rel_ranked = []
        for idx in remaining:
            frame_meta = meta_list[idx] if idx < len(meta_list) else None
            rel = self.scorer.relevance_instruction_only(
                idx=idx,
                num_candidates=len(history),
                instruction=instruction,
                frame_meta=frame_meta,
                query_text_feat=query_text_feat,
            )
            rel_ranked.append((rel, idx))
        rel_ranked.sort(key=lambda x: x[0], reverse=True)
        rel_indices = [idx for _, idx in rel_ranked[:relevance_k]]

        selected_indices = sorted(set(uniform_indices + rel_indices))
        if len(selected_indices) > target_hist_slots:
            # Prefer keeping the initial anchor and relevance picks.
            keep = {0}
            keep.update(rel_indices)
            tail_uniform = [i for i in uniform_indices if i not in keep]
            while len(keep) < target_hist_slots and len(tail_uniform) > 0:
                keep.add(tail_uniform.pop())
            selected_indices = sorted(keep)

        if len(selected_indices) < target_hist_slots:
            for i in range(len(history)):
                if i not in selected_indices:
                    selected_indices.append(i)
                if len(selected_indices) >= target_hist_slots:
                    break
            selected_indices = sorted(selected_indices)

        selected_history = [history[i] for i in selected_indices[:target_hist_slots]]
        while len(selected_history) < target_hist_slots:
            selected_history.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

        final_frames = selected_history + [current_frame]
        debug_info = {
            "strategy": self.strategy_name,
            "fallback": None,
            "selected_indices": selected_indices[:target_hist_slots],
            "uniform_indices": uniform_indices,
            "relevance_indices": rel_indices,
            "query_source": "instruction",
            "relevance_query": instruction,
            "parser_source": self.parser_source,
            "uniform_target": desired_uniform,
            "relevance_target": desired_relevance,
            "uniform_used": len(uniform_indices),
            "relevance_used": len(rel_indices),
            "state": self.state_tracker.as_dict(),
            "trace": self.state_tracker.get_trace_text(),
        }
        return final_frames, debug_info
