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
        self.parser = SubgoalParser(
            cache_dir=str(memory_cfg.SUBGOAL_CACHE_DIR),
            enabled=bool(memory_cfg.USE_SUBGOAL_PARSER),
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
            turn_threshold_deg=float(memory_cfg.TURN_THRESHOLD_DEGREES),
        )

        self.episode_id: Optional[str] = None
        self.step_id = 0
        self.frame_meta: Deque[Dict[str, float]] = deque(maxlen=int(memory_cfg.BUFFER.MAX_HISTORY_FRAMES))
        self.history_feats: List[np.ndarray] = []
        self._cached_hist_len = 0
        self._fallback_warned = False

    def reset_episode(self, episode_id: str, instruction: str):
        subgoals = self.parser.parse(instruction)
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
            return [current_frame], {"strategy": "candidate_a", "selected_indices": []}

        if len(history_frames) == 0:
            frames = self._uniform_sample([current_frame], num_frames=num_frames, width=width, height=height)
            return frames, {
                "strategy": "candidate_a",
                "fallback": "empty_history",
                "selected_indices": [],
                "state": self.state_tracker.as_dict(),
                "trace": self.state_tracker.get_trace_text(),
            }

        history = list(history_frames)
        max_hist_frames = int(self.cfg.BUFFER.MAX_HISTORY_FRAMES)
        truncated = False
        if len(history) > max_hist_frames:
            history = history[-max_hist_frames:]
            truncated = True

        target_hist_slots = num_frames - 1

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
                "turn": 0.0,
                "total": 1.0,
                "anchor": True,
            }
        }

        index_pool = list(range(1, len(history)))
        meta_list = list(self.frame_meta)[-len(history) :]

        while len(selected_indices) < min(target_hist_slots, len(history)) and len(index_pool) > 0:
            best_idx = None
            best_score = -1e9
            best_payload = None

            for idx in index_pool:
                frame_meta = meta_list[idx] if idx < len(meta_list) else None
                rel = self.scorer.relevance(
                    hist_feats[idx],
                    current_feat,
                    idx=idx,
                    num_candidates=len(history),
                    instruction=instruction,
                    frame_meta=frame_meta,
                )
                nov = self.scorer.novelty(hist_feats[idx], selected_feats)
                turn = self.scorer.turn_bonus(frame_meta)
                total = self.scorer.total_score(rel=rel, nov=nov, turn=turn)

                if total > best_score:
                    best_score = total
                    best_idx = idx
                    best_payload = {"rel": rel, "nov": nov, "turn": turn, "total": total, "anchor": False}

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            selected_feats.append(hist_feats[best_idx])
            score_details[best_idx] = best_payload
            index_pool.remove(best_idx)

        if len(selected_indices) < target_hist_slots:
            for idx in reversed(range(len(history))):
                if idx in selected_indices:
                    continue
                selected_indices.append(idx)
                score_details[idx] = score_details.get(
                    idx,
                    {"rel": 0.0, "nov": 0.0, "turn": 0.0, "total": 0.0, "anchor": False},
                )
                if len(selected_indices) >= target_hist_slots:
                    break

        selected_indices = sorted(selected_indices[:target_hist_slots])
        selected_history = [history[i] for i in selected_indices]

        while len(selected_history) < target_hist_slots:
            selected_history.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

        final_frames = selected_history + [current_frame]
        debug_info = {
            "strategy": "candidate_a",
            "fallback": None,
            "selected_indices": selected_indices,
            "score_details": score_details,
            "state": self.state_tracker.as_dict(),
            "trace": self.state_tracker.get_trace_text(),
        }
        return final_frames, debug_info
