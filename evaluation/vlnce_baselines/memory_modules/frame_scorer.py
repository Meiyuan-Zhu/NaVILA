from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


class FrameScorer:
    """Lightweight frame scorer for Candidate A MVP.

    Relevance: similarity to current observation + recency prior + instruction turn hint.
    Novelty: 1 - max similarity to selected frames.
    Turn bonus: binary bonus from frame metadata (turn magnitude >= threshold).
    """

    def __init__(self, weight_rel: float, weight_nov: float, weight_turn: float, turn_threshold_deg: float):
        self.weight_rel = weight_rel
        self.weight_nov = weight_nov
        self.weight_turn = weight_turn
        self.turn_threshold_deg = turn_threshold_deg

    def compute_feature(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape

        step_h = max(h // 12, 1)
        step_w = max(w // 12, 1)
        pooled = arr[::step_h, ::step_w, :]
        pooled = pooled[:12, :12, :].reshape(-1)
        pooled = pooled / 255.0

        hist_parts = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 255), density=True)
            hist_parts.append(hist)
        hist_feat = np.concatenate(hist_parts, axis=0)

        feat = np.concatenate([pooled, hist_feat], axis=0).astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-8
        return feat / norm

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _instruction_turn_hint(instruction: str) -> int:
        text = (instruction or "").lower()
        has_left = "left" in text
        has_right = "right" in text
        if has_left and not has_right:
            return 2
        if has_right and not has_left:
            return 3
        return -1

    def relevance(
        self,
        feat: np.ndarray,
        current_feat: np.ndarray,
        idx: int,
        num_candidates: int,
        instruction: str,
        frame_meta: Optional[Dict[str, float]] = None,
    ) -> float:
        visual_rel = self.cosine(feat, current_feat)
        recency = float(idx + 1) / float(max(num_candidates, 1))

        instruction_hint = self._instruction_turn_hint(instruction)
        action_bonus = 0.0
        if frame_meta and instruction_hint in (2, 3):
            if int(frame_meta.get("action", -1)) == instruction_hint:
                action_bonus = 0.15

        rel = 0.8 * visual_rel + 0.2 * recency + action_bonus
        return float(max(min(rel, 1.5), -1.0))

    def novelty(self, feat: np.ndarray, selected_feats: Sequence[np.ndarray]) -> float:
        if len(selected_feats) == 0:
            return 1.0
        sims = [self.cosine(feat, s) for s in selected_feats]
        max_sim = max(sims)
        return float(max(0.0, 1.0 - max_sim))

    def turn_bonus(self, frame_meta: Optional[Dict[str, float]]) -> float:
        if not frame_meta:
            return 0.0
        turn_deg = abs(float(frame_meta.get("turn_deg", 0.0)))
        return 1.0 if turn_deg >= self.turn_threshold_deg else 0.0

    def total_score(self, rel: float, nov: float, turn: float) -> float:
        return float(self.weight_rel * rel + self.weight_nov * nov + self.weight_turn * turn)
