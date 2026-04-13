from typing import Dict, List, Optional, Sequence

import numpy as np
import re
import hashlib
from PIL import Image


class FrameScorer:
    """Lightweight frame scorer for Candidate A MVP.

    Relevance: similarity to current observation + recency prior + instruction turn hint.
    Novelty: 1 - max similarity to selected frames.
    Turn bonus: binary bonus from frame metadata (turn magnitude >= threshold).
    """

    def __init__(
        self,
        weight_rel: float,
        weight_nov: float,
        weight_turn: float,
        weight_cov: float,
        turn_threshold_deg: float,
        enable_cov_in_score: bool = False,
        enable_turn_in_score: bool = False,
        use_text_semantic_relevance: bool = True,
        text_embed_dim: int = 128,
    ):
        self.weight_rel = weight_rel
        self.weight_nov = weight_nov
        self.weight_turn = weight_turn
        self.weight_cov = weight_cov
        self.turn_threshold_deg = turn_threshold_deg
        self.enable_cov_in_score = bool(enable_cov_in_score)
        self.enable_turn_in_score = bool(enable_turn_in_score)
        self.use_text_semantic_relevance = use_text_semantic_relevance
        self.text_embed_dim = max(16, int(text_embed_dim))

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
    def _cosine_to_unit_interval(cosine_val: float) -> float:
        mapped = 0.5 * (float(cosine_val) + 1.0)
        return float(max(0.0, min(1.0, mapped)))

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

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return re.findall(r"[a-z]+", (text or "").lower())

    def compute_text_feature(self, text: str) -> np.ndarray:
        vec = np.zeros(self.text_embed_dim, dtype=np.float32)
        tokens = self._tokens(text)
        if len(tokens) == 0:
            return vec

        for tok in tokens:
            digest = hashlib.sha1(tok.encode("utf-8")).digest()
            h = int.from_bytes(digest[:8], byteorder="big", signed=False)
            idx = h % self.text_embed_dim
            sign = 1.0 if (h % 2) == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def _frame_intent_feature(self, idx: int, num_candidates: int, frame_meta: Optional[Dict[str, float]]) -> np.ndarray:
        hints = ["history", "past"]
        if num_candidates > 0 and idx >= num_candidates * 0.75:
            hints.extend(["recent", "now", "current"]) 

        if frame_meta:
            action = int(frame_meta.get("action", -1))
            turn_deg = abs(float(frame_meta.get("turn_deg", 0.0)))
            if action == 1:
                hints.extend(["forward", "move", "straight"])
            elif action == 2:
                hints.extend(["left", "turn", "junction"])
            elif action == 3:
                hints.extend(["right", "turn", "junction"])
            elif action == 0:
                hints.extend(["stop", "goal"])

            if turn_deg >= self.turn_threshold_deg:
                hints.extend(["turn", "corner", "junction"])

        return self.compute_text_feature(" ".join(hints))

    def relevance(
        self,
        feat: np.ndarray,
        current_feat: np.ndarray,
        idx: int,
        num_candidates: int,
        instruction: str,
        current_subgoal_text: Optional[str] = None,
        query_text_feat: Optional[np.ndarray] = None,
        frame_meta: Optional[Dict[str, float]] = None,
    ) -> float:
        visual_rel = self._cosine_to_unit_interval(self.cosine(feat, current_feat))
        recency = float(idx + 1) / float(max(num_candidates, 1))
        recency = float(max(0.0, min(1.0, recency)))

        text_rel = 0.0
        if self.use_text_semantic_relevance:
            if query_text_feat is None:
                query_text_feat = self.compute_text_feature(current_subgoal_text or instruction)
            frame_text_feat = self._frame_intent_feature(idx=idx, num_candidates=num_candidates, frame_meta=frame_meta)
            text_rel = self._cosine_to_unit_interval(self.cosine(query_text_feat, frame_text_feat))

        rel = 0.9 * visual_rel + 0.1 * recency
        if self.use_text_semantic_relevance:
            rel = 0.7 * visual_rel + 0.1 * recency + 0.2 * text_rel
        return float(max(0.0, min(1.0, rel)))

    def relevance_instruction_only(
        self,
        idx: int,
        num_candidates: int,
        instruction: str,
        frame_meta: Optional[Dict[str, float]] = None,
        query_text_feat: Optional[np.ndarray] = None,
    ) -> float:
        """Instruction-only relevance for Candidate B v1.

        Uses text intent similarity only, intentionally ignoring visual similarity and recency.
        """
        if query_text_feat is None:
            query_text_feat = self.compute_text_feature(instruction)
        frame_text_feat = self._frame_intent_feature(
            idx=idx,
            num_candidates=num_candidates,
            frame_meta=frame_meta,
        )
        rel = self._cosine_to_unit_interval(self.cosine(query_text_feat, frame_text_feat))
        return float(max(0.0, min(1.0, rel)))

    def novelty(self, feat: np.ndarray, selected_feats: Sequence[np.ndarray]) -> float:
        if len(selected_feats) == 0:
            return 1.0
        sims = [self.cosine(feat, s) for s in selected_feats]
        max_sim = self._cosine_to_unit_interval(max(sims))
        nov = 1.0 - max_sim
        return float(max(0.0, min(1.0, nov)))

    def turn_bonus(self, frame_meta: Optional[Dict[str, float]]) -> float:
        if not frame_meta:
            return 0.0
        turn_deg = abs(float(frame_meta.get("turn_deg", 0.0)))
        return 1.0 if turn_deg >= self.turn_threshold_deg else 0.0

    @staticmethod
    def temporal_coverage(idx: int, selected_indices: Sequence[int], num_candidates: int) -> float:
        if num_candidates <= 1:
            return 0.0
        if len(selected_indices) == 0:
            return 1.0

        max_gap = float(max(1, num_candidates - 1))
        min_dist = min(abs(int(idx) - int(s)) for s in selected_indices)
        cov = float(min_dist) / max_gap
        return float(max(0.0, min(1.0, cov)))

    def total_score(self, rel: float, nov: float, cov: float, turn: float = 0.0) -> float:
        total = float(self.weight_rel * rel + self.weight_nov * nov)
        if self.enable_cov_in_score:
            total += float(self.weight_cov * cov)
        if self.enable_turn_in_score:
            total += float(self.weight_turn * turn)
        return total
