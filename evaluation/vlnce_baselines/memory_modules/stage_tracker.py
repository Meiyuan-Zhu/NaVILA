import json
import re
from typing import Dict, List


class StageTracker:
    """Lightweight stage parser and constraint enforcer for candidate_b_v3."""

    def __init__(
        self,
        interval: int = 10,
        max_stage_delta: int = 1,
        confidence_threshold: float = 0.0,
        max_evidence_chars: int = 180,
    ):
        self.interval = max(1, int(interval))
        self.max_stage_delta = max(0, int(max_stage_delta))
        self.confidence_threshold = float(confidence_threshold)
        self.max_evidence_chars = max(32, int(max_evidence_chars))
        self.subgoals: List[str] = []

    def reset(self, subgoals: List[str]):
        self.subgoals = list(subgoals or [])

    def should_infer(self, step_id: int) -> bool:
        return int(step_id) > 0 and int(step_id) % self.interval == 0

    def build_prompt(self, previous_stage_id: int, recent_actions: List[int]) -> str:
        stage_lines = []
        if len(self.subgoals) == 0:
            stage_lines.append("1. no_subgoals_available")
        else:
            for idx, sg in enumerate(self.subgoals):
                stage_lines.append(f"{idx}. {sg}")

        prev_stage = int(previous_stage_id)
        actions_text = ", ".join(str(int(a)) for a in list(recent_actions or []))
        if actions_text == "":
            actions_text = "none"

        return (
            "You are a stage tracker for embodied navigation. "
            "Given subgoals, current observation image, recent actions, and previous stage id, "
            "predict current stage. "
            "Constraint: current_stage_id must be an integer index in the valid subgoal range. "
            "Return strict JSON only with keys: current_stage_id (int), confidence (float in [0,1]), evidence (string).\n"
            f"Previous stage id: {prev_stage}\n"
            f"Recent actions (latest up to 8): [{actions_text}]\n"
            "Subgoals:\n"
            + "\n".join(stage_lines)
        )

    def parse_response(self, text: str, previous_stage_id: int) -> Dict[str, object]:
        previous_stage = int(previous_stage_id)
        num_subgoals = max(1, len(self.subgoals))
        minimum_allowed = 0
        maximum_allowed = num_subgoals - 1

        parsed_stage = previous_stage
        parsed_confidence = 0.0
        parsed_evidence = "fallback_to_previous_stage"
        validity = "fallback"

        payload = self._extract_json_payload(text)
        if payload is not None:
            if "current_stage_id" in payload:
                try:
                    parsed_stage = int(payload.get("current_stage_id"))
                except Exception:
                    parsed_stage = previous_stage
            if "confidence" in payload:
                try:
                    parsed_confidence = float(payload.get("confidence"))
                except Exception:
                    parsed_confidence = 0.0
            if "evidence" in payload:
                parsed_evidence = str(payload.get("evidence") or "")
            validity = "parsed"

        constrained_stage = max(minimum_allowed, min(maximum_allowed, int(parsed_stage)))
        if constrained_stage != parsed_stage:
            validity = validity + ":constrained_range"

        if parsed_confidence < self.confidence_threshold:
            constrained_stage = previous_stage
            validity = validity + ":low_confidence_fallback"

        parsed_confidence = max(0.0, min(1.0, parsed_confidence))
        parsed_evidence = re.sub(r"\s+", " ", parsed_evidence).strip()
        if parsed_evidence == "":
            parsed_evidence = "no_evidence"
        parsed_evidence = parsed_evidence[: self.max_evidence_chars]

        return {
            "current_stage_id": int(constrained_stage),
            "confidence": float(parsed_confidence),
            "evidence": parsed_evidence,
            "validity": validity,
            "previous_stage_id": int(previous_stage),
            "allowed_min": int(minimum_allowed),
            "allowed_max": int(maximum_allowed),
        }

    @staticmethod
    def _extract_json_payload(text: str):
        if not text:
            return None

        raw = str(text).strip()
        try:
            candidate = json.loads(raw)
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, dict):
                return candidate
        except Exception:
            return None
        return None
