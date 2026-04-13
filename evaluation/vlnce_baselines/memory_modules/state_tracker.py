from collections import deque
from typing import Deque, Dict, List, Optional


class StateTracker:
    """Lightweight online state tracker for Candidate A MVP."""

    def __init__(self, max_actions: int = 8, turn_threshold_deg: float = 20.0, use_trace: bool = False):
        self.max_actions = max_actions
        self.turn_threshold_deg = turn_threshold_deg
        self.use_trace = use_trace
        self.subgoals: List[str] = []
        self.current_subgoal_id = 0
        self.completed_subgoal_ids = set()
        self.last_milestone_text = ""
        self.recent_actions: Deque[int] = deque(maxlen=max_actions)

    def reset(self, subgoals: List[str]):
        self.subgoals = subgoals if subgoals else []
        self.current_subgoal_id = 0
        self.completed_subgoal_ids = set()
        self.last_milestone_text = ""
        self.recent_actions.clear()

    def update(self, action_id: int, yaw_delta: Optional[float] = None, cue: Optional[str] = None):
        if action_id is None:
            return

        self.recent_actions.append(int(action_id))
        abs_turn = abs(float(yaw_delta)) if yaw_delta is not None else 0.0

        should_advance = False
        if abs_turn >= self.turn_threshold_deg:
            should_advance = True
            self.last_milestone_text = "turn"
        elif cue:
            should_advance = True
            self.last_milestone_text = cue

        if should_advance and self.subgoals:
            if self.current_subgoal_id < len(self.subgoals) - 1:
                self.completed_subgoal_ids.add(self.current_subgoal_id)
                self.current_subgoal_id += 1

    def as_dict(self) -> Dict[str, object]:
        return {
            "current_subgoal_id": self.current_subgoal_id,
            "completed_subgoal_ids": sorted(list(self.completed_subgoal_ids)),
            "last_milestone_text": self.last_milestone_text,
            "recent_actions": list(self.recent_actions),
        }

    def get_trace_text(self) -> str:
        if not self.use_trace:
            return ""
        payload = self.as_dict()
        return (
            f"stage={payload['current_subgoal_id']};"
            f"recent_actions={payload['recent_actions']};"
            f"last_cue={payload['last_milestone_text'] or 'none'}"
        )

    def get_current_subgoal_text(self) -> Optional[str]:
        if not self.subgoals:
            return None
        if self.current_subgoal_id < 0 or self.current_subgoal_id >= len(self.subgoals):
            return None
        return self.subgoals[self.current_subgoal_id]

    def is_final_stage(self) -> bool:
        if not self.subgoals:
            return True
        return self.current_subgoal_id >= len(self.subgoals) - 1
