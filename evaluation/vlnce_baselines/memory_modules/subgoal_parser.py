import hashlib
import json
import os
import re
from typing import List


class SubgoalParser:
    """Rule-based subgoal parser with local disk cache."""

    def __init__(self, cache_dir: str, enabled: bool = False):
        self.cache_dir = cache_dir
        self.enabled = enabled
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse(self, instruction: str) -> List[str]:
        if not instruction:
            return []
        if not self.enabled:
            return [instruction.strip()]

        cache_key = hashlib.sha1(instruction.strip().encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list) and len(payload) > 0:
                    return payload
            except Exception:
                pass

        subgoals = self._rule_based_split(instruction)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(subgoals, f, ensure_ascii=True, indent=2)
        except Exception:
            pass

        return subgoals

    @staticmethod
    def _rule_based_split(instruction: str) -> List[str]:
        text = instruction.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" then ", ", ")
        text = text.replace(" and then ", ", ")
        parts = re.split(r",|;|\band\b", text)
        parts = [p.strip(" .") for p in parts if p and p.strip(" .")]
        return parts if parts else [text]
