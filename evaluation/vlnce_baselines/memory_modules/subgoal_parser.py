import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from typing import List


class SubgoalParser:
    """Subgoal parser with optional LLM/API path and rule fallback."""

    def __init__(
        self,
        cache_dir: str,
        enabled: bool = False,
        use_llm: bool = False,
        backend: str = "openai_compatible",
        model: str = "qwen-flash",
        api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY",
        timeout_seconds: int = 8,
        max_subgoals: int = 8,
        fallback_to_rule: bool = True,
    ):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.use_llm = use_llm
        self.backend = backend
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_subgoals = max_subgoals
        self.fallback_to_rule = fallback_to_rule
        self.last_source = "rule"
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse(self, instruction: str) -> List[str]:
        if not instruction:
            return []
        if not self.enabled:
            self.last_source = "disabled"
            return [instruction.strip()]

        normalized_instruction = instruction.strip()
        cache_key = hashlib.sha1(normalized_instruction.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cached_subgoals, cached_source = self._parse_cached_payload(payload)
                if len(cached_subgoals) > 0:
                    self.last_source = cached_source
                    return cached_subgoals
            except Exception:
                pass

        source = "rule"
        subgoals: List[str] = []
        if self.use_llm and self.backend == "openai_compatible":
            subgoals = self._try_openai_compatible_split(normalized_instruction)
            source = "llm" if len(subgoals) > 0 else "rule"

        if len(subgoals) == 0 and self.fallback_to_rule:
            subgoals = self._rule_based_split(normalized_instruction)
            source = "rule"

        if len(subgoals) == 0:
            subgoals = [normalized_instruction]
            source = "instruction"

        self.last_source = source

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "subgoals": subgoals,
                        "source": source,
                    },
                    f,
                    ensure_ascii=True,
                    indent=2,
                )
        except Exception:
            pass

        return subgoals

    @staticmethod
    def _parse_cached_payload(payload):
        if isinstance(payload, list):
            cleaned = [str(x).strip() for x in payload if str(x).strip()]
            return cleaned, "cache_list"
        if isinstance(payload, dict):
            subgoals = payload.get("subgoals", [])
            source = str(payload.get("source", "cache_dict"))
            if isinstance(subgoals, list):
                cleaned = [str(x).strip() for x in subgoals if str(x).strip()]
                return cleaned, source
        return [], "cache_invalid"

    def _try_openai_compatible_split(self, instruction: str) -> List[str]:
        api_key = os.environ.get(self.api_key_env, "")
        if api_key == "":
            return []

        prompt = (
            "Split the navigation instruction into an ordered list of short subgoals. "
            "Return strict JSON with shape {\"subgoals\": [\"...\"]}. "
            "Do not include explanations."
        )
        user_content = f"Instruction: {instruction}"

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            url=f"{self.api_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=max(1, int(self.timeout_seconds))) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return []
        except Exception:
            return []

        try:
            response_json = json.loads(body)
            message = response_json["choices"][0]["message"]["content"]
            parsed = json.loads(message)
            raw_subgoals = parsed.get("subgoals", [])
            if not isinstance(raw_subgoals, list):
                return []

            cleaned = []
            for entry in raw_subgoals:
                text = str(entry).strip().strip(".")
                if text:
                    cleaned.append(text)

            if len(cleaned) == 0:
                return []

            return cleaned[: max(1, int(self.max_subgoals))]
        except Exception:
            return []

    @staticmethod
    def _rule_based_split(instruction: str) -> List[str]:
        text = instruction.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" then ", ", ")
        text = text.replace(" and then ", ", ")
        parts = re.split(r",|;|\band\b", text)
        parts = [p.strip(" .") for p in parts if p and p.strip(" .")]
        return parts if parts else [text]
