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
        require_llm_success: bool = False,
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
        self.require_llm_success = require_llm_success
        self.last_source = "rule"
        self.last_reason = "init"
        os.makedirs(self.cache_dir, exist_ok=True)

    def parse_one_shot(self, instruction: str) -> List[str]:
        """One-shot parser.

        Input: full instruction text.
        Output: ordered subgoal list.
        """
        if not instruction:
            return []
        if not self.enabled:
            self.last_source = "disabled"
            self.last_reason = "parser_disabled"
            return [instruction.strip()]

        if self.require_llm_success and not self.use_llm:
            self.last_source = "error"
            self.last_reason = "require_llm_but_use_llm_false"
            raise RuntimeError("SubgoalParser REQUIRE_LLM_SUCCESS=True but USE_LLM=False")

        normalized_instruction = instruction.strip()
        cache_key = hashlib.sha1(normalized_instruction.encode("utf-8")).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                cached_subgoals, cached_source = self._parse_cached_payload(payload)
                if len(cached_subgoals) > 0:
                    if self.require_llm_success and cached_source != "llm":
                        self.last_reason = f"cache_bypassed_non_llm:{cached_source}"
                    else:
                        self.last_reason = f"cache_hit:{cached_source}"
                        self.last_source = cached_source
                        return cached_subgoals[: max(1, int(self.max_subgoals))]
            except Exception:
                self.last_reason = "cache_read_error"

        source = "rule"
        subgoals: List[str] = []
        if self.use_llm and self.backend == "openai_compatible":
            subgoals = self._try_openai_compatible_split(normalized_instruction)
            source = "llm" if len(subgoals) > 0 else "rule"

        if self.require_llm_success and len(subgoals) == 0:
            self.last_source = "error"
            if not self.last_reason.startswith("llm_"):
                self.last_reason = "llm_empty_response"
            raise RuntimeError(f"SubgoalParser LLM required but failed: {self.last_reason}")

        if len(subgoals) == 0 and self.fallback_to_rule:
            subgoals = self._rule_based_split(normalized_instruction)
            source = "rule"
            self.last_reason = "rule_fallback"

        if len(subgoals) == 0:
            subgoals = [normalized_instruction]
            source = "instruction"
            self.last_reason = "instruction_passthrough"

        subgoals = self._clean_subgoals(subgoals)
        self.last_source = source
        if source == "llm":
            self.last_reason = "llm_success"

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

    def parse(self, instruction: str) -> List[str]:
        # Backward compatible entrypoint.
        return self.parse_one_shot(instruction)

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
            self.last_reason = f"llm_api_key_missing:{self.api_key_env}"
            return []

        prompt = (
            "You are a one-shot navigation subgoal parser. "
            "Input is one full instruction sentence. "
            "Output must be an ordered list of atomic subgoals. "
            "Return strict JSON only with schema {\"subgoals\": [\"...\"]}. "
            "Do not output explanations."
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
        except urllib.error.HTTPError as exc:
            self.last_reason = f"llm_http_error:{getattr(exc, 'code', 'unknown')}"
            return []
        except (urllib.error.URLError, TimeoutError):
            self.last_reason = "llm_network_or_timeout"
            return []
        except Exception:
            self.last_reason = "llm_unknown_transport_error"
            return []

        try:
            response_json = json.loads(body)
            message = response_json["choices"][0]["message"]["content"]
            parsed = self._parse_llm_output(message)
            if len(parsed) == 0:
                self.last_reason = "llm_empty_content"
            return parsed
        except Exception:
            self.last_reason = "llm_response_parse_error"
            return []

    def _parse_llm_output(self, content: str) -> List[str]:
        # Prefer strict JSON, but tolerate numbered lines as backup.
        try:
            parsed = json.loads(content)
            raw_subgoals = parsed.get("subgoals", []) if isinstance(parsed, dict) else []
            if isinstance(raw_subgoals, list):
                return self._clean_subgoals(raw_subgoals)
        except Exception:
            pass

        numbered = []
        for line in (content or "").splitlines():
            m = re.match(r"^\s*\d+[\.)]\s*(.+?)\s*$", line)
            if m:
                numbered.append(m.group(1))
        return self._clean_subgoals(numbered)

    def _clean_subgoals(self, raw_subgoals: List[str]) -> List[str]:
        cleaned = []
        for entry in raw_subgoals:
            text = str(entry).strip().strip(".")
            text = re.sub(r"^\s*(and|then)\s+", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text)
            if text:
                cleaned.append(text)
        if len(cleaned) == 0:
            return []
        return cleaned[: max(1, int(self.max_subgoals))]

    @staticmethod
    def _rule_based_split(instruction: str) -> List[str]:
        text = instruction.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" and then ", ", ")
        text = text.replace(" then ", ", ")
        text = re.sub(r"\bafter that\b", ",", text)
        parts = re.split(r",|;|\.(?=\s)|\band\b", text)
        cleaned = []
        for p in parts:
            seg = p.strip(" .")
            seg = re.sub(r"^\s*(and|then)\s+", "", seg, flags=re.IGNORECASE)
            if seg:
                cleaned.append(seg)
        return cleaned if cleaned else [text]
