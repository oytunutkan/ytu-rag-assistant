import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


class LLMClient:
    def __init__(
        self,
        daily_limit=499,
        reset_hour=11,
        timezone_name="Europe/Istanbul",
        state_path=".llm_client_state.json",
    ):
        load_dotenv()
        self.daily_limit = int(os.getenv("LLM_DAILY_LIMIT", os.getenv("GEMINI_DAILY_LIMIT", daily_limit)))
        self.reset_hour = int(os.getenv("LLM_RESET_HOUR", os.getenv("GEMINI_RESET_HOUR", reset_hour)))
        self.timezone_name = os.getenv("LLM_RESET_TIMEZONE", os.getenv("GEMINI_RESET_TIMEZONE", timezone_name))
        self.state_path = Path(os.getenv("LLM_STATE_PATH", os.getenv("GEMINI_KEY_STATE_PATH", state_path)))
        self.credentials = self._load_credentials()

        if not self.credentials:
            raise ValueError("LLM servis anahtarı bulunamadı. Ortam değişkenlerini kontrol et.")

        self.state = self._load_state()
        self._reset_if_needed()

    def _load_credentials(self):
        credentials = []

        for env_name in ("LLM_API_KEYS", "GEMINI_API_KEYS", "GOOGLE_API_KEYS"):
            raw_value = os.getenv(env_name, "")
            for item in raw_value.split(","):
                value = item.strip().strip("'").strip('"')
                if value and value not in credentials:
                    credentials.append(value)

        for i in range(1, 21):
            for env_name in (f"GOOGLE_API_KEY_{i}", f"GOOGLEAPIKEY{i}", f"GOOGLE_APIKEY_{i}"):
                value = os.getenv(env_name)
                if value and value.strip() and value.strip() not in credentials:
                    credentials.append(value.strip())

        single_value = os.getenv("GOOGLE_API_KEY")
        if single_value and single_value.strip() and single_value.strip() not in credentials:
            credentials.append(single_value.strip())

        return credentials

    def _now(self):
        try:
            return datetime.now(ZoneInfo(self.timezone_name))
        except Exception:
            return datetime.now()

    def _current_period(self):
        now = self._now()
        if now.hour < self.reset_hour:
            now = now - timedelta(days=1)
        return now.strftime("%Y-%m-%d")

    def _empty_state(self):
        return {
            "period": self._current_period(),
            "active_index": 0,
            "usage": [0 for _ in self.credentials],
            "unavailable": [False for _ in self.credentials],
        }

    def _load_state(self):
        if not self.state_path.exists():
            return self._empty_state()

        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty_state()

        if len(data.get("usage", [])) != len(self.credentials):
            return self._empty_state()

        if "unavailable" not in data and "exhausted" in data:
            data["unavailable"] = data.get("exhausted", [])
        if len(data.get("unavailable", [])) != len(self.credentials):
            data["unavailable"] = [False for _ in self.credentials]

        return data

    def _save_state(self):
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _reset_if_needed(self):
        current_period = self._current_period()
        if self.state.get("period") != current_period:
            self.state = self._empty_state()
            self._save_state()

    def _is_available(self, index):
        return (
            not self.state["unavailable"][index]
            and self.state["usage"][index] < self.daily_limit
        )

    def get_current_credential(self):
        self._reset_if_needed()
        start = int(self.state.get("active_index", 0)) % len(self.credentials)

        for offset in range(len(self.credentials)):
            index = (start + offset) % len(self.credentials)
            if self._is_available(index):
                self.state["active_index"] = index
                self._save_state()
                return index + 1, self.credentials[index]

        raise RuntimeError("LLM servisi şu anda kullanılamıyor.")

    def mark_success(self, credential_id):
        self._reset_if_needed()
        index = int(credential_id) - 1
        if 0 <= index < len(self.credentials):
            self.state["usage"][index] += 1
            if self.state["usage"][index] >= self.daily_limit:
                self.state["active_index"] = (index + 1) % len(self.credentials)
            self._save_state()

    def mark_unavailable(self, credential_id, reason=""):
        self._reset_if_needed()
        index = int(credential_id) - 1
        if 0 <= index < len(self.credentials):
            self.state["unavailable"][index] = True
            self.state["active_index"] = (index + 1) % len(self.credentials)
            self.state["last_error"] = str(reason)[:500]
            self._save_state()

    def status_text(self):
        self._reset_if_needed()
        parts = []
        for i, used in enumerate(self.state["usage"], start=1):
            unavailable = self.state["unavailable"][i - 1]
            suffix = " unavailable" if unavailable else ""
            parts.append(f"provider{i}: {used}/{self.daily_limit}{suffix}")
        return " | ".join(parts)

    @staticmethod
    def is_quota_error(error_text):
        text = str(error_text).lower()
        return any(
            marker in text
            for marker in [
                "429", "rate_limit", "rate limit", "resource_exhausted",
                "quota", "api key expired", "api_key_invalid",
                "invalid api key", "permission_denied", "403", "402",
                "requires more credits", "insufficient",
            ]
        )
