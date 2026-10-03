"""LLM-клиент с поддержкой GigaChat и OpenRouter.

Провайдер выбирается через LLM_PROVIDER в .env:
    gigachat   - основной вариант, работает из РФ без VPN
    openrouter - только через VPN
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests
import urllib3
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.config import LLM_CACHE_PATH, settings

logger = logging.getLogger(__name__)


class AllModelsExhaustedError(RuntimeError):
    """Все модели или провайдеры сейчас недоступны."""


class GigaChatAuthError(RuntimeError):
    """Ошибка авторизации в GigaChat."""


GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_BASE = "https://gigachat.devices.sberbank.ru/api/v1"

# GigaChat использует корневой сертификат Минцифры, которого нет в стандартном хранилище Python
# Отключаем проверку SSL, как рекомендует документация Сбера.
VERIFY_SSL = False


@dataclass
class _Token:
    value: str
    expires_at: float

    @property
    def valid(self) -> bool:
        # запас 60 секунд, чтобы токен не истёк в момент запроса
        return time.time() < self.expires_at - 60


class _GigaChatBackend:
    """Бэкенд GigaChat с автопродлением access_token."""

    def __init__(self) -> None:
        if not settings.GIGACHAT_AUTH_KEY:
            raise GigaChatAuthError(
                "GIGACHAT_AUTH_KEY не задан в .env"
            )
        self._token: _Token | None = None
        self._model: str = settings.GIGACHAT_MODEL

    def _refresh_token(self) -> None:
        headers = {
            "Authorization": f"Basic {settings.GIGACHAT_AUTH_KEY}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"scope": settings.GIGACHAT_SCOPE}
        response = requests.post(
            GIGACHAT_OAUTH_URL,
            headers=headers,
            data=data,
            timeout=30,
            verify=VERIFY_SSL,
        )
        if response.status_code != 200:
            raise GigaChatAuthError(
                f"OAuth GigaChat вернул {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        token_value = payload.get("access_token")
        expires_at = payload.get("expires_at", 0) / 1000  # мс -> сек
        if not token_value:
            raise GigaChatAuthError("GigaChat не вернул access_token")
        self._token = _Token(value=token_value, expires_at=expires_at)
        logger.info("GigaChat: получен новый access_token")

    def _ensure_token(self) -> str:
        if self._token is None or not self._token.valid:
            self._refresh_token()
        assert self._token is not None
        return self._token.value

    def complete(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        token = self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{GIGACHAT_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
            verify=VERIFY_SSL,
        )
        if response.status_code == 401:
            # токен мог истечь между проверкой и отправкой
            self._token = None
            token = self._ensure_token()
            headers["Authorization"] = f"Bearer {token}"
            response = requests.post(
                f"{GIGACHAT_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
                verify=VERIFY_SSL,
            )
        if response.status_code == 429:
            raise AllModelsExhaustedError("GigaChat: превышен лимит запросов")
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Неожиданный ответ GigaChat: {data}") from e



@dataclass
class _ModelState:
    name: str
    cooldown_until: float = 0.0

    @property
    def available(self) -> bool:
        return time.time() >= self.cooldown_until

    def cooldown(self, seconds: int) -> None:
        self.cooldown_until = time.time() + seconds
        until = time.strftime("%H:%M:%S", time.localtime(self.cooldown_until))
        logger.warning(
            "Модель %s ушла в остывание на %d секунд (до %s)",
            self.name,
            seconds,
            until,
        )


class _OpenRouterBackend:
    COOLDOWN_SECONDS: int = 15 * 60

    def __init__(self) -> None:
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY не задан в .env")
        from openai import OpenAI

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )
        self._models = [_ModelState(name=m) for m in settings.OPENROUTER_MODELS]
        self._active_idx = 0

    def complete(
        self,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        from openai import APIStatusError

        tried: set[int] = set()
        last_error: Exception | None = None

        while len(tried) < len(self._models):
            idx = self._pick_next(tried)
            if idx is None:
                break
            tried.add(idx)
            state = self._models[idx]
            try:
                response = self._client.chat.completions.create(
                    model=state.name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._active_idx = idx
                return (response.choices[0].message.content or "").strip()
            except APIStatusError as e:
                status = e.status_code
                if status in (401, 403):
                    raise AllModelsExhaustedError(
                        f"OpenRouter вернул {status}. "
                        f"Похоже на геоблок, включи VPN или используй GigaChat."
                    ) from e
                if status in (429, 404, 503):
                    state.cooldown(self.COOLDOWN_SECONDS)
                    last_error = e
                    continue
                state.cooldown(60)
                last_error = e
            except Exception as e:
                state.cooldown(60)
                last_error = e

        raise AllModelsExhaustedError(
            "Все модели OpenRouter недоступны"
        ) from last_error

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "model": m.name,
                "available": m.available,
                "cooldown_in_sec": max(0, int(m.cooldown_until - now)),
                "is_active": i == self._active_idx,
            }
            for i, m in enumerate(self._models)
        ]

    def _pick_next(self, tried: set[int]) -> int | None:
        n = len(self._models)
        for offset in range(n):
            idx = (self._active_idx + offset) % n
            if idx in tried:
                continue
            if self._models[idx].available:
                return idx
        return None



class _LLMCache:
    def __init__(self, path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def _key(provider: str, prompt: str, system: str, temperature: float) -> str:
        raw = f"{provider}|{system}|{temperature}|{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, provider: str, prompt: str, system: str, temperature: float
    ) -> str | None:
        return self._data.get(self._key(provider, prompt, system, temperature))

    def set(
        self,
        provider: str,
        prompt: str,
        system: str,
        temperature: float,
        value: str,
    ) -> None:
        self._data[self._key(provider, prompt, system, temperature)] = value
        try:
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Не удалось сохранить кэш LLM: %s", e)


class LLMClient:
    """Универсальный клиент: GigaChat или OpenRouter."""

    def __init__(self) -> None:
        provider = settings.LLM_PROVIDER.lower()
        if provider == "gigachat":
            self._backend: Any = _GigaChatBackend()
        elif provider == "openrouter":
            self._backend = _OpenRouterBackend()
        else:
            raise ValueError(
                f"Неизвестный LLM_PROVIDER: {settings.LLM_PROVIDER}. "
                f"Допустимо: gigachat, openrouter"
            )
        self._provider_name = provider
        self._cache = (
            _LLMCache(LLM_CACHE_PATH) if settings.ENABLE_LLM_CACHE else None
        )
        logger.info("LLM: провайдер=%s", provider)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=6),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        system: str = "Ты полезный ассистент. Отвечай кратко и по делу.",
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        if self._cache is not None:
            cached = self._cache.get(
                self._provider_name, prompt, system, temperature
            )
            if cached is not None:
                return cached

        text = self._backend.complete(prompt, system, temperature, max_tokens)

        if self._cache is not None:
            self._cache.set(
                self._provider_name, prompt, system, temperature, text
            )
        return text

    def status(self) -> list[dict[str, Any]]:
        if isinstance(self._backend, _OpenRouterBackend):
            return self._backend.status()
        return [
            {
                "model": f"GigaChat ({settings.GIGACHAT_MODEL})",
                "available": True,
                "cooldown_in_sec": 0,
                "is_active": True,
            }
        ]


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    """Синглтон LLM-клиента."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm