"""Reusable OpenRouter (OpenAI-compatible) LLM service.

Single place where all agents talk to the hosted LLM. Handles retries,
timeouts, JSON parsing, and graceful failure. The API key lives only
on the backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

import httpx

from backend.config import config
from backend.utils.errors import (
    LLMConfigError,
    LLMError,
    LLMQuotaError,
    LLMTimeoutError,
)

logger = logging.getLogger("analyst.llm")

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _extract_json_text(raw: str) -> str:
    """Pull JSON out of a model reply (handles fenced blocks)."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = min(pos for pos in (text.find("{"), text.find("[")) if pos != -1) if ("{" in text or "[" in text) else -1
    if start >= 0:
        # trim trailing text after last matching bracket keep simple: try whole, then substring
        text = text[start:]
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # fallback: try to locate the object
    depth_chars = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in depth_chars:
            depth = 0
            for j in range(i, len(text)):
                if text[j] == ch:
                    depth += 1
                elif text[j] == depth_chars[ch]:
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
    return text


def try_parse_json(raw: str) -> Any | None:
    """Attempt to parse JSON from an arbitrary model reply."""
    candidates = [raw.strip()]
    candidates.append(_extract_json_text(raw))
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


class LLMService:
    """Thin wrapper around OpenRouter /chat/completions."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model or config.OPENROUTER_MODEL
        self.base_url = config.OPENROUTER_BASE_URL
        self.timeout = config.LLM_TIMEOUT_SECONDS
        if not self.api_key or "your_key" in self.api_key:
            raise LLMConfigError(
                "OPENROUTER_API_KEY is not configured. Add it to backend/.env."
            )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout, connect=15.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat request and return the raw text reply."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            "max_tokens": max_tokens,
        }
        return await self._post(payload)

    async def chat_json(
        self,
        system: str,
        user: str,
        *,
        validator: Callable[[Any], Any] | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        max_attempts: int = 3,
    ) -> Any:
        """Send a request and parse a JSON reply, retrying on bad JSON."""
        last_err: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                raw = await self.chat(system, user, temperature=temperature, max_tokens=max_tokens)
                data = try_parse_json(raw)
                if data is None:
                    raise LLMError(f"Model did not return valid JSON (attempt {attempt}).")
                if validator is not None:
                    try:
                        data = validator(data)
                    except ValueError as exc:
                        raise LLMError(f"Validator rejected model output: {exc}") from exc
                return data
            except LLMQuotaError:
                raise  # do not burn retries on insufficient credits
            except LLMError as exc:
                last_err = str(exc)
                logger.warning("LLM JSON retry %s: %s", attempt, last_err)
                if attempt == max_attempts:
                    raise LLMError(
                        f"LLM could not produce valid structured output after {max_attempts} attempts: {last_err}"
                    ) from exc
        raise LLMError(f"LLM JSON failed: {last_err}")

    async def _post(self, payload: dict) -> str:
        attempts = 0
        while True:
            attempts += 1
            try:
                resp = await self._client.post(
                    "/chat/completions", json=payload
                )
            except httpx.TimeoutException as exc:
                logger.error("LLM timeout: %s", exc)
                if attempts >= 2:
                    raise LLMTimeoutError("LLM request timed out.") from exc
                await asyncio.sleep(min(attempts * 2, 5))
                continue
            except httpx.HTTPError as exc:
                if attempts >= 2:
                    raise LLMError(f"LLM transport error: {exc}") from exc
                await asyncio.sleep(min(attempts * 2, 5))
                continue

            if resp.status_code == 200:
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    raise LLMError("LLM response missing content.") from exc

            if resp.status_code in _RETRYABLE_STATUS and attempts < 3:
                wait = _parse_retry_after(resp.headers.get("retry-after"))
                if wait is None:
                    wait = min(1.5 * attempts * attempts, 10)
                logger.warning("LLM retryable status %s, waiting %.1fs", resp.status_code, wait)
                await asyncio.sleep(wait)
                continue

            if resp.status_code in (401, 402, 403, 404):
                reason = _classify_http_error(resp)
                if resp.status_code == 402:
                    raise LLMQuotaError(
                        f"OpenRouter credit/quota limit reached: {reason}"
                    )
                raise LLMError(f"OpenRouter API error {resp.status_code}: {reason}")

            raise LLMError(f"OpenRouter API error {resp.status_code}: {resp.text[:300]}")


def _classify_http_error(resp) -> str:
    """Human-readable reason for auth/credit/not-found errors (no retry)."""
    try:
        body = resp.json()
        msg = (body.get("error") or {}).get("message") or resp.text
    except Exception:  # noqa: BLE001
        msg = resp.text
    if resp.status_code == 401:
        return "Authentication failed. Check OPENROUTER_API_KEY."
    if resp.status_code == 403:
        return "Access denied by OpenRouter."
    if resp.status_code == 404:
        return f"Model not found: {self.model}"
    return str(msg)[:200]


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None