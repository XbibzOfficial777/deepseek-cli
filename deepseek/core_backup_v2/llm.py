"""
DeepSeek CLI - OpenRouter LLM Client

Async client for OpenRouter.ai API (OpenAI-compatible).
Supports streaming, function calling (tool_use), and reasoning/thinking.

Design:
  - Deep-copy messages before every call (prevents memory corruption)
  - Exponential back-off with jitter on retryable HTTP errors
  - Streaming yields structured dicts: {type, text} or {type, tool_calls}
  - Never raises exceptions during streaming - yields error dicts instead
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp


@dataclass
class ChatMessage:
    """Single chat message in OpenAI format."""
    role: str
    content: str | None = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Structured response from non-streaming completion."""
    content: str = ""
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    latency: float = 0.0


class LLMClient:
    """Async LLM client for OpenRouter with SSE streaming and function calling."""

    RETRY_CODES = {429, 500, 502, 503, 504}
    MAX_RETRIES = 3
    BACKOFF_BASE = 1.0

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "deepseek/deepseek-r1-0528:free",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._closed = False

        # Stats
        self.total_tokens_used: int = 0
        self.request_count: int = 0

    # ── Session Management ───────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("Client is closed")
        if self._session is not None and not self._session.closed:
            return self._session
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/deepseek-cli",
                "X-Title": "DeepSeek CLI Agent",
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )
        return self._session

    async def close(self) -> None:
        self._closed = True
        if self._session and not self._session.closed:
            try:
                await asyncio.shield(self._session.close())
            except Exception:
                pass
            self._session = None

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _deep_copy(msgs: List[Dict]) -> List[Dict]:
        return json.loads(json.dumps(msgs))

    async def _backoff(self, attempt: int) -> None:
        delay = self.BACKOFF_BASE * (2 ** (attempt - 1))
        jitter = delay * 0.2 * (1 if attempt % 2 == 0 else -1)
        await asyncio.sleep(max(0.1, delay + jitter))

    # ── Non-streaming ───────────────────────────────────────

    async def complete(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
    ) -> LLMResponse:
        session = await self._get_session()
        payload = {
            "model": self.model,
            "messages": self._deep_copy(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        t0 = time.time()
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status in self.RETRY_CODES:
                        await self._backoff(attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
            except aiohttp.ClientError as exc:
                if attempt < self.MAX_RETRIES:
                    await self._backoff(attempt)
                    continue
                return LLMResponse(latency=round(time.time() - t0, 3))

            self.request_count += 1
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            self.total_tokens_used += tokens

            return LLMResponse(
                content=message.get("content", "") or "",
                thinking=message.get("reasoning_content", "") or "",
                tool_calls=message.get("tool_calls", []) or [],
                tokens_used=tokens,
                latency=round(time.time() - t0, 3),
            )

        return LLMResponse(latency=round(time.time() - t0, 3))

    # ── Streaming ───────────────────────────────────────────

    async def stream(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream tokens as dicts.

        Yields:
          {"type": "thinking", "text": "..."}
          {"type": "content",  "text": "..."}
          {"type": "tool_calls", "tool_calls": [...]}
          {"type": "usage", "tokens": N}
          {"type": "error", "text": "..."}
        """
        session = await self._get_session()
        payload = {
            "model": self.model,
            "messages": self._deep_copy(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        t0 = time.time()
        last_error: Optional[str] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            session = await self._get_session()
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status in self.RETRY_CODES:
                        last_error = f"HTTP {resp.status}"
                        await self._backoff(attempt)
                        continue
                    resp.raise_for_status()

                    tool_call_buffers: Dict[int, Dict] = {}
                    usage_data: Dict = {}

                    async for chunk_dict in self._parse_sse(resp):
                        if chunk_dict.get("type") == "raw_delta":
                            delta = chunk_dict["delta"]
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                yield {"type": "thinking", "text": reasoning}
                            content = delta.get("content")
                            if content:
                                yield {"type": "content", "text": content}
                            tc_list = delta.get("tool_calls")
                            if tc_list:
                                for tc_delta in tc_list:
                                    idx = tc_delta.get("index", 0)
                                    if idx not in tool_call_buffers:
                                        tool_call_buffers[idx] = {
                                            "id": tc_delta.get("id", ""),
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments": "",
                                            },
                                        }
                                    buf = tool_call_buffers[idx]
                                    if tc_delta.get("id"):
                                        buf["id"] = tc_delta["id"]
                                    if tc_delta.get("function"):
                                        fn = tc_delta["function"]
                                        if fn.get("name"):
                                            buf["function"]["name"] += fn["name"]
                                        if fn.get("arguments"):
                                            buf["function"]["arguments"] += fn["arguments"]
                        elif chunk_dict.get("type") == "usage":
                            usage_data = chunk_dict
                        elif chunk_dict.get("type") == "error":
                            yield chunk_dict

                    if tool_call_buffers:
                        ordered = [tool_call_buffers[i] for i in sorted(tool_call_buffers.keys())]
                        yield {"type": "tool_calls", "tool_calls": ordered}

                    if usage_data:
                        tokens = usage_data.get("tokens", 0)
                        self.total_tokens_used += tokens
                        self.request_count += 1
                        yield {"type": "usage", "tokens": tokens}

                    return  # success

            except asyncio.CancelledError:
                return
            except aiohttp.ClientError as exc:
                last_error = str(exc)
                if attempt < self.MAX_RETRIES:
                    await self._backoff(attempt)
                    continue
                break
            except Exception as exc:
                yield {"type": "error", "text": f"Stream error: {exc}"}
                return

        yield {"type": "error", "text": f"Failed after {self.MAX_RETRIES} retries. Last: {last_error}"}

    @staticmethod
    async def _parse_sse(resp: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        """Parse SSE stream into structured chunks."""
        try:
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if "usage" in chunk and not choices:
                    usage = chunk["usage"]
                    total = usage.get("total_tokens", 0)
                    if total:
                        yield {"type": "usage", "tokens": total}
                    continue

                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                yield {"type": "raw_delta", "delta": delta}

        except asyncio.CancelledError:
            return
        except Exception as exc:
            yield {"type": "error", "text": f"SSE parse error: {exc}"}
