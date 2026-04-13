"""
DeepSeek CLI - AI Agent

The core agent that orchestrates LLM calls with tool execution.
Implements an agentic loop:
  1. User sends query
  2. LLM responds (may include tool_calls)
  3. If tool_calls: execute each tool, feed results back to LLM
  4. LLM generates final response
  5. Yield streaming chunks to UI

Events yielded by process_query():
  {"type": "thinking", "text": "..."}   - LLM reasoning/thinking
  {"type": "content",  "text": "..."}   - LLM response text
  {"type": "tool_start", "name": "...", "args": {...}}  - tool execution starting
  {"type": "tool_result", "name": "...", "result": "..."} - tool result
  {"type": "tool_error", "name": "...", "error": "..."}  - tool error
  {"type": "usage", "tokens": N}        - token usage
  {"type": "error", "text": "..."}      - error message
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from .llm import LLMClient
from .memory import ConversationMemory
from ..tools.registry import ToolRegistry


class Agent:
    """AI Agent with agentic tool-calling loop."""

    MAX_TOOL_ROUNDS = 8  # max tool execution rounds per query

    def __init__(
        self,
        llm: LLMClient,
        memory: ConversationMemory,
        tools: ToolRegistry,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools

        self._query_tokens: int = 0
        self._query_start: float = 0.0

    # ── Main Query Processing ───────────────────────────────

    async def process_query(
        self,
        query: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user query with full agentic loop."""
        self._query_tokens = 0
        self._query_start = time.time()

        self.memory.add_user(query)

        for round_num in range(self.MAX_TOOL_ROUNDS + 1):
            messages = self.memory.get_openai_messages()
            tool_schemas = self.tools.get_openai_tools()

            content_buf = ""
            thinking_buf = ""
            tool_calls_acc: List[Dict] = []
            tokens_this_round = 0

            async for chunk in self.llm.stream(messages, tools=tool_schemas):
                ctype = chunk.get("type", "")

                if ctype == "thinking":
                    thinking_buf += chunk["text"]
                    yield {"type": "thinking", "text": chunk["text"]}

                elif ctype == "content":
                    content_buf += chunk["text"]
                    yield {"type": "content", "text": chunk["text"]}

                elif ctype == "tool_calls":
                    tool_calls_acc = chunk["tool_calls"]

                elif ctype == "usage":
                    tokens_this_round = chunk.get("tokens", 0)
                    self._query_tokens += tokens_this_round
                    yield {"type": "usage", "tokens": tokens_this_round}

                elif ctype == "error":
                    yield {"type": "error", "text": chunk["text"]}

            # ── If LLM wants to call tools, execute them ──
            if tool_calls_acc:
                self.memory.add_assistant(content=content_buf, tool_calls=tool_calls_acc)

                for tc in tool_calls_acc:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                    tc_id = tc.get("id", "")

                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except json.JSONDecodeError:
                        fn_args = {}

                    yield {"type": "tool_start", "name": fn_name, "args": fn_args}

                    result = await self._execute_tool(fn_name, fn_args)

                    if result.get("error"):
                        yield {"type": "tool_error", "name": fn_name, "error": result["error"]}
                        result_str = json.dumps(result)
                    else:
                        yield {"type": "tool_result", "name": fn_name, "result": result.get("output", "")[:6000]}
                        result_str = result.get("output", "")

                    self.memory.add_tool_result(tc_id, fn_name, result_str)

                content_buf = ""
                thinking_buf = ""
                continue

            # ── No tool calls - LLM gave final response ──
            break

        if content_buf.strip():
            self.memory.add_assistant(content=content_buf)

        latency = round(time.time() - self._query_start, 2)
        yield {"type": "usage", "tokens": self._query_tokens}
        yield {
            "type": "done",
            "tokens": self._query_tokens,
            "latency": latency,
            "rounds": round_num + 1,
        }

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments."""
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool.execute(**args),
            )
            return {"output": result}
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

    # ── Status & Lifecycle ──────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": self.llm.model,
            "tools": self.tools.count,
            "tool_names": self.tools.get_names(),
            "messages": self.memory.count,
            "total_tokens": self.llm.total_tokens_used,
            "requests": self.llm.request_count,
        }

    def clear(self) -> None:
        self.memory.clear()

    def compact(self, keep_pairs: int = 5) -> int:
        return self.memory.compact(keep_pairs)

    async def close(self) -> None:
        await self.llm.close()
