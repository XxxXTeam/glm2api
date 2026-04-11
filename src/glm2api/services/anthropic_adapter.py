"""Anthropic Messages API (/v1/messages) adapter.

Converts between Anthropic Messages format and the internal OpenAI
chat/completions format so the existing GLM pipeline can be reused.
"""

from __future__ import annotations

import json
import time
import uuid


def _safe_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI chat/completions
# ---------------------------------------------------------------------------


def anthropic_to_openai(payload: dict[str, object]) -> dict[str, object]:
    """Convert an Anthropic Messages request body to OpenAI chat/completions."""
    messages: list[dict[str, object]] = []

    # --- system ---
    system = payload.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # --- messages ---
    for msg in payload.get("messages", []):  # type: ignore[union-attr]
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": ""})
            continue

        # Process content blocks
        openai_content_parts: list[dict[str, object]] = []
        tool_calls: list[dict[str, object]] = []
        tool_results: list[dict[str, object]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")

            if block_type == "text":
                openai_content_parts.append({"type": "text", "text": block.get("text", "")})

            elif block_type == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    openai_content_parts.append({"type": "text", "text": str(thinking_text)})

            elif block_type == "image":
                source = block.get("source", {})
                if isinstance(source, dict):
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    if source.get("type") == "base64" and data:
                        openai_content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
                        })
                    elif source.get("type") == "url":
                        url = source.get("url", "")
                        if url:
                            openai_content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": str(url)},
                            })

            elif block_type == "tool_use":
                tool_calls.append({
                    "id": str(block.get("id", f"call_{uuid.uuid4().hex[:24]}")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(
                            block.get("input", {}), ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                })

            elif block_type == "tool_result":
                result_content = block.get("content")
                result_text = ""
                if isinstance(result_content, str):
                    result_text = result_content
                elif isinstance(result_content, list):
                    parts = []
                    for rc in result_content:
                        if isinstance(rc, dict) and rc.get("type") == "text":
                            parts.append(str(rc.get("text", "")))
                    result_text = "\n".join(parts)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": result_text,
                })

        if tool_results:
            for tr in tool_results:
                messages.append(tr)
        elif tool_calls:
            text_content = ""
            if openai_content_parts:
                text_content = "\n".join(
                    str(p.get("text", "")) for p in openai_content_parts if p.get("type") == "text"
                )
            msg_out: dict[str, object] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
            messages.append(msg_out)
        elif len(openai_content_parts) == 1 and openai_content_parts[0].get("type") == "text":
            messages.append({"role": role, "content": openai_content_parts[0].get("text", "")})
        elif openai_content_parts:
            messages.append({"role": role, "content": openai_content_parts})
        else:
            messages.append({"role": role, "content": ""})

    # --- build output payload ---
    result: dict[str, object] = {
        "model": payload.get("model", "glm-4"),
        "messages": messages,
        "stream": payload.get("stream", False),
    }
    if payload.get("max_tokens"):
        result["max_tokens"] = payload["max_tokens"]
    if payload.get("temperature") is not None:
        result["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        result["top_p"] = payload["top_p"]
    if payload.get("stop_sequences"):
        result["stop"] = payload["stop_sequences"]

    # --- tools ---
    anthropic_tools = payload.get("tools")
    if isinstance(anthropic_tools, list) and anthropic_tools:
        openai_tools = []
        for tool in anthropic_tools:
            if not isinstance(tool, dict):
                continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        if openai_tools:
            result["tools"] = openai_tools

    # --- thinking ---
    thinking = payload.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        result["reasoning_effort"] = thinking.get("budget_tokens", "medium")

    return result


# ---------------------------------------------------------------------------
# Non-streaming response conversion: OpenAI -> Anthropic
# ---------------------------------------------------------------------------


def openai_to_anthropic_response(result: dict[str, object], model: str) -> dict[str, object]:
    """Convert an OpenAI chat/completions response to Anthropic Messages format."""
    content: list[dict[str, object]] = []
    stop_reason = "end_turn"

    choices = result.get("choices", [])
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
            if isinstance(message, dict):
                # reasoning_content -> thinking block
                reasoning = message.get("reasoning_content")
                if reasoning:
                    content.append({
                        "type": "thinking",
                        "thinking": str(reasoning),
                    })

                # text content
                text = message.get("content")
                if text:
                    content.append({"type": "text", "text": str(text)})

                # tool_calls
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    stop_reason = "tool_use"
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        try:
                            input_data = json.loads(fn.get("arguments", "{}"))  # type: ignore[union-attr]
                        except (json.JSONDecodeError, TypeError):
                            input_data = {}
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": fn.get("name", ""),  # type: ignore[union-attr]
                            "input": input_data,
                        })

            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                stop_reason = "max_tokens"

    if not content:
        content.append({"type": "text", "text": ""})

    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
    output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------


class AnthropicStreamAccumulator:
    """Converts OpenAI chat/completions streaming chunks into Anthropic SSE events."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.started = False
        self.content_index = 0
        self.current_block_type: str | None = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self._pending_tool_calls: dict[int, dict[str, object]] = {}
        self._block_open = False
        self._finished = False

    def start_message(self) -> str:
        """Emit message_start event."""
        self.started = True
        msg = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
        }
        return self._sse("message_start", {"type": "message_start", "message": msg})

    def feed_chunk(self, chunk: bytes) -> list[str]:
        """Process a raw SSE chunk line (already decoded). Returns Anthropic SSE events."""
        text = chunk.decode("utf-8", errors="ignore")
        events: list[str] = []
        for line in text.split("\n\n"):
            line = line.strip()
            if not line:
                continue
            if line == "data: [DONE]":
                events.extend(self._finish())
                continue
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                events.extend(self._process_openai_chunk(data))
        return events

    def _process_openai_chunk(self, data: dict[str, object]) -> list[str]:
        events: list[str] = []
        if not self.started:
            events.append(self.start_message())

        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            # Usage update
            usage = data.get("usage")
            if isinstance(usage, dict):
                self.input_tokens = usage.get("prompt_tokens", self.input_tokens)  # type: ignore
                self.output_tokens = usage.get("completion_tokens", self.output_tokens)  # type: ignore
            return events

        choice = choices[0]
        if not isinstance(choice, dict):
            return events
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            return events
        finish_reason = choice.get("finish_reason")

        # reasoning_content -> thinking block
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if self.current_block_type != "thinking":
                if self._block_open:
                    events.append(self._content_block_stop())
                events.append(self._content_block_start("thinking", {}))
                self.current_block_type = "thinking"
            events.append(self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.content_index,
                "delta": {"type": "thinking_delta", "thinking": str(reasoning)},
            }))

        # text content
        content = delta.get("content")
        if content:
            if self.current_block_type != "text":
                if self._block_open:
                    events.append(self._content_block_stop())
                events.append(self._content_block_start("text", {"text": ""}))
                self.current_block_type = "text"
            events.append(self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self.content_index,
                "delta": {"type": "text_delta", "text": str(content)},
            }))

        # tool_calls
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_index = tc.get("index", 0)
                fn = tc.get("function", {})
                if not isinstance(fn, dict):
                    continue

                if tc_index not in self._pending_tool_calls:
                    # New tool call - close previous block, start new one
                    if self._block_open:
                        events.append(self._content_block_stop())
                    tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    tool_name = fn.get("name", "")
                    self._pending_tool_calls[tc_index] = {
                        "id": tool_id, "name": tool_name, "arguments": "",
                    }
                    events.append(self._content_block_start("tool_use", {
                        "id": tool_id, "name": tool_name, "input": {},
                    }))
                    self.current_block_type = "tool_use"
                    self.stop_reason = "tool_use"

                args_delta = fn.get("arguments", "")
                if args_delta:
                    self._pending_tool_calls[tc_index]["arguments"] += str(args_delta)  # type: ignore
                    events.append(self._sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self.content_index,
                        "delta": {"type": "input_json_delta", "partial_json": str(args_delta)},
                    }))

        if finish_reason:
            if finish_reason == "length":
                self.stop_reason = "max_tokens"
            elif finish_reason == "tool_calls":
                self.stop_reason = "tool_use"

        # Usage in final chunk
        usage = data.get("usage")
        if isinstance(usage, dict):
            self.input_tokens = usage.get("prompt_tokens", self.input_tokens)  # type: ignore
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)  # type: ignore

        return events

    def _finish(self) -> list[str]:
        if self._finished:
            return []
        self._finished = True
        events: list[str] = []
        if self._block_open:
            events.append(self._content_block_stop())
        events.append(self._sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens},
        }))
        events.append(self._sse("message_stop", {"type": "message_stop"}))
        return events

    def _content_block_start(self, block_type: str, initial: dict[str, object]) -> str:
        block: dict[str, object] = {"type": block_type}
        block.update(initial)
        self._block_open = True
        event = self._sse("content_block_start", {
            "type": "content_block_start",
            "index": self.content_index,
            "content_block": block,
        })
        return event

    def _content_block_stop(self) -> str:
        event = self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self.content_index,
        })
        self.content_index += 1
        self._block_open = False
        return event

    def _sse(self, event_type: str, data: dict[str, object]) -> str:
        return f"event: {event_type}\ndata: {_safe_json(data)}\n\n"
