"""OpenAI Responses API (/v1/responses) adapter.

Converts between the OpenAI Responses format and the internal OpenAI
chat/completions format so the existing GLM pipeline can be reused.
"""

from __future__ import annotations

import json
import time
import uuid


def _safe_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Request conversion: Responses -> OpenAI chat/completions
# ---------------------------------------------------------------------------


def responses_to_openai(payload: dict[str, object]) -> dict[str, object]:
    """Convert an OpenAI Responses API request body to chat/completions format."""
    messages: list[dict[str, object]] = []

    # --- instructions -> system ---
    instructions = payload.get("instructions")
    if instructions and isinstance(instructions, str):
        messages.append({"role": "system", "content": instructions})

    # --- input ---
    input_data = payload.get("input")
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_role = str(item.get("role", "user"))

            if item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    messages.append({"role": item_role, "content": content})
                elif isinstance(content, list):
                    openai_parts: list[dict[str, object]] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        part_type = part.get("type")
                        if part_type == "input_text":
                            openai_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part_type == "input_image":
                            image_url = part.get("image_url")
                            if image_url:
                                openai_parts.append({"type": "image_url", "image_url": {"url": str(image_url)}})
                        elif part_type == "output_text":
                            openai_parts.append({"type": "text", "text": part.get("text", "")})
                    if len(openai_parts) == 1 and openai_parts[0].get("type") == "text":
                        messages.append({"role": item_role, "content": openai_parts[0].get("text", "")})
                    elif openai_parts:
                        messages.append({"role": item_role, "content": openai_parts})

            elif item_type == "function_call_output":
                call_id = str(item.get("call_id", ""))
                tool_name = ""
                for prev_msg in reversed(messages):
                    if prev_msg.get("role") == "assistant" and isinstance(prev_msg.get("tool_calls"), list):
                        for tc in prev_msg["tool_calls"]:  # type: ignore[union-attr]
                            if isinstance(tc, dict) and tc.get("id") == call_id:
                                fn = tc.get("function", {})
                                if isinstance(fn, dict):
                                    tool_name = str(fn.get("name", ""))
                                break
                        if tool_name:
                            break
                msg: dict[str, object] = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(item.get("output", "")),
                }
                if tool_name:
                    msg["name"] = tool_name
                messages.append(msg)

            elif item_type == "function_call":
                try:
                    args = json.dumps(item.get("arguments", {}), ensure_ascii=False, separators=(",", ":")) \
                        if not isinstance(item.get("arguments"), str) else str(item.get("arguments", "{}"))
                except (TypeError, ValueError):
                    args = "{}"
                tc_entry = {
                    "id": str(item.get("call_id", f"call_{uuid.uuid4().hex[:24]}")),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name", "")),
                        "arguments": args,
                    },
                }
                if messages and messages[-1].get("role") == "assistant" and isinstance(messages[-1].get("tool_calls"), list):
                    messages[-1]["tool_calls"].append(tc_entry)  # type: ignore[union-attr]
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc_entry],
                    })

    # --- build output ---
    result: dict[str, object] = {
        "model": payload.get("model", "glm-4"),
        "messages": messages,
        "stream": payload.get("stream", False),
    }
    if payload.get("max_output_tokens") is not None:
        result["max_tokens"] = payload["max_output_tokens"]
    if payload.get("temperature") is not None:
        result["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        result["top_p"] = payload["top_p"]

    # --- tools ---
    resp_tools = payload.get("tools")
    if isinstance(resp_tools, list) and resp_tools:
        openai_tools = []
        for tool in resp_tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                })
        if openai_tools:
            result["tools"] = openai_tools
    if payload.get("tool_choice") is not None:
        result["tool_choice"] = payload["tool_choice"]

    # --- reasoning ---
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort:
            result["reasoning_effort"] = effort

    return result


# ---------------------------------------------------------------------------
# Non-streaming response: OpenAI -> Responses
# ---------------------------------------------------------------------------


def openai_to_responses(result: dict[str, object], model: str) -> dict[str, object]:
    """Convert an OpenAI chat/completions response to Responses API format."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    output: list[dict[str, object]] = []
    status = "completed"

    choices = result.get("choices", [])
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
            if isinstance(message, dict):
                # Build output message item
                msg_content: list[dict[str, object]] = []
                reasoning = message.get("reasoning_content")

                # Skip reasoning_content — Responses API has no thinking block,
                # and leaking it into visible text causes garbled output.

                text = message.get("content")
                if text:
                    msg_content.append({
                        "type": "output_text",
                        "text": str(text),
                        "annotations": [],
                    })

                if msg_content:
                    output.append({
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "status": "completed",
                        "role": "assistant",
                        "content": msg_content,
                    })

                # Tool calls -> function_call items
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        try:
                            args_str = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"  # type: ignore
                        except (TypeError, ValueError):
                            args_str = "{}"
                        output.append({
                            "type": "function_call",
                            "id": f"fc_{uuid.uuid4().hex[:24]}",
                            "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                            "name": fn.get("name", "") if isinstance(fn, dict) else "",
                            "arguments": str(args_str),
                            "status": "completed",
                        })

            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                status = "incomplete"

    usage = result.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
    output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE -> Responses SSE
# ---------------------------------------------------------------------------


class ResponsesStreamAccumulator:
    """Converts OpenAI chat/completions streaming chunks into Responses SSE events."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.started = False
        self.output_index = 0
        self.content_index = 0
        self.current_type: str | None = None  # "text" or "function_call"
        self.input_tokens = 0
        self.output_tokens = 0
        self._text_buffer = ""
        self._full_text = ""  # accumulated full text for message done event
        self._current_msg_id: str | None = None
        self._current_fc_id: str | None = None
        self._pending_tool_calls: dict[int, dict[str, str]] = {}
        self._message_started = False
        self._content_part_started = False
        self._completed_output: list[dict[str, object]] = []
        self._finished = False

    def _base_response(self, status: str = "in_progress") -> dict[str, object]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": list(self._completed_output),
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        }

    def start_response(self) -> list[str]:
        self.started = True
        events: list[str] = []
        events.append(self._sse("response.created", self._base_response()))
        events.append(self._sse("response.in_progress", self._base_response()))
        return events

    def feed_chunk(self, chunk: bytes) -> list[str]:
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
            events.extend(self.start_response())

        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
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

        # Skip reasoning_content — Responses API has no thinking block
        content = delta.get("content")
        text_delta = str(content) if content else ""

        if text_delta:
            if not self._message_started:
                events.extend(self._start_message_output())
            if not self._content_part_started:
                events.extend(self._start_content_part())
            events.append(self._sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": self._current_msg_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": text_delta,
            }))
            self._text_buffer += text_delta
            self._full_text += text_delta

        # Tool calls
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
                    if self._content_part_started:
                        events.extend(self._end_content_part())
                    if self._message_started:
                        events.extend(self._end_message_output())

                    fc_id = f"fc_{uuid.uuid4().hex[:24]}"
                    call_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
                    tool_name = fn.get("name", "")
                    self._pending_tool_calls[tc_index] = {
                        "id": fc_id, "call_id": str(call_id), "name": str(tool_name),
                        "arguments": "", "output_index": self.output_index,
                    }
                    self._current_fc_id = fc_id
                    self.current_type = "function_call"

                    fc_item: dict[str, object] = {
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": str(call_id),
                        "name": str(tool_name),
                        "arguments": "",
                        "status": "in_progress",
                    }
                    events.append(self._sse("response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": self.output_index,
                        "item": fc_item,
                    }))
                    self.output_index += 1

                args_delta = fn.get("arguments", "")
                if args_delta:
                    tc_data = self._pending_tool_calls[tc_index]
                    tc_data["arguments"] += str(args_delta)
                    events.append(self._sse("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": tc_data["id"],
                        "output_index": tc_data["output_index"],
                        "delta": str(args_delta),
                    }))

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            for tc_idx, tc_data in self._pending_tool_calls.items():
                tc_out_idx = tc_data["output_index"]
                events.append(self._sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": tc_data["id"],
                    "output_index": tc_out_idx,
                    "arguments": tc_data["arguments"],
                }))
                fc_done: dict[str, object] = {
                    "type": "function_call",
                    "id": tc_data["id"],
                    "call_id": tc_data["call_id"],
                    "name": tc_data["name"],
                    "arguments": tc_data["arguments"],
                    "status": "completed",
                }
                self._completed_output.append(fc_done)
                events.append(self._sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": tc_out_idx,
                    "item": fc_done,
                }))
            self._pending_tool_calls.clear()

        usage = data.get("usage")
        if isinstance(usage, dict):
            self.input_tokens = usage.get("prompt_tokens", self.input_tokens)  # type: ignore
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)  # type: ignore

        return events

    def _start_message_output(self) -> list[str]:
        self._current_msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._message_started = True
        self.current_type = "text"
        msg_item: dict[str, object] = {
            "type": "message",
            "id": self._current_msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return [self._sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": self.output_index,
            "item": msg_item,
        })]

    def _start_content_part(self) -> list[str]:
        self._content_part_started = True
        part: dict[str, object] = {"type": "output_text", "text": "", "annotations": []}
        return [self._sse("response.content_part.added", {
            "type": "response.content_part.added",
            "item_id": self._current_msg_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": part,
        })]

    def _end_content_part(self) -> list[str]:
        events: list[str] = []
        events.append(self._sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": self._current_msg_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "text": self._text_buffer,
        }))
        events.append(self._sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": self._current_msg_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": {"type": "output_text", "text": self._text_buffer, "annotations": []},
        }))
        self._content_part_started = False
        self.content_index += 1
        self._text_buffer = ""
        return events

    def _end_message_output(self) -> list[str]:
        events: list[str] = []
        msg_done: dict[str, object] = {
            "type": "message",
            "id": self._current_msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": self._full_text, "annotations": []}] if self._full_text else [],
        }
        self._completed_output.append(msg_done)
        events.append(self._sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": self.output_index,
            "item": msg_done,
        }))
        self._message_started = False
        self.output_index += 1
        self.content_index = 0
        return events

    def _finish(self) -> list[str]:
        if self._finished:
            return []
        self._finished = True
        events: list[str] = []
        if self._content_part_started:
            events.extend(self._end_content_part())
        if self._message_started:
            events.extend(self._end_message_output())

        for tc_idx, tc_data in self._pending_tool_calls.items():
            tc_out_idx = tc_data["output_index"]
            events.append(self._sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": tc_data["id"],
                "output_index": tc_out_idx,
                "arguments": tc_data["arguments"],
            }))
            fc_done: dict[str, object] = {
                "type": "function_call",
                "id": tc_data["id"],
                "call_id": tc_data["call_id"],
                "name": tc_data["name"],
                "arguments": tc_data["arguments"],
                "status": "completed",
            }
            self._completed_output.append(fc_done)
            events.append(self._sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": tc_out_idx,
                "item": fc_done,
            }))
        self._pending_tool_calls.clear()

        events.append(self._sse("response.completed", self._base_response("completed")))
        return events

    def _sse(self, event_type: str, data: dict[str, object]) -> str:
        return f"event: {event_type}\ndata: {_safe_json(data)}\n\n"
