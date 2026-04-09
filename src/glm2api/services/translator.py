from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from ..config import AppConfig
from ..utils.tool_parser import StreamingToolParser, parse_tool_calls_from_text


ASSISTANT_ID_PATTERN = re.compile(r"^[a-z0-9]{24,}$")


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "image_url":
            url = item.get("image_url", {}).get("url", "")
            text_parts.append(f"[image:{url}]")
        elif item_type == "file":
            url = item.get("file_url", {}).get("url", "")
            text_parts.append(f"[file:{url}]")
    return "\n".join(part for part in text_parts if part)


def tools_to_prompt(tools: list[dict[str, object]]) -> str:
    lines = [
        "## Available Tools",
        "You can invoke the following developer tools. Call a tool only when it is required and follow the JSON schema exactly when providing arguments.",
        "",
        "CRITICAL: Tool names are CASE-SENSITIVE. You MUST use the exact tool name as defined below.",
        "",
    ]
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "unknown") # type: ignore
        description = fn.get("description", "No description") # type: ignore
        parameters = json.dumps(fn.get("parameters", {}), ensure_ascii=False, separators=(",", ":")) # type: ignore
        lines.append(f"Tool `{name}`: {description}. Arguments JSON schema: {parameters}")

    lines.extend(
        [
            "",
            "## Tool Call Protocol",
            "When you decide to call a tool, you MUST respond with NOTHING except a single [function_calls] block exactly like the template below:",
            "",
            "[function_calls]",
            '[call:exact_tool_name]{"argument":"value"}[/call]',
            "[/function_calls]",
            "",
            "GLM STRICT RULES:",
            "- If the task needs file edits, shell execution, or structured tool usage, call tools instead of narrating.",
            "- The JSON arguments must stay on one line.",
            "- Do not add any extra explanation outside the [function_calls] block.",
        ]
    )
    return "\n".join(lines)


def convert_messages(messages: list[dict[str, object]], tools: list[dict[str, object]] | None) -> list[dict[str, object]]:
    processed: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "assistant" and message.get("tool_calls"):
            tool_calls = []
            for tool_call in message.get("tool_calls", []): # pyright: ignore[reportGeneralTypeIssues]
                function = tool_call.get("function", {})
                tool_calls.append(
                    f"[call:{function.get('name', 'unknown')}]{function.get('arguments', '{}')}[/call]"
                )
            content = "[function_calls]\n" + "\n".join(tool_calls) + "\n[/function_calls]"
        elif role == "tool":
            role = "user"
            content = f"[TOOL_RESULT for {message.get('tool_call_id', 'unknown')}] {extract_text_content(content)}"

        processed.append({"role": role, "content": extract_text_content(content)})

    transcript_parts: list[str] = []
    for item in processed:
        title = item["role"].replace("system", "System").replace("assistant", "Assistant").replace("user", "User")
        transcript_parts.append(f"{title}: {item['content']}".strip())

    if tools:
        transcript_parts.append(tools_to_prompt(tools))

    prompt = "\n\n".join(part for part in transcript_parts if part).strip()
    return [{"role": "user", "content": [{"type": "text", "text": prompt + "\n\nAssistant: "}]}]


def resolve_upstream_model(requested_model: str, config: AppConfig) -> tuple[str, str]:
    upstream_model = config.model_aliases.get(requested_model, requested_model)
    assistant_id = upstream_model if ASSISTANT_ID_PATTERN.fullmatch(upstream_model) else config.glm_assistant_id
    return upstream_model, assistant_id


def resolve_chat_mode(model: str, reasoning_effort: object, deep_research: object) -> str:
    lower_model = (model or "").lower()
    if deep_research or "deepresearch" in lower_model or "deep-research" in lower_model:
        return "deep_research"
    if reasoning_effort or "think" in lower_model or "zero" in lower_model:
        return "zero"
    return ""


def safe_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass
class GLMEventAccumulator:
    model: str
    conversation_id: str = ""
    created: int = field(default_factory=lambda: int(time.time()))
    parts_by_logic_id: dict[str, dict[str, object]] = field(default_factory=dict)
    last_full_text: str = ""
    last_full_reasoning: str = ""
    visible_text_sent: int = 0
    visible_reasoning_sent: int = 0
    tool_parser: StreamingToolParser = field(default_factory=StreamingToolParser)
    emitted_role: bool = False

    def consume_event(self, payload: dict[str, object]) -> tuple[list[str], str | None]:
        if not self.conversation_id and payload.get("conversation_id"):
            self.conversation_id = str(payload["conversation_id"])

        for part in payload.get("parts", []) if isinstance(payload.get("parts"), list) else []: # pyright: ignore[reportGeneralTypeIssues]
            if isinstance(part, dict) and part.get("logic_id"):
                self.parts_by_logic_id[str(part["logic_id"])] = part

        full_text, full_reasoning = self._render_full_output()
        self.last_full_text = full_text
        self.last_full_reasoning = full_reasoning
        text_delta = full_text[self.visible_text_sent :]
        reasoning_delta = full_reasoning[self.visible_reasoning_sent :]
        self.visible_text_sent = len(full_text)
        self.visible_reasoning_sent = len(full_reasoning)

        chunks: list[str] = []
        if reasoning_delta:
            chunks.append(
                self._chunk_json(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": reasoning_delta},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        visible_text_delta = self.tool_parser.consume(text_delta)
        if visible_text_delta:
            delta_payload: dict[str, object] = {"content": visible_text_delta}
            if not self.emitted_role:
                delta_payload = {"role": "assistant", "content": visible_text_delta}
                self.emitted_role = True
            chunks.append(
                self._chunk_json(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_payload,
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )
        return chunks, str(payload.get("status")) if payload.get("status") is not None else None

    def finalize(self, status: str | None, last_error: dict[str, object] | None = None) -> list[str]:
        tail_text, tool_calls = self.tool_parser.flush()
        chunks: list[str] = []
        if tail_text:
            delta_payload: dict[str, object] = {"content": tail_text}
            if not self.emitted_role:
                delta_payload = {"role": "assistant", "content": tail_text}
                self.emitted_role = True
            chunks.append(
                self._chunk_json(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_payload,
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        if status == "intervene" and last_error and last_error.get("intervene_text"):
            chunks.append(
                self._chunk_json(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "\n\n" + str(last_error["intervene_text"])},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        if tool_calls:
            for tool_call in tool_calls:
                chunks.append(
                    self._chunk_json(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": tool_call["index"],
                                                "id": tool_call["id"],
                                                "type": "function",
                                                "function": tool_call["function"],
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )

        finish_reason = "tool_calls" if tool_calls else "stop"
        chunks.append(
            self._chunk_json(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        )
        chunks.append("data: [DONE]\n\n")
        return chunks

    def build_response(self) -> dict[str, object]:
        full_text, full_reasoning = self._render_full_output()
        if not full_text and self.last_full_text:
            full_text = self.last_full_text
        if not full_reasoning and self.last_full_reasoning:
            full_reasoning = self.last_full_reasoning
        clean_content, tool_calls = parse_tool_calls_from_text(full_text.strip())
        final_content = clean_content.strip()
        message: dict[str, object] = {
            "role": "assistant",
            "content": None if tool_calls or not final_content else final_content,
            "reasoning_content": full_reasoning or None,
        }
        if tool_calls:
            message["tool_calls"] = [
                {"id": item["id"], "type": "function", "function": item["function"]}
                for item in tool_calls
            ]
        return {
            "id": self.conversation_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def _render_full_output(self) -> tuple[str, str]:
        ordered_parts = list(self.parts_by_logic_id.values())
        ordered_parts.sort(key=lambda item: str(item.get("logic_id", "")))

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for part in ordered_parts:
            content_items = part.get("content", [])
            if not isinstance(content_items, list):
                continue

            part_text = []
            part_reasoning = []
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                item_type = content.get("type")
                if item_type == "text":
                    part_text.append(str(content.get("text", "")))
                elif item_type == "think":
                    part_reasoning.append(str(content.get("think", "")))
                elif item_type == "code":
                    part_text.append(f"```python\n{content.get('code', '')}\n```")
                elif item_type == "execution_output":
                    part_text.append(str(content.get("content", "")))
                elif item_type == "image":
                    images = content.get("image", [])
                    if isinstance(images, list):
                        for image in images:
                            if isinstance(image, dict) and image.get("image_url"):
                                part_text.append(f"![image]({image['image_url']})")

            rendered_text = "\n".join(filter(None, part_text)).strip()
            rendered_reasoning = "\n".join(filter(None, part_reasoning)).strip()
            if rendered_text:
                text_parts.append(rendered_text)
            if rendered_reasoning:
                reasoning_parts.append(rendered_reasoning)

        return "\n".join(text_parts), "\n".join(reasoning_parts)

    def _chunk_json(self, patch: dict[str, object]) -> str:
        payload = {
            "id": self.conversation_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        payload.update(patch)
        return "data: " + safe_json_dumps(payload) + "\n\n"
