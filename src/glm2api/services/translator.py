from __future__ import annotations

import json
import logging
import re
import time
from bisect import insort
from dataclasses import dataclass, field
from logging import Logger

from ..config import AppConfig
from ..logging_utils import debug_dump
from ..utils.tool_parser import StreamingToolParser, parse_tool_calls_from_text


ASSISTANT_ID_PATTERN = re.compile(r"^[a-z0-9]{24,}$")


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
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
    tool_names: list[str] = []
    tool_defs: list[str] = []
    for tool in tools:
        fn = tool.get("function", {})
        name = str(fn.get("name", "unknown")) # type: ignore
        tool_names.append(name)
        description = fn.get("description", "") # type: ignore
        parameters = fn.get("parameters", {}) # type: ignore
        required = parameters.get("required", []) if isinstance(parameters, dict) else [] # type: ignore
        props = parameters.get("properties", {}) if isinstance(parameters, dict) else {} # type: ignore
        param_parts: list[str] = []
        if isinstance(props, dict):
            for pname, pdef in props.items():
                ptype = pdef.get("type", "string") if isinstance(pdef, dict) else "string"
                pdesc = pdef.get("description", "") if isinstance(pdef, dict) else ""
                req_mark = " (required)" if pname in required else ""
                param_parts.append(f"  - {pname}: {ptype}{req_mark}" + (f" — {pdesc}" if pdesc else ""))
        desc_line = f": {description}" if description else ""
        param_block = "\n".join(param_parts) if param_parts else "  (no parameters)"
        tool_defs.append(f"- `{name}`{desc_line}\n  Parameters:\n{param_block}")

    names_str = ", ".join(f"`{n}`" for n in tool_names)

    lines = [
        f"# TOOLS\n",
        f"You have EXACTLY these tools: {names_str}.",
        "You MUST NOT invent tool names. ONLY use names from this list.\n",
    ]
    lines.extend(tool_defs)
    lines.extend([
        "",
        "# HOW TO CALL",
        "To call tools, output EXACTLY this format and NOTHING ELSE in the same turn:",
        "",
        "[function_calls]",
        '[call:TOOL_NAME]{"param":"value"}[/call]',
        "[/function_calls]",
        "",
        "Multiple calls in one turn:",
        "",
        "[function_calls]",
        '[call:tool1]{"a":"1"}[/call]',
        '[call:tool2]{"b":"2"}[/call]',
        "[/function_calls]",
        "",
        "ABSOLUTE RULES:",
        "- Output ONLY the [function_calls] block when calling tools. NO text before or after.",
        "- NEVER invent tools that are not in the list above.",
        "- NEVER output tool names like open_url, browser, fetch, etc. unless they are in the list.",
        "- NEVER say 'I cannot call tools' — you CAN. Use the [function_calls] format.",
        "- NEVER suggest the user run commands manually if you have a tool that can do it.",
        "- JSON arguments must be on ONE line, valid JSON.",
        "- After receiving [TOOL_RESULT]...[/TOOL_RESULT], continue using results. Call more tools if needed.",
    ])
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
            assistant_text = extract_text_content(content).strip() if content else ""
            block = "[function_calls]\n" + "\n".join(tool_calls) + "\n[/function_calls]"
            content = f"{assistant_text}\n{block}".strip() if assistant_text else block
        elif role == "tool":
            role = "user"
            tool_name = str(message.get("name", "")).strip() or "unknown_tool"
            tool_result_text = extract_text_content(content)
            content = (
                f"[TOOL_RESULT call_id={message.get('tool_call_id', 'unknown')} name={tool_name}]\n"
                f"{tool_result_text}\n"
                f"[/TOOL_RESULT]"
            )
        elif role == "assistant" and not content:
            continue

        text = extract_text_content(content) if content else ""
        if text:
            processed.append({"role": role, "content": text})

    transcript_parts: list[str] = []

    if tools:
        transcript_parts.append(tools_to_prompt(tools))
        transcript_parts.append("# CONVERSATION\n")

    for item in processed:
        title = (
            item["role"]
            .replace("system", "System")
            .replace("assistant", "Assistant")
            .replace("user", "User")
            .replace("developer", "Developer")
        )
        transcript_parts.append(f"{title}: {item['content']}".strip())

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
    debug_enabled: bool = False
    logger: Logger | None = None
    conversation_id: str = ""
    created: int = field(default_factory=lambda: int(time.time()))
    parts_by_logic_id: dict[str, dict[str, object]] = field(default_factory=dict)
    ordered_logic_ids: list[str] = field(default_factory=list)
    last_full_text: str = ""
    last_full_reasoning: str = ""
    _part_text_sent: dict[str, int] = field(default_factory=dict)
    _part_reasoning_sent: dict[str, int] = field(default_factory=dict)
    _known_logic_ids_for_text: list[str] = field(default_factory=list)
    _known_logic_ids_for_reasoning: list[str] = field(default_factory=list)
    tool_parser: StreamingToolParser = field(default_factory=StreamingToolParser)
    emitted_role: bool = False
    _render_cache_dirty: bool = True
    _cached_full_text: str = ""
    _cached_full_reasoning: str = ""
    _cached_part_texts: dict[str, str] = field(default_factory=dict)
    _cached_part_reasonings: dict[str, str] = field(default_factory=dict)

    def consume_event(self, payload: dict[str, object]) -> tuple[list[str], str | None]:
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE 解析事件", payload)
        if not self.conversation_id and payload.get("conversation_id"):
            self.conversation_id = str(payload["conversation_id"])

        for part in payload.get("parts", []) if isinstance(payload.get("parts"), list) else []: # pyright: ignore[reportGeneralTypeIssues]
            if isinstance(part, dict) and part.get("logic_id"):
                logic_id = str(part["logic_id"])
                if logic_id not in self.parts_by_logic_id:
                    insort(self.ordered_logic_ids, logic_id)
                self.parts_by_logic_id[logic_id] = part
                self._render_cache_dirty = True

        text_delta, reasoning_delta = self._compute_deltas()
        full_text, full_reasoning = self._cached_full_text, self._cached_full_reasoning
        self.last_full_text = full_text
        self.last_full_reasoning = full_reasoning

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
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE 生成增量块", chunks)
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
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE finalize 输出", chunks)
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
        response = {
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
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM 非流式最终响应", response)
        return response

    def _compute_deltas(self) -> tuple[str, str]:
        self._render_full_output()
        text_delta_parts: list[str] = []
        reasoning_delta_parts: list[str] = []

        for logic_id in self.ordered_logic_ids:
            rendered_text = self._cached_part_texts.get(logic_id, "")
            rendered_reasoning = self._cached_part_reasonings.get(logic_id, "")

            if rendered_text:
                prev_len = self._part_text_sent.get(logic_id, 0)
                is_new = logic_id not in self._known_logic_ids_for_text
                if is_new:
                    self._known_logic_ids_for_text.append(logic_id)
                    if text_delta_parts or self._part_text_sent:
                        text_delta_parts.append("\n")
                    text_delta_parts.append(rendered_text)
                elif len(rendered_text) > prev_len:
                    text_delta_parts.append(rendered_text[prev_len:])
                self._part_text_sent[logic_id] = len(rendered_text)

            if rendered_reasoning:
                prev_len = self._part_reasoning_sent.get(logic_id, 0)
                is_new = logic_id not in self._known_logic_ids_for_reasoning
                if is_new:
                    self._known_logic_ids_for_reasoning.append(logic_id)
                    if reasoning_delta_parts or self._part_reasoning_sent:
                        reasoning_delta_parts.append("\n")
                    reasoning_delta_parts.append(rendered_reasoning)
                elif len(rendered_reasoning) > prev_len:
                    reasoning_delta_parts.append(rendered_reasoning[prev_len:])
                self._part_reasoning_sent[logic_id] = len(rendered_reasoning)

        return "".join(text_delta_parts), "".join(reasoning_delta_parts)

    def _render_full_output(self) -> tuple[str, str]:
        if not self._render_cache_dirty:
            return self._cached_full_text, self._cached_full_reasoning

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        self._cached_part_texts.clear()
        self._cached_part_reasonings.clear()
        for logic_id in self.ordered_logic_ids:
            part = self.parts_by_logic_id.get(logic_id)
            if not isinstance(part, dict):
                continue
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
                self._cached_part_texts[logic_id] = rendered_text
            if rendered_reasoning:
                reasoning_parts.append(rendered_reasoning)
                self._cached_part_reasonings[logic_id] = rendered_reasoning

        self._cached_full_text = "\n".join(text_parts)
        self._cached_full_reasoning = "\n".join(reasoning_parts)
        self._render_cache_dirty = False
        return self._cached_full_text, self._cached_full_reasoning

    def _chunk_json(self, patch: dict[str, object]) -> str:
        payload = {
            "id": self.conversation_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        payload.update(patch)
        return "data: " + safe_json_dumps(payload) + "\n\n"
