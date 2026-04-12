from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field


START_MARKER = "[function_calls]"
END_MARKER = "[/function_calls]"
CALL_PREFIX = "[call:"


def _extract_balanced_json_object(text: str, start_index: int) -> tuple[str | None, int]:
    if start_index >= len(text) or text[start_index] != "{":
        return None, start_index

    depth = 0
    in_string = False
    escaped = False

    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1], index + 1

    return None, start_index


def build_tool_calls(block_content: str, allowed_tool_names: set[str] | None = None) -> list[dict[str, object]]:
    tool_calls: list[dict[str, object]] = []
    cursor = 0
    index = 0

    while True:
        call_start = block_content.find(CALL_PREFIX, cursor)
        if call_start == -1:
            break

        name_end = block_content.find("]", call_start + len(CALL_PREFIX))
        if name_end == -1:
            break
        function_name = block_content[call_start + len(CALL_PREFIX) : name_end].strip()
        if not function_name:
            cursor = name_end + 1
            continue
        if allowed_tool_names is not None and function_name not in allowed_tool_names:
            cursor = name_end + 1
            continue

        json_start = name_end + 1
        while json_start < len(block_content) and block_content[json_start].isspace():
            json_start += 1

        raw_arguments, json_end = _extract_balanced_json_object(block_content, json_start)
        if raw_arguments is None:
            cursor = name_end + 1
            continue

        closing_tag = "[/call]"
        closing_index = block_content.find(closing_tag, json_end)
        if closing_index == -1:
            cursor = json_end
            continue

        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            cursor = closing_index + len(closing_tag)
            continue
        if not isinstance(parsed, dict):
            cursor = closing_index + len(closing_tag)
            continue

        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "index": index,
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
        index += 1
        cursor = closing_index + len(closing_tag)

    return tool_calls


def parse_tool_calls_from_text(text: str, allowed_tool_names: set[str] | None = None) -> tuple[str, list[dict[str, object]]]:
    if not text:
        return "", []

    remaining_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    cursor = 0

    while True:
        start_index = text.find(START_MARKER, cursor)
        if start_index == -1:
            remaining_parts.append(text[cursor:])
            break

        remaining_parts.append(text[cursor:start_index])
        end_index = text.find(END_MARKER, start_index + len(START_MARKER))
        if end_index == -1:
            remaining_parts.append(text[start_index:])
            break

        block_content = text[start_index + len(START_MARKER) : end_index]
        tool_calls.extend(build_tool_calls(block_content, allowed_tool_names=allowed_tool_names))
        cursor = end_index + len(END_MARKER)

    cleaned = "".join(remaining_parts).strip()
    return cleaned, tool_calls


@dataclass
class StreamingToolParser:
    plain_text_tail: str = ""
    buffered_tool_text: str = ""
    in_tool_block: bool = False
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    allowed_tool_names: set[str] | None = None

    def consume(self, chunk: str) -> str:
        if not chunk:
            return ""

        visible: list[str] = []
        pending = chunk

        while pending:
            if self.in_tool_block:
                self.buffered_tool_text += pending
                end_index = self.buffered_tool_text.find(END_MARKER)
                if end_index == -1:
                    return "".join(visible)
                block_content = self.buffered_tool_text[:end_index]
                self.tool_calls.extend(build_tool_calls(block_content, allowed_tool_names=self.allowed_tool_names))
                pending = self.buffered_tool_text[end_index + len(END_MARKER) :]
                self.buffered_tool_text = ""
                self.in_tool_block = False
                continue

            combined = self.plain_text_tail + pending
            start_index = combined.find(START_MARKER)
            if start_index == -1:
                safe_length = max(0, len(combined) - len(START_MARKER) + 1)
                visible.append(combined[:safe_length])
                self.plain_text_tail = combined[safe_length:]
                return "".join(visible)

            visible.append(combined[:start_index])
            pending = combined[start_index + len(START_MARKER) :]
            self.plain_text_tail = ""
            self.in_tool_block = True

        return "".join(visible)

    def flush(self) -> tuple[str, list[dict[str, object]]]:
        remaining = self.plain_text_tail
        self.plain_text_tail = ""
        if self.in_tool_block and self.buffered_tool_text:
            end_index = self.buffered_tool_text.find(END_MARKER)
            if end_index != -1:
                self.tool_calls.extend(build_tool_calls(self.buffered_tool_text[:end_index], allowed_tool_names=self.allowed_tool_names))
            else:
                remaining += START_MARKER + self.buffered_tool_text
            self.buffered_tool_text = ""
            self.in_tool_block = False
        return remaining, self.tool_calls
