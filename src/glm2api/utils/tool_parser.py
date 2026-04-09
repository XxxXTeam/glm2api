from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field


FUNCTION_BLOCK_PATTERN = re.compile(r"\[function_calls\](.*?)\[/function_calls\]", re.DOTALL)
CALL_PATTERN = re.compile(r"\[call:([^\]]+)\](\{.*?\})\[/call\]", re.DOTALL)
START_MARKER = "[function_calls]"
END_MARKER = "[/function_calls]"


def build_tool_calls(block_content: str) -> list[dict[str, object]]:
    tool_calls: list[dict[str, object]] = []
    for index, match in enumerate(CALL_PATTERN.finditer(block_content)):
        function_name = match.group(1).strip()
        raw_arguments = match.group(2).strip()
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
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
    return tool_calls


def parse_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, object]]]:
    match = FUNCTION_BLOCK_PATTERN.search(text)
    if not match:
        return text, []
    before = text[: match.start()].rstrip()
    return before, build_tool_calls(match.group(1))


@dataclass
class StreamingToolParser:
    plain_text_tail: str = ""
    buffered_tool_text: str = ""
    in_tool_block: bool = False
    tool_calls: list[dict[str, object]] = field(default_factory=list)

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
                self.tool_calls = build_tool_calls(self.buffered_tool_text[:end_index])
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
                self.tool_calls = build_tool_calls(self.buffered_tool_text[:end_index])
            self.buffered_tool_text = ""
            self.in_tool_block = False
        return remaining, self.tool_calls
