from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
START_TAG_PATTERN = re.compile(
    r"<(?P<tag>tool_calls|tool_call|function_call|tool_use|invoke|antml:function_call|antml:invoke)\b(?P<attrs>[^>]*)/?>",
    re.IGNORECASE,
)
TOOL_RESULT_PATTERN = re.compile(r"<tool_result\b[\s\S]*?</tool_result>", re.IGNORECASE)
TAG_NAME_HINTS = [
    "<tool",
    "</tool",
    "<function_call",
    "</function_call",
    "<invoke",
    "</invoke",
    "<antml:",
    "</antml:",
]
TAIL_GUARD = 96


def _local_name(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    if tag.startswith("antml_"):
        tag = tag.split("_", 1)[1]
    return tag.lower()


def _sanitize_xml_markup(block: str) -> str:
    return block.replace("antml:", "antml_")


def _balanced_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _leaf_text(element: ET.Element) -> str:
    return _balanced_text("".join(element.itertext()))


def _maybe_parse_json_object(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped[0] not in "{[":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_json_tool_payload(text: str) -> tuple[str, dict[str, object]] | None:
    parsed = _maybe_parse_json_object(text)
    if not parsed:
        return None
    name = str(parsed.get("tool") or parsed.get("name") or parsed.get("tool_name") or "").strip()
    if not name:
        return None
    arguments = parsed.get("params")
    if arguments is None:
        arguments = parsed.get("parameters")
    if arguments is None:
        arguments = parsed.get("arguments")
    if arguments is None:
        arguments = parsed.get("input")
    if isinstance(arguments, dict):
        return name, arguments
    return name, {}


def _coerce_leaf_value(text: str) -> object:
    stripped = text.strip()
    if stripped == "":
        return ""
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    if stripped in {"true", "false"}:
        return stripped == "true"
    if stripped == "null":
        return None
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return stripped


def _append_value(mapping: dict[str, object], key: str, value: object) -> None:
    if key not in mapping:
        mapping[key] = value
        return
    existing = mapping[key]
    if isinstance(existing, list):
        existing.append(value)
        return
    mapping[key] = [existing, value]


def _xml_value_to_object(element: ET.Element) -> object:
    children = [child for child in list(element) if isinstance(child.tag, str)]
    if not children:
        return _coerce_leaf_value(_leaf_text(element))

    named_children = all(_local_name(child.tag) in {"parameter", "argument"} and child.attrib.get("name") for child in children)
    if named_children:
        result: dict[str, object] = {}
        for child in children:
            _append_value(result, child.attrib["name"].strip(), _xml_value_to_object(child))
        return result

    result: dict[str, object] = {}
    for child in children:
        key = child.attrib.get("name", "").strip() or _local_name(child.tag)
        _append_value(result, key, _xml_value_to_object(child))
    return result


def _extract_name(element: ET.Element) -> str:
    attr_name = element.attrib.get("name", "").strip()
    if attr_name:
        return attr_name
    for tag_name in ("tool_name", "name"):
        child = element.find(tag_name)
        if child is not None:
            text = _leaf_text(child)
            if text:
                return text
    if _local_name(element.tag) == "tool_use":
        function_child = element.find("function")
        if function_child is not None:
            nested_name = function_child.attrib.get("name", "").strip()
            if nested_name:
                return nested_name
    if _local_name(element.tag) == "function_call":
        text_name = _leaf_text(element)
        if text_name and not _maybe_parse_json_object(text_name):
            return text_name
    json_payload = _extract_json_tool_payload(_leaf_text(element))
    if json_payload is not None:
        return json_payload[0]
    return ""


def _extract_arguments(element: ET.Element) -> dict[str, object]:
    direct_text = _leaf_text(element)
    json_tool_payload = _extract_json_tool_payload(direct_text)
    if json_tool_payload is not None:
        return json_tool_payload[1]
    parsed_text = _maybe_parse_json_object(direct_text)
    if parsed_text is not None:
        return parsed_text

    for tag_name in ("parameters", "arguments", "input"):
        child = element.find(tag_name)
        if child is not None:
            value = _xml_value_to_object(child)
            if isinstance(value, dict):
                return value
    if _local_name(element.tag) == "tool_use":
        function_child = element.find("function")
        if function_child is not None:
            nested_value = _xml_value_to_object(function_child)
            if isinstance(nested_value, dict):
                nested_value.pop("name", None)
                return nested_value

    parameter_like_children = [
        child
        for child in list(element)
        if isinstance(child.tag, str) and _local_name(child.tag) in {"parameter", "argument"} and child.attrib.get("name")
    ]
    if parameter_like_children:
        result: dict[str, object] = {}
        for child in parameter_like_children:
            _append_value(result, child.attrib["name"].strip(), _xml_value_to_object(child))
        return result

    result: dict[str, object] = {}
    for child in list(element):
        if not isinstance(child.tag, str):
            continue
        key = _local_name(child.tag)
        if key in {"tool_name", "name"}:
            continue
        value = _xml_value_to_object(child)
        _append_value(result, key, value)
    return result


def _build_tool_call(name: str, arguments: dict[str, object], index: int) -> dict[str, object]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "index": index,
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _parse_xml_block(block: str, allowed_tool_names: set[str] | None, start_index: int) -> tuple[list[dict[str, object]], tuple[int, int] | None]:
    try:
        root = ET.fromstring(_sanitize_xml_markup(block))
    except ET.ParseError:
        return [], None

    candidates: list[ET.Element]
    root_name = _local_name(root.tag)
    if root_name == "tool_calls":
        candidates = [child for child in list(root) if isinstance(child.tag, str)]
    else:
        candidates = [root]

    tool_calls: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_name = _local_name(candidate.tag)
        if candidate_name not in {"tool_call", "function_call", "tool_use", "invoke", "function_call"}:
            continue
        tool_name = _extract_name(candidate)
        if not tool_name:
            continue
        if allowed_tool_names is not None and tool_name not in allowed_tool_names:
            continue
        arguments = _extract_arguments(candidate)
        tool_calls.append(_build_tool_call(tool_name, arguments, len(tool_calls)))

    if not tool_calls:
        return [], None
    return tool_calls, (start_index, start_index + len(block))


def _mask_code_fences(text: str) -> str:
    masked = list(text)
    for match in CODE_FENCE_PATTERN.finditer(text):
        for index in range(match.start(), match.end()):
            masked[index] = " "
    return "".join(masked)


def _find_matching_block(masked_text: str, start_match: re.Match[str]) -> tuple[int, int] | None:
    tag_name = start_match.group("tag")
    raw_text = start_match.group(0)
    if raw_text.rstrip().endswith("/>"):
        return start_match.start(), start_match.end()

    if tag_name.lower() == "invoke":
        closing_pattern = re.compile(r"</invoke\s*>", re.IGNORECASE)
    elif tag_name.lower() == "antml:invoke":
        closing_pattern = re.compile(r"</antml:invoke\s*>", re.IGNORECASE)
    else:
        closing_pattern = re.compile(rf"</{re.escape(tag_name)}\s*>", re.IGNORECASE)

    closing_match = closing_pattern.search(masked_text, start_match.end())
    if closing_match is None:
        return None
    return start_match.start(), closing_match.end()


def _extract_tool_blocks(text: str, allowed_tool_names: set[str] | None) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    masked_text = _mask_code_fences(text)
    spans: list[tuple[int, int]] = []
    tool_calls: list[dict[str, object]] = []
    cursor = 0

    while cursor < len(masked_text):
        match = START_TAG_PATTERN.search(masked_text, cursor)
        if match is None:
            break
        span = _find_matching_block(masked_text, match)
        if span is None:
            break

        start, end = span
        block_calls, parsed_span = _parse_xml_block(text[start:end], allowed_tool_names, start)
        if parsed_span is not None and block_calls:
            for offset, tool_call in enumerate(block_calls, start=len(tool_calls)):
                tool_call["index"] = offset
            spans.append(parsed_span)
            tool_calls.extend(block_calls)
            cursor = end
            continue

        cursor = match.end()

    return spans, tool_calls


FUNCTION_PARAMETER_BLOCK_PATTERN = re.compile(
    r"(?P<prefix><function_call>\s*(?P<name>[^<\s][\s\S]*?)\s*</function_call>)"
    r"(?P<params>(?:\s*<function\s+parameter\s+name=\"(?P<param_name>[^\"]+)\">[\s\S]*?</function\s+parameter>)+)",
    re.IGNORECASE,
)
FUNCTION_PARAMETER_ITEM_PATTERN = re.compile(
    r"<function\s+parameter\s+name=\"(?P<name>[^\"]+)\">(?P<value>[\s\S]*?)</function\s+parameter>",
    re.IGNORECASE,
)


def _parse_function_parameter_markup(text: str, allowed_tool_names: set[str] | None) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    spans: list[tuple[int, int]] = []
    tool_calls: list[dict[str, object]] = []
    masked = _mask_code_fences(text)
    for match in FUNCTION_PARAMETER_BLOCK_PATTERN.finditer(masked):
        original = text[match.start():match.end()]
        if "```" in original:
            continue
        tool_name = _balanced_text(match.group("name"))
        if not tool_name:
            continue
        if allowed_tool_names is not None and tool_name not in allowed_tool_names:
            continue
        arguments: dict[str, object] = {}
        for item in FUNCTION_PARAMETER_ITEM_PATTERN.finditer(text[match.start():match.end()]):
            arguments[item.group("name").strip()] = _coerce_leaf_value(item.group("value"))
        spans.append((match.start(), match.end()))
        tool_calls.append(_build_tool_call(tool_name, arguments, len(tool_calls)))
    return spans, tool_calls


def _merge_tool_parse_results(
    primary_spans: list[tuple[int, int]],
    primary_calls: list[dict[str, object]],
    extra_spans: list[tuple[int, int]],
    extra_calls: list[dict[str, object]],
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    spans = list(primary_spans)
    calls = list(primary_calls)
    for span, call in zip(extra_spans, extra_calls):
        start, end = span
        if any(not (end <= existing_start or start >= existing_end) for existing_start, existing_end in spans):
            continue
        call["index"] = len(calls)
        spans.append(span)
        calls.append(call)
    spans.sort(key=lambda item: item[0])
    return spans, calls


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    cleaned = "".join(parts)
    cleaned = TOOL_RESULT_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _find_unmatched_fence_start(text: str) -> int | None:
    last_open = None
    cursor = 0
    while True:
        index = text.find("```", cursor)
        if index == -1:
            break
        if last_open is None:
            last_open = index
        else:
            last_open = None
        cursor = index + 3
    return last_open


def _find_incomplete_block_start(text: str) -> int | None:
    masked_text = _mask_code_fences(text)
    cursor = 0
    while cursor < len(masked_text):
        match = START_TAG_PATTERN.search(masked_text, cursor)
        if match is None:
            break
        span = _find_matching_block(masked_text, match)
        if span is None:
            return match.start()
        cursor = span[1]
    return None


def _find_partial_tag_start(text: str) -> int | None:
    for hint in TAG_NAME_HINTS:
        max_overlap = min(len(hint) - 1, len(text))
        for size in range(max_overlap, 0, -1):
            if text.endswith(hint[:size]):
                return len(text) - size
    return None


def _split_stream_text(
    text: str,
    allowed_tool_names: set[str] | None,
    final: bool,
) -> tuple[str, str, list[dict[str, object]]]:
    hold_from_candidates = [index for index in (
        _find_unmatched_fence_start(text),
        _find_incomplete_block_start(text),
    ) if index is not None]

    if not final:
        partial_start = _find_partial_tag_start(text)
        if partial_start is not None:
            hold_from_candidates.append(partial_start)

    if final:
        safe_end = min(hold_from_candidates) if hold_from_candidates else len(text)
    elif hold_from_candidates:
        safe_end = min(hold_from_candidates)
    else:
        safe_end = len(text)

    processable = text[:safe_end]
    remainder = text[safe_end:]
    fallback_spans, fallback_calls = _parse_function_parameter_markup(processable, allowed_tool_names)
    spans, tool_calls = _extract_tool_blocks(processable, allowed_tool_names)
    spans, tool_calls = _merge_tool_parse_results(fallback_spans, fallback_calls, spans, tool_calls)
    visible = _remove_spans(processable, spans)
    return visible, remainder, tool_calls


def parse_tool_calls_from_text(text: str, allowed_tool_names: set[str] | None = None) -> tuple[str, list[dict[str, object]]]:
    if not text:
        return "", []
    fallback_spans, fallback_calls = _parse_function_parameter_markup(text, allowed_tool_names)
    spans, tool_calls = _extract_tool_blocks(text, allowed_tool_names)
    spans, tool_calls = _merge_tool_parse_results(fallback_spans, fallback_calls, spans, tool_calls)
    return _remove_spans(text, spans), tool_calls


@dataclass
class StreamingToolParser:
    pending_text: str = ""
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    allowed_tool_names: set[str] | None = None

    def consume(self, chunk: str) -> str:
        if not chunk:
            return ""
        self.pending_text += chunk
        visible, remainder, parsed_calls = _split_stream_text(
            self.pending_text,
            allowed_tool_names=self.allowed_tool_names,
            final=False,
        )
        self.pending_text = remainder
        self.tool_calls.extend(parsed_calls)
        return visible

    def flush(self) -> tuple[str, list[dict[str, object]]]:
        visible, remainder, parsed_calls = _split_stream_text(
            self.pending_text,
            allowed_tool_names=self.allowed_tool_names,
            final=True,
        )
        self.pending_text = ""
        self.tool_calls.extend(parsed_calls)
        return (visible + remainder).strip(), self.tool_calls
