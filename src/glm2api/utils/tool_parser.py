from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
TOOL_RESULT_PATTERN = re.compile(r"<(?:ml_)?tool_result\b[\s\S]*?</(?:ml_)?tool_result>", re.IGNORECASE)
START_TAG_PATTERN = re.compile(r"<(?P<tag>(?:ml_)?tool_calls|(?:ml_)?tool_call)\b[^>]*>", re.IGNORECASE)
PARAM_NAME_TAG_PATTERN = re.compile(r"<param_name>\s*(.*?)\s*</param_name>", re.IGNORECASE | re.DOTALL)
PARAM_VALUE_TAG_PATTERN = re.compile(r"<param_value>\s*(.*?)\s*</param_value>", re.IGNORECASE | re.DOTALL)
TOOL_CHATTER_PATTERN = re.compile(
    r"(?is)"
    r"(?:the\s+open_url\s+tool\s+is\s+not\s+available.*?|"
    r"let\s+me\s+use\s+the\s+correct.*?|"
    r"i\s+apologize\s+for\s+the\s+repeated.*?|"
    r"the\s+tool\s+open_url\s+appears\s+to\s+be\s+blocked.*?|"
    r"open_url工具被阻止.*?|"
    r"让我改用.*?工具.*?|"
    r"非常抱歉.*?|"
    r"我无法继续尝试.*?|"
    r"总结问题[:：].*?)(?=<(?:ml_)?tool_calls\b|$)"
)
TAG_NAME_HINTS = [
    "<m",
    "</m",
    "<ml_",
    "</ml_",
    "<ml_tool_calls",
    "</ml_tool_calls",
    "<ml_tool_call",
    "</ml_tool_call",
    "<ml_tool_name",
    "</ml_tool_name",
    "<ml_parameters",
    "</ml_parameters",
    "<ml_tool_result",
    "</ml_tool_result",
    "<tool_calls",
    "</tool_calls",
]


def _local_name(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    return tag.lower()


def _balanced_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _leaf_text(element: ET.Element) -> str:
    return _balanced_text("".join(element.itertext()))


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

    repeated_item_only = all(_local_name(child.tag) == "item" for child in children)
    if repeated_item_only:
        return [_xml_value_to_object(child) for child in children]

    result: dict[str, object] = {}
    for child in children:
        key = child.attrib.get("name", "").strip() or _local_name(child.tag)
        _append_value(result, key, _xml_value_to_object(child))
    return result


def _extract_tool_name(element: ET.Element) -> str:
    for tag_name in ("ml_tool_name", "tool_name"):
        tool_name_element = element.find(tag_name)
        if tool_name_element is not None:
            return _leaf_text(tool_name_element)
    return ""


def _extract_arguments(element: ET.Element) -> dict[str, object] | None:
    for tag_name in ("ml_parameters", "parameters"):
        parameters_element = element.find(tag_name)
        if parameters_element is not None:
            parsed = _xml_value_to_object(parameters_element)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
    return None


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


def _parse_tool_call_element(
    element: ET.Element,
    allowed_tool_names: set[str] | None,
    index: int,
) -> dict[str, object] | None:
    if _local_name(element.tag) not in {"tool_call", "ml_tool_call"}:
        return None

    tool_name = _extract_tool_name(element)
    if not tool_name:
        return None
    if allowed_tool_names is not None and tool_name not in allowed_tool_names:
        return None

    arguments = _extract_arguments(element)
    if arguments is None:
        return None

    return _build_tool_call(tool_name, arguments, index)


def _extract_malformed_tool_call_from_root(
    root: ET.Element,
    allowed_tool_names: set[str] | None,
    index: int,
) -> dict[str, object] | None:
    root_name = _local_name(root.tag)
    if root_name not in {"tool_calls", "ml_tool_calls"}:
        return None

    tool_name = _extract_tool_name(root)
    if not tool_name:
        return None
    if allowed_tool_names is not None and tool_name not in allowed_tool_names:
        return None

    for tag_name in ("ml_parameters", "parameters"):
        parameters_element = root.find(tag_name)
        if parameters_element is not None:
            parsed = _xml_value_to_object(parameters_element)
            arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
            return _build_tool_call(tool_name, arguments, index)

    names = [match.group(1).strip() for match in PARAM_NAME_TAG_PATTERN.finditer(ET.tostring(root, encoding="unicode"))]
    values = [match.group(1).strip() for match in PARAM_VALUE_TAG_PATTERN.finditer(ET.tostring(root, encoding="unicode"))]
    if names and values and len(names) == len(values):
        arguments = {
            key: _coerce_leaf_value(value)
            for key, value in zip(names, values, strict=False)
            if key
        }
        return _build_tool_call(tool_name, arguments, index)
    if names and not values:
        return None

    direct_pairs: dict[str, object] = {}
    children = [child for child in list(root) if isinstance(child.tag, str)]
    for child in children:
        key = _local_name(child.tag)
        if key in {"tool_name", "ml_tool_name", "tool_call", "ml_tool_call"}:
            continue
        if key in {"param_name", "param_value"}:
            continue
        direct_pairs[key] = _xml_value_to_object(child)
    if direct_pairs:
        return _build_tool_call(tool_name, direct_pairs, index)
    return None


def _parse_xml_block(
    block: str,
    allowed_tool_names: set[str] | None,
    start_index: int,
) -> tuple[list[dict[str, object]], tuple[int, int] | None]:
    try:
        root = ET.fromstring(block)
    except ET.ParseError:
        return [], None

    root_name = _local_name(root.tag)
    if root_name in {"tool_calls", "ml_tool_calls"}:
        candidates = [child for child in list(root) if isinstance(child.tag, str)]
    elif root_name in {"tool_call", "ml_tool_call"}:
        candidates = [root]
    else:
        return [], None

    tool_calls: list[dict[str, object]] = []
    for candidate in candidates:
        parsed = _parse_tool_call_element(candidate, allowed_tool_names, len(tool_calls))
        if parsed is not None:
            tool_calls.append(parsed)

    if not tool_calls:
        malformed = _extract_malformed_tool_call_from_root(root, allowed_tool_names, 0)
        if malformed is not None:
            tool_calls.append(malformed)

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
    tag_name = start_match.group("tag").lower()
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
        if match.group("tag").lower().startswith("ml_"):
            spans.append((start, end))
            cursor = end
            continue

        cursor = match.end()

    return spans, tool_calls


def _remove_spans(text: str, spans: list[tuple[int, int]], *, trim_outer_whitespace: bool = True) -> str:
    if not spans:
        cleaned = TOOL_RESULT_PATTERN.sub("", text)
        cleaned = TOOL_CHATTER_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip() if trim_outer_whitespace else cleaned

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
    cleaned = TOOL_CHATTER_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() if trim_outer_whitespace else cleaned


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
        max_overlap = min(len(hint), len(text))
        for size in range(max_overlap, 0, -1):
            if text.endswith(hint[:size]):
                return len(text) - size
    return None


def _looks_like_tool_markup_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("<ml_") or stripped.startswith("</ml_"):
        return True
    if stripped.startswith("<tool_") or stripped.startswith("</tool_"):
        return True
    if stripped.startswith("<m") and any(token in stripped for token in ("ml_", "tool_", "tool_calls", "tool_result")):
        return True
    return False


def _split_stream_text(
    text: str,
    allowed_tool_names: set[str] | None,
    final: bool,
) -> tuple[str, str, list[dict[str, object]]]:
    hold_from_candidates = [
        index
        for index in (_find_unmatched_fence_start(text), _find_incomplete_block_start(text))
        if index is not None
    ]

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
    spans, tool_calls = _extract_tool_blocks(processable, allowed_tool_names)
    visible = _remove_spans(processable, spans, trim_outer_whitespace=final)
    return visible, remainder, tool_calls


def parse_tool_calls_from_text(text: str, allowed_tool_names: set[str] | None = None) -> tuple[str, list[dict[str, object]]]:
    if not text:
        return "", []
    spans, tool_calls = _extract_tool_blocks(text, allowed_tool_names)
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
        tail = "" if _looks_like_tool_markup_fragment(remainder) else remainder
        return (visible + tail).strip(), self.tool_calls
