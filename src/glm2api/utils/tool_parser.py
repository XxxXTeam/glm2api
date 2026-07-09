from __future__ import annotations

import orjson
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .tool_protocol import BLOCKED_NATIVE_TOOL_NAMES

# Strip model apology loops before tool calls (e.g. "I apologize, let me use...").
# GLM inserts this chatter after blocked tool calls, wasting tokens + confusing parsers.
TOOL_CHATTER_PATTERN = re.compile(
    r"(?is)"
    r"(?:the\s+open_url\s+tool\s+is\s+not\s+available.*?|"
    r"let\s+me\s+use\s+the\s+correct.*?|"
    r"i\s+apologize\s+for\s+the\s+repeated.*?|"
    r"the\s+tool\s+open_url\s+appears\s+to\s+be\s+blocked.*?|"
    r"open_url\u5de5\u5177\u88ab\u963b\u6b62.*?)(?=<(?:ml_)?tool_calls\b)"
)


CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```")
TOOL_RESULT_PATTERN = re.compile(
    r"<(?:(?:\|DSML\|)|ml_)?tool_result\b[\s\S]*?</(?:(?:\|DSML\|)|ml_)?tool_result>",
    re.IGNORECASE,
)
START_TAG_PATTERN = re.compile(
    r"<(?P<tag>\|DSML\|tool_calls|tool_calls|ml_tool_calls|ml_tool_call)\b[^>]*>",
    re.IGNORECASE,
)
DSML_TAG_PATTERN = re.compile(r"</?\|DSML\|(?P<name>tool_calls|invoke|parameter|tool_result)\b", re.IGNORECASE)
DSML_OPEN_TAG_PATTERN = re.compile(
    r"<\|dsml\|(?P<name>tool_calls|toolcalls|invoke|parameter|tool_result|toolresult)\b(?P<attrs>[^<>]*?)>",
    re.IGNORECASE,
)
DSML_CLOSE_TAG_PATTERN = re.compile(
    r"</\|dsml\|(?P<name>tool_calls|toolcalls|invoke|parameter|tool_result|toolresult)\s*\|?\s*>",
    re.IGNORECASE,
)
DSML_COMPACT_CLOSE_TAG_PATTERN = re.compile(
    r"(?:</\|dsml|<\|/dsml)(?P<name>toolcalls|invoke|parameter|toolresult)\s*\|\s*>",
    re.IGNORECASE,
)
DSML_DOUBLE_PIPE_CLOSE_TAG_PATTERN = re.compile(
    r"<\|\|dsml\|(?P<name>tool_calls|toolcalls|invoke|parameter|tool_result|toolresult)\s*\|?\s*>",
    re.IGNORECASE,
)
DSML_TOOL_CALLS_CLOSE_PATTERN = re.compile(
    r"(?:</\|dsml\|tool_calls\s*>|</\|dsml\|tool_calls\s*\|\s*>|</\|dsmltool_?calls\s*\|\s*>|<\|/dsmltool_?calls\s*\|\s*>)",
    re.IGNORECASE,
)
DSML_TOOL_CALLS_TRAILING_CLOSE_PATTERN = re.compile(
    r"(?:</\|dsml\|tool_calls\s*>|</\|dsml\|tool_calls\s*\|\s*>|</\|dsmltool_?calls\s*\|\s*>|<\|/dsmltool_?calls\s*\|\s*>|</\|dsml\|tool_calls\s*$|</\|dsmltool_?calls\s*\|?\s*$|<\|/dsmltool_?calls\s*\|?\s*$)",
    re.IGNORECASE,
)
PARAM_NAME_TAG_PATTERN = re.compile(r"<param_name>\s*(.*?)\s*</param_name>", re.IGNORECASE | re.DOTALL)
PARAM_VALUE_TAG_PATTERN = re.compile(r"<param_value>\s*(.*?)\s*</param_value>", re.IGNORECASE | re.DOTALL)

# Match complete <|DSML|invoke ...> ... </|DSML|invoke> blocks (early streaming detection)
# After _repair_malformed_dsml normalization, tags are canonical
INVOKE_COMPLETE_PATTERN = re.compile(
    r"<\|DSML\|invoke\s+[^>]+>.*?</\|DSML\|invoke\s*>",
    re.DOTALL,
)

# Fast-path guard: cheap scan for any DSML-relevant characters before full parsing
_DSML_FAST = re.compile(r"<|\b(tool_call|invoke|parameter)\b", re.IGNORECASE).search
# Precompiled regex for collapsing excessive newlines
_CLEAN_NEWLINES_RE = re.compile(r"\n{3,}")


def _local_name(tag: str) -> str:
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    return tag.lower()


def _canonical_dsml_name(name: str) -> str:
    normalized = name.lower().replace("_", "")
    if normalized == "toolcalls":
        return "tool_calls"
    if normalized == "toolresult":
        return "tool_result"
    return normalized


def _repair_malformed_dsml(block: str) -> str:
    if "<|" not in block and "]]|>" not in block:
        return block

    repaired = block.replace("]]|>", "]]>")
    if "<![CDATA[" in repaired:
        repaired = re.sub(
            r"(?<!\])\]>(?=</\|dsml\|parameter\b|</\|DSML\|parameter\b|</parameter\b|</\|dsmlparameter\|)",
            "]]>",
            repaired,
            flags=re.IGNORECASE,
        )

    def replace_open(match: re.Match[str]) -> str:
        name = _canonical_dsml_name(match.group("name"))
        attrs = match.group("attrs").rstrip("|").rstrip()
        return f"<|DSML|{name}{attrs}>"

    def replace_close(match: re.Match[str]) -> str:
        return f"</|DSML|{_canonical_dsml_name(match.group('name'))}>"

    repaired = DSML_OPEN_TAG_PATTERN.sub(replace_open, repaired)
    repaired = DSML_CLOSE_TAG_PATTERN.sub(replace_close, repaired)
    repaired = DSML_COMPACT_CLOSE_TAG_PATTERN.sub(replace_close, repaired)
    repaired = DSML_DOUBLE_PIPE_CLOSE_TAG_PATTERN.sub(replace_close, repaired)
    repaired = re.sub(
        r"(?:</\|dsml\|tool_calls|</\|dsmltool_?calls|<\|/dsmltool_?calls)\s*\|?\s*$",
        "</|DSML|tool_calls>",
        repaired,
        flags=re.IGNORECASE,
    )
    return repaired


def _normalize_dsml_to_xml(block: str) -> str:
    repaired = _repair_malformed_dsml(block)
    return DSML_TAG_PATTERN.sub(lambda match: match.group(0).replace("|DSML|", ""), repaired)


def _is_allowed_tool_name(tool_name: str, allowed_tool_names: set[str] | None) -> bool:
    if tool_name in BLOCKED_NATIVE_TOOL_NAMES:
        return False
    return allowed_tool_names is None or tool_name in allowed_tool_names


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
            return orjson.loads(stripped)
        except orjson.JSONDecodeError:
            if stripped.startswith("[") and not stripped.endswith("]"):
                try:
                    return orjson.loads(stripped + "]")
                except orjson.JSONDecodeError:
                    pass
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
    if _local_name(element.tag) == "invoke":
        return element.attrib.get("name", "").strip()
    for tag_name in ("ml_tool_name", "tool_name"):
        tool_name_element = element.find(tag_name)
        if tool_name_element is not None:
            return _leaf_text(tool_name_element)
    return ""


def _extract_arguments(element: ET.Element) -> dict[str, object] | None:
    if _local_name(element.tag) == "invoke":
        parameters: dict[str, object] = {}
        parameter_children = [
            child
            for child in list(element)
            if isinstance(child.tag, str) and _local_name(child.tag) == "parameter"
        ]
        for child in parameter_children:
            key = child.attrib.get("name", "").strip()
            if key:
                _append_value(parameters, key, _xml_value_to_object(child))
        return parameters

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
            "arguments": orjson.dumps(arguments).decode("utf-8"),
        },
    }


def _parse_tool_call_element(
    element: ET.Element,
    allowed_tool_names: set[str] | None,
    index: int,
) -> dict[str, object] | None:
    if _local_name(element.tag) not in {"invoke", "tool_call", "ml_tool_call"}:
        return None

    tool_name = _extract_tool_name(element)
    if not tool_name:
        return None
    if not _is_allowed_tool_name(tool_name, allowed_tool_names):
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
    if not _is_allowed_tool_name(tool_name, allowed_tool_names):
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
        root = ET.fromstring(_normalize_dsml_to_xml(block))
    except ET.ParseError:
        return [], None

    root_name = _local_name(root.tag)
    if root_name in {"tool_calls", "ml_tool_calls"}:
        candidates = [
            child
            for child in list(root)
            if isinstance(child.tag, str) and _local_name(child.tag) in {"invoke", "tool_call", "ml_tool_call"}
        ]
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
    if "<" not in text and "|" not in text:  # fast-path: no DSML markers
        return text
    masked = list(text)
    for match in CODE_FENCE_PATTERN.finditer(text):
        for index in range(match.start(), match.end()):
            masked[index] = " "
    return "".join(masked)


def _find_matching_block(
    masked_text: str,
    start_match: re.Match[str],
    *,
    allow_trailing_close: bool = False,
) -> tuple[int, int] | None:
    tag_name = start_match.group("tag").lower()
    if tag_name == "|dsml|tool_calls":
        closing_pattern = DSML_TOOL_CALLS_TRAILING_CLOSE_PATTERN if allow_trailing_close else DSML_TOOL_CALLS_CLOSE_PATTERN
    else:
        closing_pattern = re.compile(rf"</{re.escape(tag_name)}\s*>", re.IGNORECASE)
    closing_match = closing_pattern.search(masked_text, start_match.end())
    if closing_match is None:
        return None
    return start_match.start(), closing_match.end()


def _extract_tool_blocks(
    text: str,
    allowed_tool_names: set[str] | None,
    *,
    allow_trailing_close: bool = False,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    masked_text = _mask_code_fences(text)
    if "<" not in masked_text:  # fast-path: no tags at all
        return [], []
    spans: list[tuple[int, int]] = []
    tool_calls: list[dict[str, object]] = []
    cursor = 0

    while cursor < len(masked_text):
        match = START_TAG_PATTERN.search(masked_text, cursor)
        if match is None:
            break
        span = _find_matching_block(masked_text, match, allow_trailing_close=allow_trailing_close)
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
        if match.group("tag").lower() in {"|dsml|tool_calls", "tool_calls", "ml_tool_calls", "ml_tool_call"}:
            spans.append((start, end))
            cursor = end
            continue

        cursor = match.end()

    return spans, tool_calls


def _remove_spans(text: str, spans: list[tuple[int, int]], *, trim_outer_whitespace: bool = True) -> str:
    if not spans:
        cleaned = TOOL_RESULT_PATTERN.sub("", text)
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
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Strip model apology loops before tool calls (TOOL_CHATTER_PATTERN)
    cleaned = TOOL_CHATTER_PATTERN.sub("", cleaned)
    return cleaned.strip() if trim_outer_whitespace else cleaned


def parse_tool_calls_from_text(text: str, allowed_tool_names: set[str] | None = None) -> tuple[str, list[dict[str, object]]]:
    if not text:
        return "", []
    spans, tool_calls = _extract_tool_blocks(text, allowed_tool_names, allow_trailing_close=True)
    return _remove_spans(text, spans), tool_calls


# ---------------------------------------------------------------------------
# Early invoke extraction — emit tool calls during streaming (not just at finalize)
# ---------------------------------------------------------------------------

def _has_unclosed_outer_block(text: str) -> int | None:
    """Return the index of the start of an unclosed outer DSML block, or None."""
    if not text or ("<" not in text and "|" not in text):
        return None
    masked = _mask_code_fences(text)
    start_match = START_TAG_PATTERN.search(masked)
    if start_match is None:
        return None
    span = _find_matching_block(masked, start_match, allow_trailing_close=False)
    return start_match.start() if span is None else None


def _extract_early_invoke_tool_calls(
    text: str,
    allowed_tool_names: set[str] | None,
    already_emitted_ends: set[int],
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    """Find complete <|DSML|invoke>...</|DSML|invoke> blocks inside an unclosed outer block.

    Returns (spans_to_mark_emitted, new_tool_calls).
    Used during streaming to emit tool calls before the outer </|DSML|tool_calls> arrives.
    """
    if not _has_unclosed_outer_block(text):
        return [], []

    normalized = _repair_malformed_dsml(text)
    new_spans: list[tuple[int, int]] = []
    new_calls: list[dict[str, object]] = []

    for match in INVOKE_COMPLETE_PATTERN.finditer(normalized):
        end = match.end()
        if end in already_emitted_ends:
            continue

        block = match.group(0)
        block_calls, _ = _parse_xml_block(block, allowed_tool_names, 0)
        if block_calls:
            new_spans.append((match.start(), end))
            new_calls.extend(block_calls)

    return new_spans, new_calls


# ---------------------------------------------------------------------------
# Helper: detect partial DSML tags at buffer end (streaming)
# ---------------------------------------------------------------------------

def _is_partial_dsml_tag(text: str) -> int | None:
    """Find position of an unclosed DSML-like tag at buffer end.

    Returns the index to hold from, or None if no partial tag detected.
    Used to avoid emitting fragments of DSML markup as visible text.

    Covers three cases:
      1. Bare `<` at end — could start any DSML tag (1-char prefix of <|, <m, <tool_, etc.)
      2. Partial tag name without `>` — e.g. `<|DSML|tool_calls` (no > yet)
      3. Complete opening tag without matching close — e.g. `<ml_tool_calls>` alone
    """
    if not text or ("<" not in text and "|" not in text):
        return None
    lowered = text.lower()

    # Priority 1: Complete opening tag without matching close.
    # e.g. `<ml_tool_calls>` with no `</ml_tool_calls>` yet.
    # Check BEFORE bare-< below so incomplete-block wins over trailing <.
    masked = _mask_code_fences(text)
    start_match = START_TAG_PATTERN.search(masked)
    if start_match is not None:
        span = _find_matching_block(masked, start_match, allow_trailing_close=False)
        if span is None:
            return start_match.start()

    # Priority 2: Bare < at end — could start any DSML tag.
    if text.endswith("<"):
        return len(text) - 1

    # Priority 3: Unclosed tag name (opening <|, <m, etc. with no > yet).
    for prefix in ("<|", "<m", "</m", "<tool_", "</tool_", "<invoke", "</invoke", "<parameter", "</parameter"):
        idx = lowered.rfind(prefix)
        if idx != -1 and ">" not in text[idx:]:
            return idx

    return None


def _clean_visible(text: str) -> str:
    """Strip model chatter and tool_result blocks from visible text."""
    cleaned = TOOL_CHATTER_PATTERN.sub("", text)
    cleaned = TOOL_RESULT_PATTERN.sub("", cleaned)
    cleaned = _CLEAN_NEWLINES_RE.sub("\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Streaming tool parser — state machine
# ---------------------------------------------------------------------------

@dataclass
class StreamingToolParser:
    """Streaming parser for DSML tool calls.

    Three-state machine:
      IDLE -> IN_DSML (on opening tag) -> IDLE (on matching close tag)
      Text outside DSML blocks is passed through as visible.
      Text inside DSML blocks is held until block completes, then parsed.

    Ponytail: no regex soup. Uses _extract_tool_blocks for complete blocks
    and _is_partial_dsml_tag to avoid leaking partial markup.
    """
    pending_text: str = ""
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    allowed_tool_names: set[str] | None = None
    # Tracks already-emitted invoke block ends (for early streaming extraction)
    _emitted_invoke_ends: set[int] = field(default_factory=set)

    def consume(self, chunk: str) -> str:
        """Feed a streaming text chunk. Returns visible (non-DSML) text."""
        if not chunk:
            return ""
        self.pending_text += chunk

        # Fast-path: if no DSML markers, skip all expensive parsing
        if _DSML_FAST(self.pending_text) is None:
            visible = self.pending_text
            self.pending_text = ""
            return _clean_visible(visible)

        # 1. Look for complete DSML blocks (outer <|DSML|tool_calls> closed)
        spans, new_calls = _extract_tool_blocks(
            self.pending_text, self.allowed_tool_names, allow_trailing_close=False
        )

        if spans:
            # Complete blocks found. Emit text outside all blocks.
            visible_parts: list[str] = []
            cursor = 0
            for start, end in spans:
                if start > cursor:
                    visible_parts.append(self.pending_text[cursor:start])
                cursor = end

            # Remaining text after last block — check for partial new DSML tag
            remaining = self.pending_text[cursor:]
            partial_pos = _is_partial_dsml_tag(remaining)
            if partial_pos is not None:
                visible_parts.append(remaining[:partial_pos])
                self.pending_text = remaining[partial_pos:]
            else:
                visible_parts.append(remaining)
                self.pending_text = ""

            # Deduplicate: only add new calls not already emitted via early extraction
            for tc in new_calls:
                if tc["id"] not in {e["id"] for e in self.tool_calls}:
                    self.tool_calls.append(tc)
            return _clean_visible("".join(visible_parts))

        # 1b. Early invoke extraction: complete <|DSML|invoke> inside unclosed outer block.
        #     These get emitted immediately so the client sees tool calls during streaming.
        invoke_spans, early_calls = _extract_early_invoke_tool_calls(
            self.pending_text, self.allowed_tool_names, self._emitted_invoke_ends
        )
        if early_calls:
            for tc in early_calls:
                tc["index"] = len(self.tool_calls)
                self.tool_calls.append(tc)
                self._emitted_invoke_ends.update(end for _, end in invoke_spans)

        # 2. No complete blocks. Check for partial (incoming) DSML tag.
        partial_pos = _is_partial_dsml_tag(self.pending_text)
        if partial_pos is not None:
            visible = self.pending_text[:partial_pos]
            self.pending_text = self.pending_text[partial_pos:]
            return _clean_visible(visible)

        # 3. No DSML content at all — emit everything.
        visible = self.pending_text
        self.pending_text = ""
        return _clean_visible(visible)

    def flush(self) -> tuple[str, list[dict[str, object]]]:
        """Finalize. Returns (remaining_visible_text, all_tool_calls)."""
        if not self.pending_text.strip():
            return "", self.tool_calls

        cleaned, remaining_calls = parse_tool_calls_from_text(
            self.pending_text.strip(), self.allowed_tool_names
        )
        # Deduplicate: only add calls not already emitted via early extraction
        emitted_ids = {tc["id"] for tc in self.tool_calls}
        for tc in remaining_calls:
            if tc["id"] not in emitted_ids:
                self.tool_calls.append(tc)
        self.pending_text = ""
        self._emitted_invoke_ends.clear()
        return cleaned.strip(), self.tool_calls
