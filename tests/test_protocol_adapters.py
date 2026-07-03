import json
import threading
import time
import urllib.request
from types import SimpleNamespace

from glm2api import server as server_module
from glm2api.services.anthropic_adapter import anthropic_to_openai
from glm2api.services.glm_auth import GLMAccessTokenManager
from glm2api.services.glm_client import GLMWebClient, UpstreamAPIError

from glm2api.services.responses_adapter import ResponsesStreamAccumulator, openai_to_responses, responses_to_openai


class _DummyConfig:
    glm_user_agent = "Mozilla/5.0"
    glm_use_guest_refresh_token = False


def test_get_browser_headers_includes_random_x_forwarded_for():
    manager = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
    manager.config = _DummyConfig()

    headers = manager.get_browser_headers()
    xff = headers["X-Forwarded-For"]
    octets = xff.split(".")

    assert len(octets) == 4
    assert all(part.isdigit() for part in octets)
    assert 1 <= int(octets[0]) <= 223
    assert int(octets[0]) not in {10, 127, 169, 172, 192}


def test_responses_to_openai_preserves_tool_choice():
    payload = {
        "model": "glm-4",
        "input": "hi",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "查询天气",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }

    converted = responses_to_openai(payload)

    assert converted["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_responses_to_openai_accepts_sdk_style_input_messages():
    payload = {
        "model": "glm-4",
        "input": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            },
        ],
    }

    converted = responses_to_openai(payload)

    assert converted["messages"][0] == {"role": "user", "content": "hi"}
    assert converted["messages"][1] == {"role": "assistant", "content": "hello"}
    assert converted["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }


def test_openai_to_responses_exposes_output_text_and_standard_fields():
    response = openai_to_responses(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        },
        model="glm-4",
    )

    assert response["object"] == "response"
    assert response["output_text"] == "hello"
    assert response["error"] is None
    assert response["incomplete_details"] is None
    assert response["usage"] == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}


def test_responses_stream_uses_openai_event_envelope():
    accumulator = ResponsesStreamAccumulator(model="glm-4")

    events = accumulator.start_response()
    events.extend(
        accumulator.feed_chunk(
            b'data: {"choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}\n\n'
            b"data: [DONE]\n\n"
        )
    )

    payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in events
        if event.startswith("event: ")
    ]

    assert payloads[0]["type"] == "response.created"
    assert payloads[0]["sequence_number"] == 0
    assert payloads[0]["response"]["object"] == "response"
    assert payloads[0]["response"]["usage"] is None
    assert payloads[0]["response"]["parallel_tool_calls"] is True
    text_delta = next(payload for payload in payloads if payload["type"] == "response.output_text.delta")
    assert text_delta["delta"] == "hi"
    assert text_delta["response_id"] == payloads[0]["response"]["id"]
    assert payloads[-1]["type"] == "response.completed"
    assert payloads[-1]["response"]["status"] == "completed"
    assert payloads[-1]["response"]["completed_at"] is not None
    assert payloads[-1]["response"]["usage"]["total_tokens"] == 0
    assert events[-1] == "data: [DONE]\n\n"


def test_responses_stream_buffers_split_sse_blocks_until_done():
    accumulator = ResponsesStreamAccumulator(model="glm-4")

    events = accumulator.feed_chunk(
        b'data: {"choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}\n'
    )
    events.extend(accumulator.feed_chunk(b"\ndata: [DO"))
    events.extend(accumulator.feed_chunk(b"NE]\n\n"))

    payload_events = [event for event in events if event.startswith("event: ")]
    payloads = [json.loads(event.split("data: ", 1)[1]) for event in payload_events]

    assert any(payload["type"] == "response.output_text.delta" and payload["delta"] == "hi" for payload in payloads)
    assert payloads[-1]["type"] == "response.completed"
    assert events[-1] == "data: [DONE]\n\n"


def test_responses_stream_completes_on_finish_reason_without_done_sentinel():
    accumulator = ResponsesStreamAccumulator(model="glm-4")

    events = accumulator.feed_chunk(
        b'data: {"choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
    )

    payload_events = [event for event in events if event.startswith("event: ")]
    payloads = [json.loads(event.split("data: ", 1)[1]) for event in payload_events]

    assert payloads[-1]["type"] == "response.completed"
    assert payloads[-1]["response"]["usage"]["input_tokens"] == 2
    assert payloads[-1]["response"]["usage"]["output_tokens"] == 3
    assert events[-1] == "data: [DONE]\n\n"


def test_responses_http_stream_sends_keepalive_while_upstream_is_idle(monkeypatch):
    monkeypatch.setattr(server_module, "RESPONSES_STREAM_HEARTBEAT_SECONDS", 0.01)

    class FakeGLM:
        def stream_chat_completion(self, payload):
            yield b'data: {"choices":[{"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}\n\n'
            time.sleep(0.05)
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

    class FakeLogger:
        def debug(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass

    config = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=[],
        debug_dump_all=False,
        exposed_models=["glm-4"],
    )
    server = server_module.GLM2APIServer(config, FakeGLM(), FakeLogger())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server._server.server_address[1]
    try:
        body = json.dumps({"model": "glm-4", "input": "hi", "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/responses",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            chunks: list[str] = []
            while True:
                try:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk.decode("utf-8"))
                    if "data: [DONE]\n\n" in chunks[-1]:
                        break
                except TimeoutError:
                    break
            stream_text = "".join(chunks)
    finally:
        server.shutdown()
        thread.join(timeout=1)

    assert ": keep-alive\n\n" in stream_text
    assert "response.completed" in stream_text


def test_anthropic_to_openai_maps_tool_choice_variants():
    any_payload = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "any"},
    }
    tool_payload = {
        "model": "glm-4",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }

    any_converted = anthropic_to_openai(any_payload)
    tool_converted = anthropic_to_openai(tool_payload)

    assert any_converted["tool_choice"] == "required"
    assert tool_converted["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_glm_client_raises_for_sse_error_event():
    client = GLMWebClient.__new__(GLMWebClient)

    try:
        client._raise_for_event_error(
            {
                "status": "error",
                "last_error": {"error_code": 10025, "err_msg": "stream request error"},
                "parts": [],
            },
            stream=True,
        )
    except UpstreamAPIError as exc:
        assert exc.status_code == 502
        assert "10025" in str(exc)
        assert "stream request error" in str(exc)
    else:
        raise AssertionError("expected UpstreamAPIError")
