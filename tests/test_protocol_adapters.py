from glm2api.services.anthropic_adapter import anthropic_to_openai
from glm2api.services.glm_client import GLMWebClient, UpstreamAPIError
from glm2api.services.responses_adapter import responses_to_openai


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
