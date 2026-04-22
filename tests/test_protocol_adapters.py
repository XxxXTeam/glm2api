from glm2api.services.anthropic_adapter import anthropic_to_openai
from glm2api.services.glm_auth import GLMAccessTokenManager
from glm2api.services.glm_client import GLMWebClient, UpstreamAPIError
from glm2api.services.responses_adapter import responses_to_openai


class _DummyConfig:
    glm_user_agent = "Mozilla/5.0"


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
