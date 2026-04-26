from glm2api.services.translator import GLMEventAccumulator, convert_messages


def test_convert_messages_injects_xml_tool_prompt_and_history():
    converted = convert_messages(
        messages=[
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"上海"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "call_1",
                "content": "晴",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert "<ml_tool_calls>" in prompt
    assert "<ml_tool_result call_id=\"call_1\" name=\"get_weather\">" in prompt
    assert "# TOOL USE PROTOCOL" in prompt
    assert "Use the private ml-prefixed canonical format below exactly." in prompt
    assert "The server will parse this XML intermediate language back into standard OpenAI tool_calls." in prompt
    assert "<actual_parameter_name><![CDATA[value]]></actual_parameter_name>" in prompt
    assert "never use a literal <param_name> placeholder tag" in prompt
    assert "Do not use <tool_calls>, <tool_call>, <tool_name>, <parameters>, <function_call>, <tool_use>, <invoke>, or any legacy wrapper." in prompt
    assert "# BLOCKED TOOLS" not in prompt
    assert "Ignore any tool names that are not listed in Allowed tool names" in prompt
    assert "Do not mention blocked browser, web, or open_url style tools at all." in prompt


def test_accumulator_build_response_maps_xml_to_openai_tool_calls():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"get_weather"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<ml_tool_calls><ml_tool_call><ml_tool_name>get_weather</ml_tool_name>"
                            "<ml_parameters><city>上海</city></ml_parameters></ml_tool_call></ml_tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    message = response["choices"][0]["message"]

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"city":"上海"}'


def test_convert_messages_respects_tool_choice_none_and_specific():
    none_converted = convert_messages(
        messages=[{"role": "user", "content": "直接回答"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="none",
    )
    none_prompt = none_converted[0]["content"][0]["text"]
    assert "# TOOL SCHEMAS" not in none_prompt

    specific_converted = convert_messages(
        messages=[{"role": "user", "content": "查天气"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    specific_prompt = specific_converted[0]["content"][0]["text"]
    assert "You must call exactly `get_weather` before giving a final answer." in specific_prompt


def test_convert_messages_repairs_cherry_fetch_url_and_skips_invalid_tool_error_history():
    converted = convert_messages(
        messages=[
            {
                "role": "user",
                "content": "使用工具访问 https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "function": {
                            "name": "mcp__CherryFetch__fetchJson",
                            "arguments": '{"param_name":"url"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_bad",
                "content": "{\"isError\":true,\"content\":[{\"type\":\"text\",\"text\":\"Invalid input: expected string, received undefined\"}]}",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "mcp__CherryFetch__fetchJson",
                    "description": "Fetch a JSON file from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert "<url><![CDATA[https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8]]></url>" in prompt
    assert "expected string, received undefined" not in prompt


def test_accumulator_repairs_param_name_only_tool_call_with_fallback_url():
    accumulator = GLMEventAccumulator(
        model="glm-test",
        allowed_tool_names={"mcp__CherryFetch__fetchJson"},
        fallback_tool_url="https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8",
    )
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<ml_tool_calls><ml_tool_call><ml_tool_name>mcp__CherryFetch__fetchJson</ml_tool_name>"
                            "<ml_parameters><param_name><![CDATA[url]]></param_name></ml_parameters>"
                            "</ml_tool_call></ml_tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    message = response["choices"][0]["message"]

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "mcp__CherryFetch__fetchJson"
    assert (
        message["tool_calls"][0]["function"]["arguments"]
        == '{"url":"https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8"}'
    )


def test_accumulator_keeps_markdown_block_separators_between_parts():
    accumulator = GLMEventAccumulator(model="glm-test")

    first_chunks, _ = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {"type": "text", "text": "## 查询结果：IP 地址 `1.1.1.1` 的归属地信息"},
                    ],
                }
            ],
        }
    )
    second_chunks, _ = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "2",
                    "content": [
                        {"type": "text", "text": "| 字段 | 值 |\n|---|---|\n| 查询 IP | `1.1.1.1` |"},
                    ],
                }
            ],
        }
    )

    assert first_chunks
    assert second_chunks[0].find("\\n\\n") != -1

    response = accumulator.build_response()
    assert response["choices"][0]["message"]["content"] == (
        "## 查询结果：IP 地址 `1.1.1.1` 的归属地信息\n\n"
        "| 字段 | 值 |\n|---|---|\n| 查询 IP | `1.1.1.1` |"
    )
