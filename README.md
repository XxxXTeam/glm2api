# glm2api

`glm2api` 是一个基于标准库实现的轻量代理服务，用来把智谱清言 `chatglm.cn` Web 端接口转换成 OpenAI Chat Completions 兼容接口。

项目目标：

- 保持最小依赖，降低环境要求和部署成本
- 按模块拆分配置、日志、协议转换、上游调用逻辑
- 尽量复用 `docs` 中已经验证过的 GLM 签名、请求头和 SSE 解析思路

## 已实现能力

- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `stream=true` 的 SSE 流式转发
- `GET /v1/models`
- `GET /health`
- 彩色终端日志
- `.env` 配置加载
- 基于 `refresh_token` 自动刷新 `access_token`
- GLM 的 `[function_calls]` 协议转 OpenAI `tool_calls`
- 文本消息、工具消息，以及常见 `image_url` / `file` 引用上传
- OpenAI 图片生成请求映射到 GLM 绘图 assistant

## 快速开始

1. 复制配置文件：

```powershell
Copy-Item .env.example .env
```

2. 填写 `.env` 里的 `GLM_REFRESH_TOKEN`

3. 启动服务：

```powershell
.\.venv\Scripts\python.exe main.py
```

4. 调用测试：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## 目录结构

```text
src/glm2api/
  app.py                应用装配
  config.py             .env 与运行配置
  logging_utils.py      彩色日志
  model_profiles.py     模型画像
  server.py             HTTP 服务入口
  services/
    glm_auth.py         GLM token 刷新与签名
    glm_client.py       上游请求、附件上传、SSE 读取
    translator.py       OpenAI <-> GLM 协议转换
  utils/
    tool_parser.py      [function_calls] 解析
tests/
  test_core.py
```

## OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-required-if-server-api-keys-empty",
)

resp = client.chat.completions.create(
    model="glm-4",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
)

print(resp.choices[0].message.content)
```

## 图片生成示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-required-if-server-api-keys-empty",
)

image = client.images.generate(
    model="gpt-image-1",
    prompt="画个枫叶",
    size="1024x1024",
)

print(image.data[0].url)
```

当前已支持的图片参数：

- `prompt`
- `model`
- `n`
- `size`
- `response_format`
- `style`
- `scene`

## 说明

- 这个实现优先遵循 KISS 和 YAGNI，只保留当前 `docs` 足以支撑的核心链路。
- 没有引入 FastAPI、httpx、python-dotenv 等依赖，方便在当前仓库直接运行和二次维护。
- 如果后续你想扩展多账号池、轮询重试、上下文摘要、更多 provider，可以继续在现有模块上扩展，而不需要推倒重来。
