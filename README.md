# glm2api 用户手册

`glm2api` 是一个本地代理服务，用来把 `chatglm.cn` 的网页接口转换成 OpenAI 兼容接口，方便你直接接入 OpenAI SDK、Cherry Studio、Open WebUI、LobeChat 或其他兼容 OpenAI API 的工具。

支持的主要接口：

- `POST /v1/chat/completions`
- `POST /v1/images/generations`
- `GET /v1/models`
- `GET /health`

## 1. 使用前准备

启动前请确认：

- 你已经登录过 `https://chatglm.cn`
> 其实不登陆也行,但是会有部分限制?
- 你能获取到有效的 `refresh_token`
- 本地已准备好 Python 虚拟环境

## 2. 获取 GLM Refresh Token

获取方式：

1. 打开 `https://chatglm.cn`
2. 登录你的账号
3. 按 `F12` 打开开发者工具
4. 进入 `Application`
5. 查看 `Local Storage` 或相关存储项
6. 找到 `chatglm_refresh_token`

拿到后，将它填入 `.env` 文件中的：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

## 3. 配置文件

先复制示例配置：

```bash
cp .env.example .env
```

最少只需要改这一项：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

常用配置说明：

- `HOST`
  服务监听地址。只给本机使用时填 `127.0.0.1`，局域网访问可填 `0.0.0.0`

- `PORT`
  服务端口，默认 `8000`

- `API_PREFIX`
  OpenAI 兼容路径前缀，默认 `/v1`

- `GLM_ASSISTANT_ID`
  普通对话使用的 assistant id

- `GLM_IMAGE_ASSISTANT_ID`
  图片生成使用的 assistant id

- `GLM_DELETE_CONVERSATION`
  是否在请求结束后自动删除 GLM 会话记录

- `SERVER_API_KEYS`
  如果你希望访问本地代理时也带 Bearer Token，可以在这里填写

- `EXPOSED_MODELS`
  `/v1/models` 对外展示的模型列表

- `GLM_MODEL_ALIASES`
  模型别名映射，格式为 `对外模型名=上游模型名或assistant_id`

说明：

- 当上游返回新的 `refresh_token` 时，程序会自动写回 `.env`
- 如果你的 `.env` 不存在，程序无法自动落盘新的 token

## 4. 启动服务

直接运行：

```powershell
uv run .\main.py
```

或者：

```powershell
.\.venv\Scripts\python.exe main.py
```

启动成功后你会看到类似日志：

```text
启动服务 host=127.0.0.1 port=8000 prefix=/v1 models=...
```

## 5. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{"status":"ok"}
```

## 6. 查询模型列表

```bash
curl http://127.0.0.1:8000/v1/models
```

返回的是当前配置里暴露的模型列表。

## 7. 聊天接口

### 7.1 Curl 示例

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm-4\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}]}"
```

### 7.2 Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ],
)

print(resp.choices[0].message.content)
```

### 7.3 流式示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

stream = client.chat.completions.create(
    model="glm-4",
    messages=[{"role": "user", "content": "写一首七言绝句"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if getattr(delta, "content", None):
        print(delta.content, end="")
```

## 8. 图片生成接口

### 8.1 Curl 示例

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gpt-image-1\",\"prompt\":\"画个枫叶\",\"size\":\"1024x1024\"}"
```

### 8.2 Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

image = client.images.generate(
    model="gpt-image-1",
    prompt="画个枫叶",
    size="1024x1024",
)

print(image.data[0].url)
```

### 8.3 当前支持的图片参数

- `prompt`
- `model`
- `n`
- `size`
- `response_format`
- `style`
- `scene`

说明：

- 默认返回图片 URL
- 如果 `response_format=b64_json`，会返回 base64 图片数据
- `size` 会自动映射到 GLM 所需的宽高比例

## 9. 鉴权方式

如果 `.env` 中 `SERVER_API_KEYS` 为空，则本地接口默认不校验 Bearer Token。

如果你配置了：

```env
SERVER_API_KEYS=sk-local-1,sk-local-2
```

那么请求时需要带：

```http
Authorization: Bearer sk-local-1
```

## 10. 日志说明

程序默认输出彩色日志，常见内容包括：

- 服务启动
- 请求进入队列
- 上游请求转发
- 会话删除结果
- 错误原因

如果你想查看更多细节，可以把 `.env` 中的：

```env
LOG_LEVEL=DEBUG
```

## 11. 常见问题

### 11.1 启动时报 `GLM_REFRESH_TOKEN` 缺失

说明 `.env` 中没有填写有效 token。

### 11.2 返回“请等待其他对话生成完毕”

说明同一账号在 GLM 侧存在并发限制。程序已经内置串行队列和自动等待重试。

### 11.3 返回“请登录后继续使用”

说明当前账号状态无效，或者 token 已失效，需要重新登录并更新 `refresh_token`。

### 11.4 流式响应客户端迟迟不结束

当前版本已经在服务端补齐 OpenAI 标准的 `data: [DONE]` 结束标记。如果仍有问题，请检查你的客户端是否支持标准 SSE。

## 12. 目录说明

```text
src/glm2api/
  app.py                应用入口
  config.py             配置读取
  logging_utils.py      日志输出
  server.py             OpenAI 兼容 HTTP 服务
  services/
    glm_auth.py         token 刷新与写回
    glm_client.py       GLM 请求转发
    translator.py       GLM 响应转换
  utils/
    tool_parser.py      工具调用解析
```

## 13. 适用场景

适合以下情况：

- 想把 GLM 接到 OpenAI SDK
- 想在本地工具中统一使用 OpenAI 风格接口
- 想把 GLM 聊天和绘图一起接入现有工作流

如果你后续还想扩展更多接口，可以继续在当前结构上增加路由和上游映射逻辑。
