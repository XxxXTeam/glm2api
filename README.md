# ChatGLM 2 API

## 已取消维护

![图片1](https://pro-cs-freq.udeskcs.com/chat_upload/tid26/0017732a0b8e70d778b5e64a735a0d851777298167_1777298167388_a4dee0?OSSAccessKeyId=LTAI5tGLdqHDT2hBX8kvY2WR&Expires=1777326967&Signature=s0Oteyi4PnrSBtSb64kTxkv6Tf8%3D)

![图片2](https://pro-cs-freq.udeskcs.com/chat_upload/tid26/0a5998835b5cbd47e4e7585875d2402a1777298186_1777298186395_69d965?OSSAccessKeyId=LTAI5tGLdqHDT2hBX8kvY2WR&Expires=1777326986&Signature=CWOWOGcScsfCK5ITXKCm5wD8Ncw%3D)

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
- 你能获取到有效的 `refresh_token`，或者接受游客模式的能力限制
- 本地已准备好 Python 虚拟环境

## 2. 获取 GLM Refresh Token / 游客模式

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

如果你不想登录账号，也可以直接启用游客模式：

```env
GLM_USE_GUEST_REFRESH_TOKEN=true
```

如果既没有配置 `token.txt`，也没有配置 `GLM_REFRESH_TOKEN`，程序也会自动退回游客模式，并在请求失败时自动重新获取新的游客 `refresh_token` 后重试。

## 3. 配置文件

先复制示例配置：

```bash
cp .env.example .env
```

如果当前目录没有 `.env`，程序启动时也会自动从 `.env.example` 复制一份默认配置再继续加载。

推荐优先准备 `token.txt`，每行一个账号的 `refresh_token`：

```text
token-a
token-b
token-c
```

如果你暂时只有一个账号，也可以继续只改 `.env` 里的这一项：

```env
GLM_REFRESH_TOKEN=你的_refresh_token
```

如果你想显式固定走游客模式，可以这样写：

```env
GLM_USE_GUEST_REFRESH_TOKEN=true
GLM_GUEST_MAX_RETRIES=3
```

启用游客模式后，程序会按 `GLM_MAX_CONCURRENCY` 自动创建同等数量的游客账号槽位，让每个并发请求优先使用独立游客账号，避免多个并发长期挤在同一游客会话上。

常用配置说明：

- `HOST`
  服务监听地址。只给本机使用时填 `127.0.0.1`，局域网访问可填 `0.0.0.0`

- `PORT`
  服务端口，默认 `8000`

- `API_PREFIX`
  OpenAI 兼容路径前缀，默认 `/v1`

- `DEBUG_DUMP_ALL`
  调试狂暴模式。开启后会自动切到 `DEBUG`，并打印入站原始请求、转发给 GLM 的原始 body、上游原始响应和 SSE 分片、工具调用转换结果等几乎所有调试信息
  当 LOG_LEVEL=DEBUG（或 DEBUG_DUMP_ALL=true）时，自动在 log/glm2api_debug.log 写入日志文件（LOG_LEVEL=INFO — 只有终端输出，不写文件）

- `GLM_ASSISTANT_ID`
  普通对话使用的 assistant id

- `GLM_TOKEN_FILE`
  多账号 token 文件路径，默认 `token.txt`，每行一个 `refresh_token`

- `GLM_IMAGE_ASSISTANT_ID`
  图片生成使用的 assistant id

- `GLM_USE_GUEST_REFRESH_TOKEN`
  显式启用游客 ck；开启后会忽略已配置的账号 token

- `GLM_GUEST_MAX_RETRIES`
  游客 ck 请求失败时，最多自动重新拉取游客 token 并重试多少次

- `GLM_DELETE_CONVERSATION`
  是否在请求结束后自动删除 GLM 会话记录

- `GLM_MAX_CONCURRENCY`
  本地代理允许同时占用的上游执行槽位数量，默认 `3`

- `SERVER_API_KEYS`
  如果你希望访问本地代理时也带 Bearer Token，可以在这里填写

说明：

- 如果存在 `token.txt`，程序会优先从这里加载多账号
- 如果显式设置了 `GLM_USE_GUEST_REFRESH_TOKEN=true`，程序会直接走游客模式
- 游客模式下会按 `GLM_MAX_CONCURRENCY` 自动扩展游客账号池，尽量做到每个并发槽位对应一个独立游客账号
- 当某个账号请求失败时，会自动切换到下一账号继续尝试
- 如果本轮所有账号都失败，下一次会从第一个账号重新开始
- 当上游返回新的 `refresh_token` 时，多账号模式会自动写回 `token.txt` 对应行
- 单账号兜底模式下，程序仍会自动写回 `.env`
- 游客模式下不会把临时游客 `refresh_token` 落盘到 `.env` 或 `token.txt`
- 如果完全没有配置账号 token，程序会自动获取游客 `refresh_token` 作为兜底
- 如果你的 `.env` 不存在，程序无法自动落盘新的 token
- `/v1/models` 返回的模型列表已经固定写在代码中，不再通过配置文件自定义

## 4. 启动服务

### 拉取源代码

```bash
git clone https://github.com/XxxXTeam/glm2api.git
```
### 安装依赖

```bash
uv sync
```
### 运行项目

```bash
uv run .\main.py
```

或者：

```bash
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
  -d "{\"model\":\"glm-image-1\",\"prompt\":\"画个枫叶\",\"size\":\"1024x1024\"}"
```

### 8.2 Python OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

image = client.images.generate(
    model="glm-image-1",
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
- 并发槽位获取/释放
- 上游请求转发
- 会话删除结果
- 错误原因

如果你想查看更多细节，可以把 `.env` 中的：

```env
LOG_LEVEL=DEBUG
```

## 11. 常见问题

### 11.1 启动时报 `GLM_REFRESH_TOKEN` 缺失

新版本默认会自动退回游客模式；如果你仍想固定使用账号，请检查 `.env` 或 `token.txt` 里的 `refresh_token` 是否填写正确。

### 11.2 返回“请等待其他对话生成完毕”

说明同一账号在 GLM 侧存在并发限制。程序已经内置串行队列和自动等待重试。

### 11.3 返回“请登录后继续使用”

说明当前账号状态无效，或者 token 已失效，需要重新登录并更新 `refresh_token`
