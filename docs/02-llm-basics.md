# 第2章：LLM 基础

## 2.1 LangChain 中的两种模型接口

LangChain 提供两种核心模型接口：

| 接口 | 用途 | 输入 | 输出 |
|------|------|------|------|
| **ChatModel** | 对话场景（主流） | 消息列表 | AIMessage |
| **LLM** | 纯文本补全 | 字符串 | 字符串 |

> **实际开发中 99% 使用 ChatModel**，LLM 接口已较少使用。

## 2.2 ChatModel 基础使用

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 创建模型实例
llm = ChatOpenAI(
    model="gpt-4o-mini",   # 模型名称
    temperature=0,          # 0=确定性输出，1=随机性更强
    max_tokens=1024,        # 最大输出 token 数
)
```

### 常用模型提供商

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_deployment="my-deployment",
    api_version="2024-06-01",
)
```

## 2.3 消息类型 (Message Types)

ChatModel 使用消息（Message）作为输入输出单位：

```python
from langchain_core.messages import (
    SystemMessage,    # 系统指令
    HumanMessage,     # 用户消息
    AIMessage,        # AI 回复
)

# 方式一：使用消息对象
messages = [
    SystemMessage(content="你是一个 Python 编程专家。"),
    HumanMessage(content="什么是列表推导式？"),
]

response = llm.invoke(messages)
print(type(response))      # <class 'langchain_core.messages.AIMessage'>
print(response.content)     # AI 的回复文本
print(response.response_metadata)  # 元数据（token 用量等）
```

### 方式二：使用便捷的元组语法（推荐）

```python
messages = [
    ("system", "你是一个 Python 编程专家。"),   # 等价于 SystemMessage
    ("human", "什么是列表推导式？"),             # 等价于 HumanMessage
]

response = llm.invoke(messages)
```

## 2.4 调用方式

### 同步调用

```python
response = llm.invoke("你好")
print(response.content)
```

### 异步调用（推荐用于生产环境）

```python
import asyncio

async def main():
    response = await llm.ainvoke("你好")
    print(response.content)

asyncio.run(main())
```

### 批量调用

```python
# 一次发送多个请求，提高效率
responses = llm.batch(["你好", "再见", "谢谢"])
for r in responses:
    print(r.content)
```

## 2.5 流式输出 (Streaming)

流式输出让用户更快看到响应，提升体验：

```python
# 同步流式
for chunk in llm.stream("请写一首关于编程的短诗"):
    print(chunk.content, end="", flush=True)
print()  # 换行

# 异步流式
async def stream_response():
    async for chunk in llm.astream("请写一首关于编程的短诗"):
        print(chunk.content, end="", flush=True)
    print()
```

## 2.6 Token 用量追踪

```python
response = llm.invoke("你好")

# 获取 token 用量
token_usage = response.response_metadata.get("token_usage", {})
print(f"输入 tokens: {token_usage.get('prompt_tokens')}")
print(f"输出 tokens: {token_usage.get('completion_tokens')}")
print(f"总 tokens: {token_usage.get('total_tokens')}")
```

## 2.7 错误处理

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

# 设置速率限制
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,    # 每秒最多 2 次请求
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    rate_limiter=rate_limiter,
)

# 基本错误处理
from openai import RateLimitError, APIError

try:
    response = llm.invoke("你好")
except RateLimitError:
    print("请求频率过高，请稍后重试")
except APIError as e:
    print(f"API 错误: {e}")
```

## 2.8 结构化输出 (Structured Output)

让 LLM 返回符合特定 schema 的结构化数据：

```python
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    """电影评价的结构"""
    title: str = Field(description="电影名称")
    rating: int = Field(description="评分 1-10")
    summary: str = Field(description="一句话评价")
    recommend: bool = Field(description="是否推荐")

# 方法一：with_structured_output
structured_llm = llm.with_structured_output(MovieReview)

result = structured_llm.invoke("评价一下电影《盗梦空间》")
print(result.title)       # 盗梦空间
print(result.rating)      # 9
print(result.recommend)   # True
print(type(result))       # MovieReview
```

## 2.9 本章小结

- LangChain 核心使用 **ChatModel** 接口，通过消息列表交互
- 三种消息类型：`SystemMessage`、`HumanMessage`、`AIMessage`
- 支持 `invoke`（同步）、`ainvoke`（异步）、`batch`（批量）、`stream`（流式）
- `with_structured_output` 可以让 LLM 返回结构化的 Pydantic 对象
- 始终关注 token 用量和错误处理
