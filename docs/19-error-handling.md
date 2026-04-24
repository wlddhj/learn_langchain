# 第19章：异常处理与稳定性

## 19.1 为什么异常处理重要

LLM 应用面临多种不确定性：

| 错误类型 | 原因 | 影响 |
|---------|------|------|
| **API 错误** | 网络问题、服务不可用 | 请求失败 |
| **速率限制** | 超过调用频率限制 | 请求被拒绝 |
| **超时** | 响应时间过长 | 用户等待 |
| **Token 限制** | 输入/输出超出限制 | 截断或失败 |
| **模型错误** | LLM 返回无效内容 | 解析失败 |
| **工具错误** | 工具执行异常 | Agent 崩溃 |

**稳定性是生产环境的首要要求**。

## 19.2 常见错误类型

### OpenAI API 错误

```python
from openai import (
    APIError,           # 通用 API 错误
    APIConnectionError, # 连接错误
    RateLimitError,     # 速率限制
    APITimeoutError,    # 超时
    AuthenticationError, # 认证失败
    BadRequestError,    # 请求格式错误
)
```

### Anthropic API 错误

```python
from anthropic import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OverloadedError,    # 服务过载（Anthropic 特有）
)
```

## 19.3 基础错误处理

### 简单 try-except

```python
from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError, APITimeoutError

llm = ChatOpenAI(model="gpt-4o-mini")

def safe_invoke(prompt: str) -> str:
    """带错误处理的 LLM 调用"""
    try:
        response = llm.invoke(prompt)
        return response.content

    except RateLimitError:
        return "请求频率过高，请稍后重试"

    except APITimeoutError:
        return "请求超时，请稍后重试"

    except APIConnectionError:
        return "网络连接失败，请检查网络"

    except AuthenticationError:
        return "API Key 无效，请检查配置"

    except BadRequestError as e:
        return f"请求格式错误: {e}"

    except APIError as e:
        return f"API 错误: {e}"

    except Exception as e:
        return f"未知错误: {e}"

# 使用
result = safe_invoke("你好")
print(result)
```

### 异步错误处理

```python
import asyncio
from openai import APIError, RateLimitError

async def async_safe_invoke(prompt: str) -> str:
    """异步版本的错误处理"""
    try:
        response = await llm.ainvoke(prompt)
        return response.content

    except RateLimitError:
        await asyncio.sleep(2)  # 等待后重试
        return await async_safe_invoke(prompt)

    except APITimeoutError:
        return "请求超时"

    except APIError as e:
        return f"API 错误: {e}"

asyncio.run(async_safe_invoke("你好"))
```

## 19.4 with_retry —— 自动重试

### 基础重试

```python
from langchain_core.runnables import RunnableRetry

# 给 chain 添加自动重试
chain = prompt | llm | parser

chain_with_retry = chain.with_retry(
    stop_after_attempt=3,      # 最多重试 3 次
    wait_exponential_jitter=True,  # 指数退避 + 随机抖动
    retry_if_exception_type=(RateLimitError, APITimeoutError),  # 仅对这些错误重试
)

result = chain_with_retry.invoke({"topic": "AI"})
```

### 自定义重试策略

```python
# 更精细的重试配置
chain_with_retry = chain.with_retry(
    stop_after_attempt=5,           # 最多 5 次
    max_wait_seconds=60,            # 最大等待 60 秒
    wait_exponential_jitter=True,
    wait_exponential_multiplier=2,  # 指数乘数
    wait_exponential_min=1,         # 最小等待 1 秒
    retry_if_exception_type=(
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
    ),
)

# 重试时机：
# 第1次失败 → 等待 1-2 秒
# 第2次失败 → 等待 2-4 秒
# 第3次失败 → 等待 4-8 秒
# 第4次失败 → 等待 8-16 秒
# 第5次失败 → 抛出异常
```

### 异步 chain 的重试

```python
# 异步版本同样支持
async def async_retry_example():
    result = await chain_with_retry.ainvoke({"topic": "AI"})
    return result

asyncio.run(async_retry_example())
```

## 19.5 with_fallbacks —— 失败回退

### 多模型回退

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 主模型失败时，切换到备用模型
primary_llm = ChatOpenAI(model="gpt-4o")
fallback_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
emergency_llm = ChatOpenAI(model="gpt-4o-mini")

chain_with_fallbacks = (prompt | primary_llm | parser).with_fallbacks(
    [
        prompt | fallback_llm | parser,
        prompt | emergency_llm | parser,
    ]
)

# 执行过程：
# 1. 尝试 gpt-4o
# 2. 失败 → 尝试 claude-sonnet-4
# 3. 失败 → 尝试 gpt-4o-mini
# 4. 全失败 → 抛出最后一个异常

result = chain_with_fallbacks.invoke({"topic": "AI"})
```

### 回退 + 重试组合

```python
# 重试 + 回退的组合策略
robust_chain = chain.with_retry(
    stop_after_attempt=2,
    retry_if_exception_type=(RateLimitError,),
).with_fallbacks([
    fallback_chain.with_retry(stop_after_attempt=2),
])

# 执行过程：
# 1. 主模型重试 2 次
# 2. 全失败 → 备用模型重试 2 次
# 3. 全失败 → 抛出异常
```

### 异步回退

```python
async def async_fallback_example():
    result = await chain_with_fallbacks.ainvoke({"topic": "AI"})
    return result

asyncio.run(async_fallback_example())
```

## 19.6 速率限制

### 内置速率限制器

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

# 创建速率限制器
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,      # 每秒最多 2 次请求
    check_every_n_seconds=0.1,  # 每 0.1 秒检查一次
    max_bucket_size=10,         # 桶的最大容量（允许短时突发）
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    rate_limiter=rate_limiter,
)

# 超过速率时自动等待
for i in range(10):
    response = llm.invoke(f"问题{i}")
    print(f"完成 {i}")
```

### 自定义速率限制

```python
import asyncio
from asyncio import Semaphore

class AsyncRateLimiter:
    """异步速率限制器"""

    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.semaphore = Semaphore(10)  # 限制并发数
        self.last_time = 0

    async def acquire(self):
        await self.semaphore.acquire()
        now = time.time()
        wait_time = self.last_time + self.min_interval - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self.last_time = time.time()

    def release(self):
        self.semaphore.release()


async def rate_limited_invoke(prompt: str):
    limiter = AsyncRateLimiter(2)  # 每秒 2 次
    await limiter.acquire()
    try:
        return await llm.ainvoke(prompt)
    finally:
        limiter.release()
```

## 19.7 超时控制

### LLM 调用超时

```python
from langchain_openai import ChatOpenAI

# 构造时设置超时
llm = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=30,  # 30秒超时
    max_retries=2,       # 超时后重试 2 次
)

response = llm.invoke("你好")
```

### asyncio 超时

```python
async def with_timeout():
    try:
        response = await asyncio.wait_for(
            llm.ainvoke("复杂问题"),
            timeout=30.0,
        )
        return response.content

    except asyncio.TimeoutError:
        return "请求超时，请简化问题或稍后重试"

asyncio.run(with_timeout())
```

### Chain 级别超时

```python
# 使用 RunnableLambda 包装超时逻辑
from langchain_core.runnables import RunnableLambda

async def timeout_wrapper(input_dict, timeout_seconds=30):
    try:
        return await asyncio.wait_for(
            chain.ainvoke(input_dict),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return {"error": "超时"}

safe_chain = RunnableLambda(timeout_wrapper)
```

## 19.8 工具的异常处理

### 工具内部处理

```python
from langchain_core.tools import tool

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询"""
    try:
        result = execute_sql(sql)
        if not result:
            return "查询结果为空"
        return format_result(result)

    except ConnectionError:
        return "数据库连接失败，请稍后重试"

    except TimeoutError:
        return "查询超时，请简化查询条件"

    except ValueError as e:
        return f"查询格式错误: {e}"

    except Exception as e:
        return f"查询失败: {type(e).__name__} - {e}"
        # 注意：不抛出异常，返回错误文本让 Agent 理解
```

### 工具级错误处理原则

```python
# ❌ 错误：抛出异常让 Agent 崩溃
@tool
def bad_tool(query: str) -> str:
    result = risky_operation(query)
    return result  # 失败时抛异常

# ✅ 正确：返回错误文本让 Agent 处理
@tool
def good_tool(query: str) -> str:
    try:
        result = risky_operation(query)
        return result
    except Exception as e:
        return f"操作失败: {e}。请尝试其他方法。"
```

## 19.9 LangGraph 的错误处理

### 节点级错误处理

```python
from langgraph.graph import StateGraph, START, END

def robust_node(state):
    """带错误处理的节点"""
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "error": None}

    except RateLimitError:
        # 返回错误状态，让图决定下一步
        return {"error": "rate_limit"}

    except Exception as e:
        return {"error": str(e)}

def error_handler(state):
    """错误处理节点"""
    if state.get("error") == "rate_limit":
        return {"messages": [("system", "等待后重试")]}
    return {"messages": [("system", f"发生错误: {state['error']}")]}

def route_after_node(state):
    if state.get("error"):
        return "error_handler"
    return END

builder = StateGraph(State)
builder.add_node("main", robust_node)
builder.add_node("error_handler", error_handler)

builder.add_edge(START, "main")
builder.add_conditional_edges("main", route_after_node)
builder.add_edge("error_handler", END)
```

### Graph 级别重试

```python
from langgraph.pregel import RetryPolicy

# 编译时添加重试策略
graph = builder.compile(
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        exponential_base=2,
        jitter=True,
        retry_on=(RateLimitError, APITimeoutError),
    )
)
```

## 19.10 输入验证

### 参数校验

```python
from pydantic import BaseModel, Field, validator

class UserInput(BaseModel):
    """用户输入验证"""
    question: str = Field(min_length=1, max_length=1000)

    @validator('question')
    def sanitize(cls, v):
        # 清理潜在危险字符
        if any(word in v.lower() for word in ['drop', 'delete', 'truncate']):
            raise ValueError("包含潜在危险关键词")
        return v.strip()

def safe_invoke(user_input: str):
    try:
        validated = UserInput(question=user_input)
        return chain.invoke({"question": validated.question})
    except ValueError as e:
        return f"输入验证失败: {e}"
```

### Token 预估检查

```python
def check_token_limit(text: str, max_tokens: int = 8000):
    """检查输入是否会超出 token 限制"""
    # 粗略估算：每 4 字符约 1 token
    estimated = len(text) / 4

    if estimated > max_tokens:
        raise ValueError(
            f"输入过长，预估 {estimated} tokens，超过限制 {max_tokens}"
        )

    return text

def invoke_with_check(prompt: str):
    try:
        check_token_limit(prompt)
        return llm.invoke(prompt)
    except ValueError as e:
        return f"输入过长，请缩短内容: {e}"
```

## 19.11 日志与监控

### 错误日志记录

```python
import logging

logger = logging.getLogger("llm_app")
logger.setLevel(logging.ERROR)

def logged_invoke(prompt: str):
    """带日志记录的调用"""
    try:
        response = llm.invoke(prompt)
        return response.content

    except RateLimitError as e:
        logger.warning(f"速率限制: {e}")
        return "请稍后重试"

    except Exception as e:
        logger.error(f"调用失败: {e}", exc_info=True)
        return f"发生错误: {e}"
```

### 错误统计

```python
from collections import Counter

class ErrorTracker:
    """错误统计"""

    def __init__(self):
        self.errors = Counter()
        self.total_calls = 0

    def track(self, func):
        def wrapper(*args, **kwargs):
            self.total_calls += 1
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.errors[type(e).__name__] += 1
                raise
        return wrapper

    def report(self):
        return {
            "total_calls": self.total_calls,
            "error_rate": sum(self.errors.values()) / self.total_calls,
            "errors": dict(self.errors),
        }

tracker = ErrorTracker()

@tracker.track
def tracked_invoke(prompt):
    return llm.invoke(prompt)
```

## 19.12 稳定性设计清单

| 检查项 | 说明 |
|--------|------|
| **API 错误处理** | 捕获 RateLimitError、TimeoutError 等 |
| **重试机制** | `with_retry()` 处理瞬时错误 |
| **回退策略** | `with_fallbacks()` 多模型备选 |
| **速率限制** | `InMemoryRateLimiter` 或自定义 |
| **超时设置** | 构造时 `request_timeout` 或 `asyncio.wait_for` |
| **工具错误** | 内部捕获异常，返回错误文本 |
| **输入验证** | Pydantic 校验 + Token 预估 |
| **错误日志** | 记录异常详情 |
| **错误统计** | 监控错误率 |

## 19.13 本章小结

- 使用 try-except 捕获 API 错误：`RateLimitError`、`TimeoutError` 等
- `with_retry()` 自动重试，支持指数退避
- `with_fallbacks()` 失败时切换备用模型
- `InMemoryRateLimiter` 控制调用频率
- 使用 `request_timeout` 或 `asyncio.wait_for` 设置超时
- 工具内部捕获异常，返回描述性错误文本
- LangGraph 支持节点级和 Graph 级错误处理
- 输入验证：Pydantic 校验 + Token 预估
- 生产环境必备：错误日志 + 错误统计 + 监控