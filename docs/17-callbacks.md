# 第17章：回调系统 (Callbacks)

## 17.1 为什么需要回调

在开发和生产环境中，需要追踪 LLM 应用的行为：

| 场景 | 需求 |
|------|------|
| **调试** | 查看 LLM 输入输出、工具调用细节 |
| **监控** | 记录 token 用量、响应时间 |
| **告警** | 异常发生时触发通知 |
| **审计** | 记录完整的执行链路 |
| **计费** | 统计各用户的 API 消耗 |

回调系统是 LangChain 提供的**可观测性基础设施**。

## 17.2 回调的触发时机

LangChain 定义了丰富的回调事件：

```
┌─────────────────────────────────────────────────────┐
│  on_llm_start                                        │
│      ↓                                               │
│  on_llm_stream (多次)                                │
│      ↓                                               │
│  on_llm_end                                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  on_chain_start                                      │
│      ↓                                               │
│  [内部组件的回调]                                     │
│      ↓                                               │
│  on_chain_end / on_chain_error                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  on_tool_start                                       │
│      ↓                                               │
│  on_tool_end / on_tool_error                         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  on_retriever_start                                  │
│      ↓                                               │
│  on_retriever_end                                    │
└─────────────────────────────────────────────────────┘
```

## 17.3 BaseCallbackHandler

所有回调处理器继承自 `BaseCallbackHandler`：

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器"""

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        **kwargs,
    ) -> None:
        """LLM 开始调用时触发"""
        print(f"[LLM 开始] prompts: {prompts}")

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM 结束时触发"""
        print(f"[LLM 结束] response: {response}")

    def on_llm_error(self, error, **kwargs) -> None:
        """LLM 出错时触发"""
        print(f"[LLM 错误] {error}")

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        """Chain 开始时触发"""
        print(f"[Chain 开始] inputs: {inputs}")

    def on_chain_end(self, outputs, **kwargs) -> None:
        """Chain 结束时触发"""
        print(f"[Chain 结束] outputs: {outputs}")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """工具开始时触发"""
        print(f"[工具开始] {serialized.get('name')}: {input_str}")

    def on_tool_end(self, output, **kwargs) -> None:
        """工具结束时触发"""
        print(f"[工具结束] output: {output}")

    def on_retriever_start(self, serialized, query, **kwargs) -> None:
        """检索开始时触发"""
        print(f"[检索开始] query: {query}")

    def on_retriever_end(self, documents, **kwargs) -> None:
        """检索结束时触发"""
        print(f"[检索结束] documents: {len(documents)}")
```

## 17.4 使用回调的三种方式

### 方式一：构造时传入

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[MyCallbackHandler()],
)

# 所有调用都会触发回调
response = llm.invoke("你好")
```

### 方式二：调用时传入

```python
llm = ChatOpenAI(model="gpt-4o-mini")

# 仅这次调用使用回调
response = llm.invoke(
    "你好",
    callbacks=[MyCallbackHandler()],
)
```

### 方式三：全局配置

```python
import os

# 通过环境变量全局启用 LangSmith 回调
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"

# 之后所有调用自动记录到 LangSmith
```

## 17.5 实用回调示例

### Token 计数回调

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenCounterCallback(BaseCallbackHandler):
    """统计 token 使用量"""

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response, **kwargs) -> None:
        """每次 LLM 结束时累计 token"""
        self.call_count += 1

        # 从 response_metadata 获取 token 信息
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)

    def get_summary(self) -> dict:
        """获取统计摘要"""
        return {
            "total_calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        """估算成本（gpt-4o-mini 价格）"""
        # 参考: $0.15/1M input, $0.60/1M output
        input_cost = self.prompt_tokens * 0.15 / 1_000_000
        output_cost = self.completion_tokens * 0.60 / 1_000_000
        return input_cost + output_cost


# 使用
counter = TokenCounterCallback()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[counter])

# 执行多次调用
for i in range(5):
    llm.invoke(f"问题{i}: 什么是机器学习？")

# 查看统计
print(counter.get_summary())
```

### 响应时间追踪

```python
import time

class TimingCallback(BaseCallbackHandler):
    """追踪各组件的响应时间"""

    def __init__(self):
        self.timings = {}
        self._start_times = {}

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._start_times['llm'] = time.time()

    def on_llm_end(self, response, **kwargs) -> None:
        elapsed = time.time() - self._start_times.get('llm', time.time())
        self.timings.setdefault('llm', []).append(elapsed)

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        tool_name = serialized.get('name', 'unknown')
        self._start_times[f'tool_{tool_name}'] = time.time()

    def on_tool_end(self, output, serialized=None, **kwargs) -> None:
        tool_name = serialized.get('name', 'unknown') if serialized else 'unknown'
        elapsed = time.time() - self._start_times.get(f'tool_{tool_name}', time.time())
        self.timings.setdefault(f'tool_{tool_name}', []).append(elapsed)

    def get_report(self) -> str:
        """生成时间报告"""
        report = "响应时间报告:\n"
        for component, times in self.timings.items():
            avg = sum(times) / len(times)
            report += f"  {component}: 平均 {avg:.2f}s (共 {len(times)} 次)\n"
        return report


# 使用
timer = TimingCallback()
chain = prompt | llm | parser

for _ in range(3):
    chain.invoke({"topic": "AI"})

print(timer.get_report())
```

### 日志记录回调

```python
import logging
import json
from datetime import datetime

class LoggingCallback(BaseCallbackHandler):
    """将执行过程记录到日志文件"""

    def __init__(self, log_file: str = "langchain_execution.log"):
        self.logger = logging.getLogger("langchain_callback")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )
        self.logger.addHandler(handler)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.logger.info(json.dumps({
            "event": "llm_start",
            "model": serialized.get("id", ["unknown"])[-1],
            "prompts": prompts[:100],  # 截断
        }, ensure_ascii=False))

    def on_llm_end(self, response, **kwargs) -> None:
        self.logger.info(json.dumps({
            "event": "llm_end",
            "content": response.content[:200] if hasattr(response, 'content') else str(response)[:200],
        }, ensure_ascii=False))

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.logger.info(json.dumps({
            "event": "tool_start",
            "tool": serialized.get("name", "unknown"),
            "input": str(input_str)[:200],
        }, ensure_ascii=False))

    def on_tool_end(self, output, **kwargs) -> None:
        self.logger.info(json.dumps({
            "event": "tool_end",
            "output": str(output)[:200],
        }, ensure_ascii=False))
```

### 异常告警回调

```python
class AlertingCallback(BaseCallbackHandler):
    """出错时发送告警"""

    def __init__(self, alert_function=None):
        self.alert_function = alert_function or self._default_alert

    def _default_alert(self, message: str):
        """默认告警方式（打印）"""
        print(f"⚠️ ALERT: {message}")

    def on_llm_error(self, error, **kwargs) -> None:
        self.alert_function(f"LLM 错误: {error}")

    def on_tool_error(self, error, **kwargs) -> None:
        self.alert_function(f"工具错误: {error}")

    def on_chain_error(self, error, **kwargs) -> None:
        self.alert_function(f"Chain 错误: {error}")


# 自定义告警函数（如发送邮件、调用 webhook）
def send_alert(message: str):
    """发送告警到外部系统"""
    import httpx
    # 示例：调用 webhook
    httpx.post(
        "https://your-alert-webhook.com/alert",
        json={"message": message},
        timeout=5.0,
    )

alert_callback = AlertingCallback(alert_function=send_alert)
```

## 17.6 流式输出的回调

流式输出时，回调会多次触发：

```python
class StreamingCallback(BaseCallbackHandler):
    """处理流式输出的回调"""

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        print("\n[开始生成]")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """每次生成新 token 时触发"""
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs) -> None:
        print("\n[生成完成]")

# 使用
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[StreamingCallback()])

for chunk in llm.stream("写一首短诗"):
    pass  # 回调自动处理打印
```

## 17.7 在 LangGraph 中使用回调

```python
from langgraph.graph import StateGraph, START, END

def my_node(state):
    # 节点内部也可以使用回调
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        callbacks=[TokenCounterCallback()],
    )
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 全局回调：通过 invoke 的 config 传入
result = graph.invoke(
    {"messages": [("user", "你好")]},
    config={"callbacks": [MyCallbackHandler()]},
)
```

## 17.8 回调的异步版本

异步回调处理器继承 `AsyncCallbackHandler`：

```python
from langchain_core.callbacks import AsyncCallbackHandler
import asyncio

class AsyncLoggingCallback(AsyncCallbackHandler):
    """异步回调处理器"""

    async def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        # 异步操作，如写入数据库
        await asyncio.sleep(0)  # 模拟异步操作
        print(f"[Async LLM 开始] {prompts}")

    async def on_llm_end(self, response, **kwargs) -> None:
        await self._save_to_db(response)

    async def _save_to_db(self, response):
        """异步保存到数据库"""
        # 实际实现
        pass


# 使用
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[AsyncLoggingCallback()])

# 异步调用
async def main():
    response = await llm.ainvoke("你好")

asyncio.run(main())
```

## 17.9 回调 vs LangSmith

| 维度 | 自定义回调 | LangSmith |
|------|----------|-----------|
| 配置复杂度 | 需编写代码 | 环境变量即可 |
| 数据存储 | 自己管理 | 云端自动 |
| 可视化 | 无 | 完整 UI |
| 成本分析 | 手动实现 | 自动 |
| 适用场景 | 定制需求、实时告警 | 开发调试、生产监控 |

**推荐策略**：
- 开发阶段：使用 LangSmith 快速调试
- 生产环境：自定义回调 + LangSmith 组合

## 17.10 回调设计最佳实践

### 1. 避免阻塞主流程

```python
# ❌ 错误：回调中执行耗时操作
class SlowCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        time.sleep(5)  # 阻塞！
        save_to_db(response)

# ✅ 正确：使用异步或后台线程
class FastCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        threading.Thread(target=save_to_db, args=[response]).start()
```

### 2. 处理回调本身的异常

```python
class RobustCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        try:
            self._process(response)
        except Exception as e:
            # 回调失败不应影响主流程
            logging.error(f"回调处理失败: {e}")

    def _process(self, response):
        # 处理逻辑
        pass
```

### 3. 合理截断数据

```python
class SafeCallback(BaseCallbackHandler):
    MAX_LENGTH = 500  # 防止日志过大

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        truncated = [p[:self.MAX_LENGTH] for p in prompts]
        self.logger.info(f"prompts: {truncated}")
```

## 17.11 本章小结

- `BaseCallbackHandler` 是回调系统的核心基类
- 回调可监控 LLM、Chain、Tool、Retriever 等所有组件
- 三种使用方式：构造时传入、调用时传入、全局配置
- 常用回调：Token计数、时间追踪、日志记录、异常告警
- 流式输出有 `on_llm_new_token` 事件
- 异步场景使用 `AsyncCallbackHandler`
- 回调失败不应阻塞主流程
- 生产环境建议自定义回调 + LangSmith 组合使用