# 第11章：Agent 执行控制与调试

> **说明**：旧版 LangChain 使用 `AgentExecutor` 类来管理 Agent 执行循环。当前最佳实践是使用 LangGraph（第13章），它提供了更精细的执行控制。本章基于 LangGraph 讲解 Agent 的执行控制、调试和优化技巧。

## 11.1 为什么调试 Agent 很重要

Agent 是非确定性的——同样的输入可能产生不同的执行路径。这带来挑战：

| 问题 | 原因 | 后果 |
|------|------|------|
| 选错工具 | 工具描述不够清晰 | 结果错误 |
| 参数错误 | LLM 理解偏差 | 工具执行失败 |
| 无限循环 | LLM 反复调用工具 | 超时/超预算 |
| 成本失控 | 每次 LLM 调用都消耗 token | 费用激增 |

因此，**系统化的调试和监控能力是 Agent 生产化的关键**。

### 三种调试手段对比

| 手段 | 适用场景 | 详细程度 |
|------|---------|---------|
| `stream()` | 快速查看执行步骤 | 中 |
| `astream_events()` | 详细追踪每个事件 | 高 |
| 自定义 Callback | 生产环境监控 | 可定制 |
| LangSmith | 可视化分析 | 最高 |

## 11.2 Agent 的工作循环

理解 Agent 的执行流程是调试和优化的基础：

```
┌─────────────────────────────────┐
│          用户输入                │
└──────────┬──────────────────────┘
           ↓
┌──────────────────────────────────┐
│  Step 1: LLM 分析，决定是否调用工具  │
│  (如果不需要工具 → 直接回答 → 结束)  │
└──────────┬───────────────────────┘
           ↓ 调用工具
┌──────────────────────────────────┐
│  Step 2: 执行工具，获取结果        │
└──────────┬───────────────────────┘
           ↓
┌──────────────────────────────────┐
│  Step 3: LLM 分析工具结果         │
│  (如果还需要更多信息 → 回到 Step 1) │
│  (如果信息足够 → 生成最终回答)      │
└──────────────────────────────────┘
```

## 11.3 使用 LangGraph 的执行控制

### 设置最大步数

```python
# 限制 agent 的最大循环次数
result = agent.invoke(
    {"messages": [("user", "帮我查天气")]},
    config={"recursion_limit": 5},  # 最多执行5步
)
```

### 提前停止

```python
from langgraph.prebuilt import create_react_agent

# 通过回调在特定条件下停止
def should_continue(state):
    messages = state["messages"]
    tool_calls = messages[-1].tool_calls if hasattr(messages[-1], "tool_calls") else []
    if len(messages) > 10:  # 消息太多时强制停止
        return "end"
    return "continue"
```

## 11.4 流式输出与调试

### 事件流 (Stream Events)

```python
# 详细的执行过程追踪
async for event in agent.astream_events(
    {"messages": [("user", "北京天气怎么样？")]},
    version="v2",
):
    kind = event["event"]

    if kind == "on_chat_model_start":
        print("LLM 开始思考...")

    elif kind == "on_chat_model_stream":
        # LLM 的流式输出
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)

    elif kind == "on_tool_start":
        print(f"\n🔧 调用工具: {event['name']}")
        print(f"   参数: {event['data'].get('input')}")

    elif kind == "on_tool_end":
        print(f"   结果: {event['data'].get('output', '')[:100]}")
```

### 普通流式

```python
for event in agent.stream({"messages": [("user", "北京天气怎么样？")]}):
    for node_name, node_output in event.items():
        print(f"\n--- 节点: {node_name} ---")
        if "messages" in node_output:
            for msg in node_output["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  工具调用: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"  内容: {msg.content[:100]}")
```

## 11.5 错误处理

### 工具执行错误

```python
@tool
def risky_tool(query: str) -> str:
    """可能失败的工具"""
    try:
        result = call_external_api(query)
        return result
    except ConnectionError:
        return "错误：网络连接失败，请稍后重试"
    except TimeoutError:
        return "错误：请求超时，请稍后重试"
    except Exception as e:
        return f"错误：{type(e).__name__} - {str(e)}"
```

> **关键**：工具内部捕获异常并返回描述性文本，而不是让异常传播。这样 Agent 能理解错误并尝试其他方案。

### 错误处理的三个层次

| 层次 | 在哪处理 | 策略 |
|------|---------|------|
| **工具级** | 工具函数内部 | try/except 返回错误文本 |
| **Agent 级** | Agent 调用包装器 | 重试机制、超时控制 |
| **Graph 级** | LangGraph 条件路由 | 错误节点、降级处理 |

经验法则：**尽量在工具级处理**，因为 Agent 能理解错误信息并自主调整策略。

### Agent 级别的错误处理

```python
from langchain_core.messages import HumanMessage
import json

def run_agent_safely(agent, user_input: str, max_retries: int = 3):
    """安全的 Agent 执行包装"""
    for attempt in range(max_retries):
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 10},
            )
            return result["messages"][-1].content

        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {e}")
            if attempt == max_retries - 1:
                return "抱歉，处理您的请求时遇到了问题，请稍后重试。"

    return "抱歉，多次尝试均失败。"
```

## 11.6 使用 LangSmith 调试

LangSmith 是调试 Agent 的最佳工具：

```python
import os

# 配置 LangSmith
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# 之后所有 agent 调用都会自动记录到 LangSmith
result = agent.invoke({"messages": [("user", "帮我查天气")]})

# 在 LangSmith 中可以看到：
# - 完整的 LLM 输入输出
# - 每次工具调用的参数和结果
# - Token 用量和延迟
# - 执行流程图
```

## 11.7 Agent 性能优化

### 减少工具调用次数

```python
# 1. 提供更详细的工具描述
@tool
def search_user(user_id: str) -> str:
    """根据用户ID查询用户的完整信息，包括姓名、邮箱、注册时间、会员等级。
    当需要任何用户相关信息时使用此工具，一次查询获取所有信息。"""
    pass

# 2. 工具返回更多信息
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    # 返回完整信息，避免 Agent 需要再次查询
    return f"""
城市: {city}
温度: 25°C
天气: 晴天
湿度: 60%
风速: 3级
建议: 适合户外活动
"""
```

### 使用更快的模型

```python
# 简单任务用小模型，复杂任务用大模型
from langchain_openai import ChatOpenAI

fast_model = ChatOpenAI(model="gpt-4o-mini")    # 快速
smart_model = ChatOpenAI(model="gpt-4o")         # 强大

# 根据任务复杂度选择
agent = create_react_agent(
    model=fast_model,  # 大多数场景 mini 就够了
    tools=[...],
)
```

## 11.8 Agent 的可观测性

### 记录 Token 用量

```python
def analyze_agent_run(result):
    """分析 Agent 运行的资源消耗"""
    messages = result["messages"]
    tool_calls = 0
    total_content = 0

    for msg in messages:
        total_content += len(msg.content) if msg.content else 0
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls += len(msg.tool_calls)

    print(f"总消息数: {len(messages)}")
    print(f"工具调用次数: {tool_calls}")
    print(f"总内容长度: {total_content} 字符")
```

### 自定义回调

```python
from langchain_core.callbacks import BaseCallbackHandler

class AgentCallbackHandler(BaseCallbackHandler):
    """自定义回调，记录 Agent 行为"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("🤖 LLM 开始思考...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 调用工具: {serialized.get('name')}")
        print(f"   输入: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"   输出: {str(output)[:100]}")

    def on_llm_end(self, response, **kwargs):
        print("🤖 LLM 思考完成")

# 使用回调
result = agent.invoke(
    {"messages": [("user", "帮我查天气")]},
    config={"callbacks": [AgentCallbackHandler()]},
)
```

## 11.9 本章小结

- Agent 执行是一个"LLM 分析 → 工具调用 → 结果观察"的循环
- 使用 `recursion_limit` 防止无限循环
- 工具内部应捕获异常，返回描述性错误信息
- 使用 `astream_events` 详细追踪执行过程
- LangSmith 是调试 Agent 的最佳工具
- 优化工具描述和返回内容可减少调用次数
