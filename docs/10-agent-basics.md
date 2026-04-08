# 第10章：Agent 基础

## 10.1 什么是 Agent

Agent = LLM + 工具 + 自主决策循环

```
普通 LLM 调用：  用户 → LLM → 回答
Agent 调用：     用户 → LLM → 思考 → 选择工具 → 执行 → 观察结果 → 继续思考或回答
```

**核心区别**：Agent 能自主决定下一步做什么，而不是一次性给出答案。

## 10.2 ReAct 模式（Reasoning + Acting）

最常见的 Agent 模式，循环执行：

```
Thought: 我需要查询北京的天气
Action: search_weather(city="北京")
Observation: 晴天，25°C

Thought: 我已经获取了天气信息，可以回答了
Answer: 北京今天是晴天，气温25°C
```

## 10.3 使用 create_react_agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. 定义工具
@tool
def search_weather(city: str) -> str:
    """查询城市天气"""
    weather = {"北京": "晴天 25°C", "上海": "多云 22°C"}
    return weather.get(city, "未找到天气信息")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，如 '2 + 3 * 4'"""
    try:
        result = eval(expression)  # 仅用于演示，生产环境不要用 eval
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# 2. 创建 Agent
model = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(
    model=model,
    tools=[search_weather, calculator],
)

# 3. 运行 Agent
result = agent.invoke({
    "messages": [("user", "北京天气怎么样？如果温度是25度，那25*2是多少？")]
})

# 查看完整执行过程
for msg in result["messages"]:
    print(f"[{msg.__class__.__name__}] {msg.content}")
```

### 输出解析

```python
# 获取最终回答
final_message = result["messages"][-1]
print(final_message.content)

# 查看工具调用过程
from langchain_core.messages import ToolMessage, AIMessage

for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"调用工具: {tc['name']}({tc['args']})")
    elif isinstance(msg, ToolMessage):
        print(f"工具结果: {msg.content}")
```

## 10.4 流式输出

```python
# 流式查看 Agent 的思考过程
for event in agent.stream({"messages": [("user", "北京天气怎么样？")]}):
    for key, value in event.items():
        if key == "agent":
            print(f"Agent 思考: {value['messages'][-1].content[:100]}")
        elif key == "tools":
            print(f"工具结果: {value['messages'][-1].content}")
```

## 10.5 带 System Prompt 的 Agent

```python
agent = create_react_agent(
    model=model,
    tools=[search_weather, calculator],
    prompt="你是一个智能助手。请用中文回答。回答要简洁明了。",
)

# 更复杂的 system prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """你是一个客服助手。请遵循以下规则：

1. 始终使用中文回答
2. 如果需要计算，使用 calculator 工具
3. 如果查询天气，使用 search_weather 工具
4. 无法回答的问题，诚实说明
5. 回答要简洁，不超过100字
"""
```

## 10.6 Agent 的状态结构

`create_react_agent` 返回的 agent 接收和返回一个 state 字典：

```python
# 输入 state
input_state = {
    "messages": [
        ("user", "你好"),
        # 或者
        HumanMessage(content="你好"),
    ]
}

# 输出 state
output_state = {
    "messages": [
        HumanMessage(content="你好"),
        AIMessage(content=""),  # 工具调用决策
        ToolMessage(content="..."),  # 工具结果
        AIMessage(content="你好！有什么可以帮你的？"),  # 最终回答
    ]
}
```

## 10.7 Agent 类型对比

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| **ReAct** | 思考→行动→观察循环 | 通用场景（推荐） |
| **OpenAI Tools** | 使用 OpenAI function calling | OpenAI 模型 |
| **Structured Chat** | 处理多输入工具 | 复杂工具参数 |

> **推荐**：使用 LangGraph 的 `create_react_agent`，它是当前的最佳实践。

## 10.8 多轮对话 Agent

```python
# Agent 天然支持多轮对话，只需追加消息
messages = []

# 第一轮
messages.append(("user", "北京天气怎么样？"))
result = agent.invoke({"messages": messages})
messages = result["messages"]

# 第二轮（Agent 能记住上下文）
messages.append(("user", "那上海呢？"))
result = agent.invoke({"messages": messages})
messages = result["messages"]

# 获取最终回答
print(messages[-1].content)
```

## 10.9 Agent 的局限性

1. **可靠性**：LLM 可能选择错误的工具或传错参数
2. **成本**：每次循环都消耗 token
3. **延迟**：多次工具调用增加响应时间
4. **无限循环**：可能陷入死循环（需要设置最大步数）

### 安全设置

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=model,
    tools=[search_weather, calculator],
)

# 通过 recursion_limit 限制最大步数
result = agent.invoke(
    {"messages": [("user", "帮我算个数学题")]},
    config={"recursion_limit": 10},  # 最多10步
)
```

## 10.10 完整示例：智能问答 Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_knowledge_base(query: str) -> str:
    """搜索内部知识库"""
    kb = {
        "退货政策": "购买后7天内可无条件退货，需保持商品完好。",
        "配送时间": "标准配送3-5个工作日，加急配送1-2个工作日。",
        "会员优惠": "金卡会员享9折，钻石会员享8折。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return "未找到相关信息"

@tool
def query_order_status(order_id: str) -> str:
    """查询订单状态"""
    orders = {
        "ORD-001": "已发货，预计明天送达",
        "ORD-002": "仓库备货中",
        "ORD-003": "已签收",
    }
    return orders.get(order_id, "订单不存在")

@tool
def calculate_price(price: float, discount: float) -> str:
    """计算折扣后的价格"""
    final_price = price * (1 - discount / 100)
    return f"原价 ¥{price}，{discount}% 折扣后 ¥{final_price:.2f}"

# 创建 Agent
model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(
    model=model,
    tools=[search_knowledge_base, query_order_status, calculate_price],
    prompt="你是一个电商客服助手。请用中文回答，态度友好。",
)

# 测试
result = agent.invoke({
    "messages": [("user", "我的订单ORD-001到哪了？另外问下退货政策")]
})

for msg in result["messages"]:
    msg_type = msg.__class__.__name__
    if msg_type == "AIMessage" and not msg.tool_calls:
        print(f"AI: {msg.content}")
    elif msg_type == "ToolMessage":
        print(f"  ↳ 工具结果: {msg.content[:80]}")
```

## 10.11 本章小结

- Agent = LLM + 工具 + 自主决策循环
- ReAct 是最常见的 Agent 模式：思考→行动→观察
- 使用 LangGraph 的 `create_react_agent` 快速创建 Agent
- Agent 接收消息列表，自动管理工具调用循环
- 注意设置递归限制防止无限循环
- Agent 天然支持多轮对话
