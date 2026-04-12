# 第10章：Agent 基础

## 10.1 什么是 Agent

Agent = LLM + 工具 + 自主决策循环

```
普通 LLM 调用：  用户 → LLM → 回答
Agent 调用：     用户 → LLM → 思考 → 选择工具 → 执行 → 观察结果 → 继续思考或回答
```

**核心区别**：Agent 能自主决定下一步做什么，而不是一次性给出答案。

### Agent vs Chain 的区别

| 维度 | Chain（链） | Agent（代理） |
|------|------------|--------------|
| 执行流程 | 固定，预定义 | 动态，LLM 自主决定 |
| 工具使用 | 不使用或硬编码 | LLM 选择何时用哪个工具 |
| 适用场景 | 明确的输入→输出 | 需要推理和多步骤的复杂任务 |
| 可控性 | 高 | 取决于 LLM 能力 |
| 成本 | 低（一次调用） | 较高（可能多次 LLM + 工具调用） |

**何时用 Chain**：翻译、摘要、格式转换等确定性的任务。
**何时用 Agent**：需要查询数据库+计算+生成报告等不确定性的复合任务。

## 10.2 ReAct 模式（Reasoning + Acting）

最常见的 Agent 模式，循环执行：

```
Thought: 我需要查询北京的天气
Action: search_weather(city="北京")
Observation: 晴天，25°C

Thought: 我已经获取了天气信息，可以回答了
Answer: 北京今天是晴天，气温25°C
```

## 10.3 Agent 的内部状态

`create_react_agent` 创建的 Agent 接收和返回一个 state 字典：

```python
# 输入 state
{"messages": [("user", "你好")]}

# 输出 state（经过一次工具调用后）
{
    "messages": [
        HumanMessage(content="北京天气怎么样？"),    # 用户消息
        AIMessage(tool_calls=[{...}]),             # LLM 决定调用工具
        ToolMessage(content="晴天 25°C"),           # 工具执行结果
        AIMessage(content="北京今天是晴天..."),      # LLM 最终回答
    ]
}
```

关键：**所有上下文都在 `messages` 列表中**，包括用户消息、AI 思考、工具调用和结果。

## 10.4 使用 create_react_agent

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

## 10.5 流式输出

```python
# 流式查看 Agent 的思考过程
for event in agent.stream({"messages": [("user", "北京天气怎么样？")]}):
    for key, value in event.items():
        if key == "agent":
            print(f"Agent 思考: {value['messages'][-1].content[:100]}")
        elif key == "tools":
            print(f"工具结果: {value['messages'][-1].content}")
```

## 10.6 带 System Prompt 的 Agent

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

## 10.7 Agent Prompt 设计原则

`create_react_agent` 返回的 agent 接收和返回一个 state 字典：

好的 Agent Prompt 应包含：

1. **角色定义**：你是谁（如"你是电商客服"）
2. **行为规则**：何时用哪个工具
3. **回答风格**：语言、长度、语气
4. **边界条件**：无法回答时怎么办

```python
# 好的 prompt 示例
"""你是「智选商城」客服。规则：
1. 用中文回答，态度友好
2. 查订单 → query_order 工具
3. 查商品 → search_product 工具
4. 不确定时诚实说明，建议联系人工客服"""

# 差的 prompt 示例
"""你是助手。"""  # 太模糊，Agent 不知道何时用工具
```

## 10.8 Agent 类型对比

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| **ReAct** | 思考→行动→观察循环 | 通用场景（推荐） |
| **OpenAI Tools** | 使用 OpenAI function calling | OpenAI 模型 |
| **Structured Chat** | 处理多输入工具 | 复杂工具参数 |

> **推荐**：使用 LangGraph 的 `create_react_agent`，它是当前的最佳实践。

## 10.9 多轮对话 Agent

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

## 10.10 Agent 的局限性与应对

| 问题 | 说明 | 应对策略 |
|------|------|---------|
| 可靠性 | LLM 可能选错工具或传错参数 | 详细描述 + 参数类型注解 |
| 成本 | 每次循环都消耗 token | 优化工具描述，减少不必要调用 |
| 延迟 | 多次工具调用增加响应时间 | 用快速模型，减少工具数量 |
| 无限循环 | 可能陷入死循环 | 设置 `recursion_limit` |
| 幻觉 | 可能编造不存在的工具 | 限制可用工具，明确边界 |

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

## 10.11 完整示例：智能问答 Agent

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

## 10.12 本章小结

- Agent = LLM + 工具 + 自主决策循环
- ReAct 是最常见的 Agent 模式：思考→行动→观察
- 使用 LangGraph 的 `create_react_agent` 快速创建 Agent
- Agent 接收消息列表，自动管理工具调用循环
- 注意设置递归限制防止无限循环
- Agent 天然支持多轮对话
