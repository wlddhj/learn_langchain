# 第15章：多 Agent 系统

## 15.1 为什么需要多 Agent

单 Agent 的局限：
- 工具太多导致 LLM 选择困难
- 单个 LLM 的能力有上限
- 复杂任务难以用单一循环完成

多 Agent 的优势：
- **分工协作**：每个 Agent 专注一个领域
- **专业高效**：每个 Agent 使用定制 prompt 和工具
- **可扩展**：新增 Agent 即可扩展能力

## 15.2 常见多 Agent 模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **Supervisor** | 中心调度器分配任务 | 复杂工作流管理 |
| **Swarm** | Agent 间直接交接 | 灵活协作 |
| **Hierarchical** | 多层级管理 | 大型项目 |

## 15.3 Supervisor 模式

一个"主管"Agent 负责理解用户意图并分配给专业 Agent：

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ===== State =====
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

# ===== 专业 Agent 1：研究 Agent =====
@tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    return f"搜索结果: {query} 的相关信息..."

research_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search_web],
    prompt="你是一个研究助手。负责搜索和分析信息。",
)

def research_node(state: MultiAgentState):
    result = research_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

# ===== 专业 Agent 2：写作 Agent =====
@tool
def write_document(content: str) -> str:
    """撰写文档"""
    return f"已撰写文档: {content[:100]}..."

writing_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[write_document],
    prompt="你是一个写作助手。负责撰写和编辑文档。",
)

def writing_node(state: MultiAgentState):
    result = writing_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

# ===== Supervisor =====
def supervisor(state: MultiAgentState):
    """主管 Agent：决定下一步由谁处理"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke([
        ("system", """你是任务调度器。根据对话内容决定下一步：

- 需要搜索或研究信息 → 回复 "researcher"
- 需要撰写或编辑文档 → 回复 "writer"
- 任务已完成 → 回复 "FINISH"

只回复一个词。"""),
        *state["messages"],
    ])

    decision = response.content.strip().lower()

    if "researcher" in decision:
        return {"next_agent": "researcher"}
    elif "writer" in decision:
        return {"next_agent": "writer"}
    else:
        return {"next_agent": "FINISH"}

def route_to_agent(state: MultiAgentState):
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]

# ===== 构建图 =====
builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("researcher", research_node)
builder.add_node("writer", writing_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_to_agent)
builder.add_edge("researcher", "supervisor")  # 执行完回到 supervisor
builder.add_edge("writer", "supervisor")

graph = builder.compile()

# 运行
result = graph.invoke({
    "messages": [HumanMessage(content="帮我研究AI Agent的最新进展，然后写一份摘要报告")]
})

for msg in result["messages"]:
    if isinstance(msg, AIMessage) and msg.content:
        print(f"AI: {msg.content[:200]}")
```

## 15.4 Swarm 模式

Agent 之间直接交接，无需中心调度：

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 定义交接工具
@tool
def transfer_to_sales():
    """将对话转交给销售 Agent"""
    return "转交成功"

@tool
def transfer_to_support():
    """将对话转交给技术支持 Agent"""
    return "转交成功"

# 销售 Agent
sales_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[transfer_to_support],
    prompt="你是销售顾问。处理购买和定价问题。无法回答技术问题时转给技术支持。",
)

# 技术支持 Agent
support_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[transfer_to_sales],
    prompt="你是技术支持。处理技术问题。涉及购买时转给销售。",
)

# 用 Supervisor 模式实现交接
def router(state):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            if "sales" in tc["name"]:
                return "sales"
            elif "support" in tc["name"]:
                return "support"
    return END
```

## 15.5 层级模式 (Hierarchical)

```
              [Top Supervisor]
              /              \
    [Mid-Supervisor-1]  [Mid-Supervisor-2]
      /          \          /          \
[Agent-A]  [Agent-B]  [Agent-C]  [Agent-D]
```

```python
# 简化的层级模式示例

# 底层 Agent
code_writer = create_react_agent(model, code_tools, prompt="写代码...")
code_reviewer = create_react_agent(model, review_tools, prompt="审查代码...")
test_writer = create_react_agent(model, test_tools, prompt="写测试...")

# 中层 Supervisor：管理代码开发流程
def dev_supervisor(state):
    """管理代码开发：写代码 → 审查 → 写测试"""
    # 决策逻辑
    pass

dev_builder = StateGraph(DevState)
dev_builder.add_node("writer", code_writer_node)
dev_builder.add_node("reviewer", code_reviewer_node)
dev_builder.add_node("tester", test_writer_node)
dev_builder.add_node("supervisor", dev_supervisor)
# ... 添加边

# 顶层 Supervisor：管理整体项目
def top_supervisor(state):
    """决定任务分配给开发团队还是文档团队"""
    pass
```

## 15.6 Agent 间通信模式

### 共享 State

```python
class SharedState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: str       # 研究 Agent 写入
    draft: str                # 写作 Agent 写入
    feedback: str             # 审核 Agent 写入
    status: str               # 当前状态
```

### 消息传递

```python
def agent_a_node(state):
    """Agent A 完成工作后，在消息中指示下一个 Agent"""
    result = agent_a.invoke({"messages": state["messages"]})
    # 添加一条指示消息
    result["messages"].append(
        AIMessage(content="研究完成，请写作助手基于以上信息撰写文章。")
    )
    return {"messages": result["messages"]}
```

## 15.7 完整示例：内容创作团队

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    research: str
    outline: str
    article: str
    feedback: str
    next: str

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. 研究员
def researcher(state: TeamState):
    response = llm.invoke([
        SystemMessage(content="你是研究专家。搜索和分析信息，给出详细的研究笔记。"),
        HumanMessage(content=f"研究以下主题：{state['topic']}"),
    ])
    return {"research": response.content}

# 2. 大纲编辑
def outliner(state: TeamState):
    response = llm.invoke([
        SystemMessage(content="你是大纲编辑。基于研究笔记，创建文章大纲。"),
        HumanMessage(content=f"研究笔记：{state['research']}\n\n请创建文章大纲。"),
    ])
    return {"outline": response.content}

# 3. 写作者
def writer(state: TeamState):
    response = llm.invoke([
        SystemMessage(content="你是写作者。根据大纲和研究笔记撰写完整文章。"),
        HumanMessage(content=f"""大纲：{state['outline']}

研究笔记：{state['research']}

请撰写完整文章。"""),
    ])
    return {"article": response.content}

# 4. 审稿人
def reviewer(state: TeamState):
    response = llm.invoke([
        SystemMessage(content="你是审稿人。审查文章质量，给出反馈。回复 PASS 或 REVISE。"),
        HumanMessage(content=f"审查以下文章：\n{state['article']}"),
    ])
    content = response.content
    decision = "PASS" if "PASS" in content.upper() else "REVISE"
    return {"feedback": content, "next": decision}

# 5. 路由
def review_decision(state: TeamState):
    if state.get("next") == "PASS":
        return END
    return "writer"  # 需要修改，回到写作者

# 构建图
builder = StateGraph(TeamState)
builder.add_node("researcher", researcher)
builder.add_node("outliner", outliner)
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "outliner")
builder.add_edge("outliner", "writer")
builder.add_edge("writer", "reviewer")
builder.add_conditional_edges("reviewer", review_decision)

graph = builder.compile()

# 运行
result = graph.invoke({
    "topic": "AI Agent 的发展趋势",
    "messages": [],
})
print(result["article"])
```

## 15.8 多 Agent 系统设计原则

1. **明确的职责划分**：每个 Agent 有清晰的职责边界
2. **简洁的通信协议**：通过共享 State 或消息传递
3. **避免 Agent 过多**：3-5 个 Agent 通常足够
4. **设置最大循环次数**：防止 Agent 间无限传递
5. **可观测性**：每个 Agent 的输入输出都应该可追踪

## 15.9 本章小结

- **Supervisor 模式**：中心调度器分配任务（最常用）
- **Swarm 模式**：Agent 间直接交接
- **层级模式**：多层级管理，适合大型项目
- Agent 通过共享 State 或消息传递进行通信
- 每个专业 Agent 有自己的 prompt 和工具
- 注意设置循环限制和可观测性
