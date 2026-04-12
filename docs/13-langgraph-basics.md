# 第13章：LangGraph 基础

## 13.1 为什么需要 LangGraph

LangChain 的 Agent 虽然方便，但有局限：
- 执行流程固定（ReAct 循环）
- 难以实现复杂的多步骤工作流
- 缺乏对执行流程的精细控制

LangGraph 提供了**图（Graph）**的方式来定义 Agent 的工作流：

```
LangChain Agent:  LLM → 工具 → LLM → 工具 → ...（固定循环）
LangGraph:       自定义节点 → 自定义边 → 条件路由 → 循环/终止（完全可控）
```

## 13.2 核心概念

| 概念 | 说明 |
|------|------|
| **State** | 图的共享状态，在节点之间传递 |
| **Node** | 节点，执行具体操作的函数 |
| **Edge** | 边，定义节点之间的转移 |
| **Conditional Edge** | 条件边，根据条件选择下一个节点 |

## 13.3 Node 的要求与限制

节点函数必须遵循以下规则：

| 规则 | 说明 |
|------|------|
| **接收 State** | 第一个参数是当前 state（`TypedDict` 类型） |
| **返回 dict** | 返回值必须是 dict，只包含要更新的字段 |
| **只更新部分字段** | 不需要返回所有字段，未返回的字段保持不变 |
| **字段类型匹配** | 返回的值类型必须与 State 定义一致 |
| **名称唯一** | 同一个图中 `add_node` 的名称不能重复 |

```python
class MyState(TypedDict):
    question: str
    answer: str
    score: int

# ✅ 正确：只返回要更新的字段
def my_node(state: MyState):
    response = llm.invoke(state["question"])
    return {"answer": response.content}  # score 不变

# ❌ 错误：返回了非 dict 类型
def bad_node(state: MyState):
    return response.content  # 必须是 dict

# ❌ 错误：返回了 State 中不存在的字段
def bad_node2(state: MyState):
    return {"unknown_field": "值"}  # 字段不在 State 定义中
```

### add_messages reducer

当 State 字段使用 `Annotated[list, add_messages]` 时，返回的列表会**追加**而非**替换**：

```python
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # 自动追加

def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    # 返回的消息会追加到 messages，不会覆盖之前的
    return {"messages": [response]}
```

## 13.4 第一个 StateGraph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. 定义 State（状态）
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，add_messages 自动处理追加

# 2. 定义节点函数
def chatbot(state: State):
    """聊天机器人节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 返回要更新的 state 字段

# 3. 构建图
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("chatbot", chatbot)

# 添加边
graph_builder.add_edge(START, "chatbot")    # 入口 → chatbot
graph_builder.add_edge("chatbot", END)      # chatbot → 结束

# 编译图
graph = graph_builder.compile()

# 4. 运行
result = graph.invoke({"messages": [("user", "你好")]})
print(result["messages"][-1].content)
```

### 可视化图结构

```python
from IPython.display import Image, display

# 生成流程图
try:
    img = Image(graph.get_graph().draw_mermaid_png())
    display(img)
except Exception:
    # 需要 pip install grandalf
    pass
```

## 13.5 带工具的 Graph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 工具
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}: 晴天 25°C"

tools = [get_weather]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# 节点
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: State):
    """判断是否需要继续调用工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 构建图
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))  # 内置工具执行节点

# 添加边
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_continue)
graph_builder.add_edge("tools", "chatbot")  # 工具执行后回到 chatbot

# 编译
graph = graph_builder.compile()

# 运行
result = graph.invoke({"messages": [("user", "北京天气怎么样？")]})
for msg in result["messages"]:
    print(f"[{msg.__class__.__name__}] {msg.content}")
```

### 图的执行流程

```
START → chatbot → should_continue?
                      ↓
              有工具调用? ─── Yes ──→ tools → chatbot（循环）
                      ↓
                      No → END
```

## 13.6 条件路由

根据不同条件路由到不同节点：

```python
from typing import Literal

def route_query(state: State) -> Literal["math_agent", "search_agent", "chat_agent"]:
    """根据问题类型路由到不同的 Agent"""
    last_message = state["messages"][-1]
    content = last_message.content.lower()

    if any(word in content for word in ["计算", "数学", "等于"]):
        return "math_agent"
    elif any(word in content for word in ["搜索", "查找", "查询"]):
        return "search_agent"
    else:
        return "chat_agent"

# 构建图
graph_builder = StateGraph(State)

graph_builder.add_node("router", route_node)        # 路由节点
graph_builder.add_node("math_agent", math_handler)   # 数学处理
graph_builder.add_node("search_agent", search_handler) # 搜索处理
graph_builder.add_node("chat_agent", chat_handler)    # 通用聊天

graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", route_query)
graph_builder.add_edge("math_agent", END)
graph_builder.add_edge("search_agent", END)
graph_builder.add_edge("chat_agent", END)

graph = graph_builder.compile()
```

## 13.7 自定义 State

State 不限于消息，可以包含任意数据：

```python
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    question: str                    # 研究问题
    search_results: list[str]        # 搜索结果
    analysis: str                    # 分析结果
    answer: str                      # 最终答案
    iteration: int                   # 迭代次数
    messages: Annotated[list, add_messages]

# 节点
def search(state: ResearchState):
    results = search_web(state["question"])
    return {
        "search_results": results,
        "iteration": state.get("iteration", 0) + 1,
    }

def analyze(state: ResearchState):
    # 基于搜索结果进行分析
    analysis = llm.invoke(f"分析以下信息：{state['search_results']}")
    return {"analysis": analysis.content}

def generate_answer(state: ResearchState):
    answer = llm.invoke(
        f"基于分析结果回答问题：\n"
        f"问题: {state['question']}\n"
        f"分析: {state['analysis']}"
    )
    return {"answer": answer.content}
```

## 13.8 循环与迭代

实现"研究-评估"循环：

```python
def should_continue_research(state: ResearchState):
    """评估回答质量，决定是否继续研究"""
    if state.get("iteration", 0) >= 3:
        return "generate"  # 最多3轮
    # 可以用 LLM 评估回答质量
    return "search"  # 继续搜索

graph_builder = StateGraph(ResearchState)

graph_builder.add_node("search", search)
graph_builder.add_node("analyze", analyze)
graph_builder.add_node("generate", generate_answer)

graph_builder.add_edge(START, "search")
graph_builder.add_edge("search", "analyze")
graph_builder.add_conditional_edges("analyze", should_continue_research)
graph_builder.add_edge("generate", END)
```

## 13.9 流式输出

```python
# 流式输出最终结果
for event in graph.stream({"messages": [("user", "你好")]}):
    for node, output in event.items():
        print(f"节点 {node}:")
        if "messages" in output:
            for msg in output["messages"]:
                if msg.content:
                    print(f"  {msg.content[:100]}")

# 流式 token
async for event in graph.astream_events(
    {"messages": [("user", "写一首诗")]},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)
```

## 13.10 完整示例：代码审查 Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class CodeReviewState(TypedDict):
    code: str
    language: str
    issues: str
    suggestions: str
    improved_code: str
    review_passed: bool

llm = ChatOpenAI(model="gpt-4o-mini")

def find_issues(state: CodeReviewState):
    """查找代码问题"""
    prompt = ChatPromptTemplate.from_template(
        "分析以下{language}代码的问题：\n```{language}\n{code}\n```"
    )
    chain = prompt | llm
    result = chain.invoke({"language": state["language"], "code": state["code"]})
    return {"issues": result.content}

def suggest_fixes(state: CodeReviewState):
    """给出修复建议"""
    prompt = ChatPromptTemplate.from_template(
        "基于以下问题给出修复建议：\n问题：{issues}\n原代码：{code}"
    )
    chain = prompt | llm
    result = chain.invoke({"issues": state["issues"], "code": state["code"]})
    return {"suggestions": result.content}

def improve_code(state: CodeReviewState):
    """改进代码"""
    prompt = ChatPromptTemplate.from_template(
        "根据建议改进代码：\n原代码：{code}\n建议：{suggestions}"
    )
    chain = prompt | llm
    result = chain.invoke({"code": state["code"], "suggestions": state["suggestions"]})
    return {"improved_code": result.content, "review_passed": True}

# 构建图
builder = StateGraph(CodeReviewState)
builder.add_node("find_issues", find_issues)
builder.add_node("suggest_fixes", suggest_fixes)
builder.add_node("improve_code", improve_code)

builder.add_edge(START, "find_issues")
builder.add_edge("find_issues", "suggest_fixes")
builder.add_edge("suggest_fixes", "improve_code")
builder.add_edge("improve_code", END)

review_graph = builder.compile()

# 运行
result = review_graph.invoke({
    "code": "def add(a,b): return a+b",
    "language": "python",
})
print(result["issues"])
print(result["improved_code"])
```

## 13.11 本章小结

- LangGraph 用**图**的方式定义 Agent 工作流
- 核心概念：State（状态）、Node（节点）、Edge（边）
- `StateGraph` 创建图，`add_node` 添加节点，`add_edge` 添加边
- `add_conditional_edges` 实现条件路由
- `ToolNode` 是内置的工具执行节点
- 自定义 State 可以在节点间传递任意数据
- 支持循环、条件分支等复杂流程
