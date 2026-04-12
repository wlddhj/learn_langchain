# 第14章：LangGraph 进阶

## 14.1 状态持久化的意义

普通 Graph 每次调用都是全新的，没有记忆。持久化解决的核心问题：

| 场景 | 无持久化 | 有持久化 |
|------|---------|---------|
| 多轮对话 | 每次都忘 | 自动记住上下文 |
| 人机交互 | 无法暂停 | 暂停等待人工确认 |
| 错误恢复 | 从头重跑 | 从断点继续 |
| 时间旅行 | 无法回溯 | 回到任意检查点 |

### 检查点存储选项

| 存储方式 | 适用场景 | 持久性 |
|---------|---------|--------|
| `MemorySaver` | 开发测试 | 内存，进程退出即丢失 |
| `SqliteSaver` | 单机部署 | 磁盘文件 |
| `PostgresSaver` | 生产环境 | 数据库，高可用 |

## 14.2 状态持久化 (Persistence)

让 Graph 可以暂停和恢复执行，支持长时间运行的工作流：

### MemorySaver（内存检查点）

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点存储
checkpointer = MemorySaver()

# 编译时添加 checkpointer
graph = builder.compile(checkpointer=checkpointer)

# 使用 thread_id 隔离不同的对话/工作流
config = {"configurable": {"thread_id": "conversation-1"}}

# 第一轮
result1 = graph.invoke(
    {"messages": [("user", "我叫小明")]},
    config=config,
)

# 第二轮（同一个 thread_id，自动加载之前的状态）
result2 = graph.invoke(
    {"messages": [("user", "我叫什么？")]},
    config=config,
)
print(result2["messages"][-1].content)  # "你叫小明"
```

### SQLite 持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 使用 SQLite 存储检查点（持久化到磁盘）
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    result = graph.invoke(
        {"messages": [("user", "你好")]},
        config={"configurable": {"thread_id": "user-1"}},
    )
```

## 14.3 人机交互 (Human-in-the-Loop)

### 为什么需要人机交互

Agent 全自动运行有风险——特别是在以下场景：

- **敏感操作**：发送邮件、删除数据、执行交易
- **质量把关**：AI 生成的代码/文案需要人工审核
- **关键决策**：多方案选择时需要人类判断
- **合规要求**：某些操作必须有审批记录

在关键决策点暂停，等待人工确认后继续：

### interrupt_before vs interrupt_after

| 参数 | 暂停时机 | 适用场景 |
|------|---------|---------|
| `interrupt_before=["execute"]` | 执行节点之前 | 审批敏感操作 |
| `interrupt_after=["analyze"]` | 执行节点之后 | 确认分析结果 |

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# 在执行 "execute" 节点之前暂停，等待人工确认
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute"],  # 在 execute 节点前暂停
    # interrupt_after=["analyze"],  # 或在某个节点后暂停
)

# 第一次调用：执行到 "execute" 之前暂停
result = graph.invoke(
    {"task": "删除数据库中的过期数据"},
    config={"configurable": {"thread_id": "task-1"}},
)

# 查看当前状态
state = graph.get_state({"configurable": {"thread_id": "task-1"}})
print(state.values)      # 当前 state 的值
print(state.next)        # 下一个要执行的节点: ("execute",)

# 人工确认后继续执行
result = graph.invoke(
    None,  # 不需要新输入，使用之前的状态
    config={"configurable": {"thread_id": "task-1"}},
)
```

### 人工审批模式

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    email_draft: str
    recipient: str
    approved: bool
    sent: bool

def draft_email(state: ApprovalState):
    draft = llm.invoke(f"写一封关于{state.get('topic', '项目更新')}的邮件").content
    return {"email_draft": draft}

def review_email(state: ApprovalState):
    # 这个节点会被跳过（interrupt_before），用于人工审批
    pass

def send_email(state: ApprovalState):
    if state.get("approved"):
        # 实际发送邮件
        return {"sent": True}
    return {"sent": False}

builder = StateGraph(ApprovalState)
builder.add_node("draft", draft_email)
builder.add_node("review", review_email)
builder.add_node("send", send_email)

builder.add_edge(START, "draft")
builder.add_edge("draft", "review")
builder.add_edge("review", "send")
builder.add_edge("send", END)

checkpointer = MemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["review"],  # 在审批前暂停
)

# 使用
config = {"configurable": {"thread_id": "email-1"}}
result = graph.invoke({"topic": "项目进度更新"}, config)

# 人工查看草稿
state = graph.get_state(config)
print("邮件草稿:", state.values.get("email_draft"))

# 人工批准
graph.invoke({"approved": True}, config)
```

## 14.4 子图 (Subgraph)

### 什么时候用子图

- **复用**：同一个工作流在多个主图中使用（如"搜索+分析"模块）
- **封装**：把复杂流程封装为一个黑盒节点
- **测试**：子图可以独立编译和测试
- **团队协作**：不同成员开发不同的子图

将复杂流程拆分为可复用的子图：

```python
from langgraph.graph import StateGraph, START, END

# 子图：研究模块
class ResearchState(TypedDict):
    query: str
    findings: str

def web_search(state: ResearchState):
    # 搜索逻辑
    return {"findings": f"搜索 {state['query']} 的结果..."}

def analyze(state: ResearchState):
    return {"findings": f"分析: {state['findings']}"}

research_builder = StateGraph(ResearchState)
research_builder.add_node("search", web_search)
research_builder.add_node("analyze", analyze)
research_builder.add_edge(START, "search")
research_builder.add_edge("search", "analyze")
research_builder.add_edge("analyze", END)

research_graph = research_builder.compile()

# 子图：写作模块
class WritingState(TypedDict):
    topic: str
    findings: str
    article: str

def outline(state: WritingState):
    return {"article": f"大纲: {state['topic']}"}

def write_content(state: WritingState):
    return {"article": f"基于 {state.get('findings', '')} 撰写文章"}

writing_builder = StateGraph(WritingState)
writing_builder.add_node("outline", outline)
writing_builder.add_node("write", write_content)
writing_builder.add_edge(START, "outline")
writing_builder.add_edge("outline", "write")
writing_builder.add_edge("write", END)

writing_graph = writing_builder.compile()

# 在主图中使用子图
class MainState(TypedDict):
    topic: str
    findings: str
    article: str

main_builder = StateGraph(MainState)
main_builder.add_node("research", research_graph)  # 添加子图作为节点
main_builder.add_node("write", writing_graph)       # 添加另一个子图

main_builder.add_edge(START, "research")
main_builder.add_edge("research", "write")
main_builder.add_edge("write", END)

main_graph = main_builder.compile()
```

## 14.5 并行执行

### 并行执行的原理

当多个节点从同一个节点分叉出去时，LangGraph 会并行执行它们：

```
       ┌── node_a ──┐
START ─┤            ├── combine → END
       └── node_b ──┘
```

注意：合并节点（combine）会**等待所有并行节点完成后**才执行。

```python
from langgraph.graph import StateGraph, START, END
import operator
from typing import Annotated

class ParallelState(TypedDict):
    question: str
    web_results: str
    db_results: str
    final_answer: str

def web_search(state: ParallelState):
    # 搜索网络
    return {"web_results": "网络搜索结果..."}

def db_query(state: ParallelState):
    # 查询数据库
    return {"db_results": "数据库查询结果..."}

def combine(state: ParallelState):
    # 合并结果
    return {"final_answer": f"{state['web_results']}\n{state['db_results']}"}

builder = StateGraph(ParallelState)
builder.add_node("web", web_search)
builder.add_node("db", db_query)
builder.add_node("combine", combine)

# web 和 db 并行执行（都从 START 开始）
builder.add_edge(START, "web")
builder.add_edge(START, "db")     # 并行！

# 两者都完成后执行 combine
builder.add_edge("web", "combine")
builder.add_edge("db", "combine")
builder.add_edge("combine", END)

graph = builder.compile()
```

## 14.6 错误恢复

### 方式一：节点内部重试

```python
import time
from langgraph.graph import StateGraph, START, END

class RobustState(TypedDict):
    task: str
    result: str
    error: str

def call_api_with_retry(url: str, max_retries: int = 3) -> dict:
    """带重试的 API 调用"""
    import httpx

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except (httpx.TimeoutException, httpx.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 指数退避: 1s, 2s, 4s
            time.sleep(wait)

def robust_node(state: RobustState):
    """带错误恢复的节点"""
    try:
        data = call_api_with_retry(f"https://api.example.com/{state['task']}")
        return {"result": str(data), "error": ""}
    except Exception as e:
        return {"result": "", "error": f"任务失败: {e}"}
```

### 方式二：LangGraph 级别的重试

```python
from langgraph.graph import StateGraph, START, END

# 构建图时，可以给节点包装重试逻辑
def build_robust_graph():
    builder = StateGraph(RobustState)

    # 正常处理节点
    def process(state: RobustState):
        if state.get("error"):
            return {"result": "已跳过失败的任务"}
        return {"result": f"处理完成: {state['task']}"}

    # 错误处理节点
    def handle_error(state: RobustState):
        if state.get("error"):
            print(f"捕获错误: {state['error']}")
            # 可以记录日志、发送告警、降级处理
            return {"result": "降级处理完成", "error": ""}
        return {}

    def route_by_error(state: RobustState):
        if state.get("error"):
            return "handle_error"
        return "process"

    builder.add_node("process", process)
    builder.add_node("handle_error", handle_error)

    builder.add_edge(START, "process")
    builder.add_conditional_edges("process", route_by_error)
    builder.add_edge("handle_error", END)

    return builder.compile()
```

### 方式三：使用 Checkpointer 恢复中断的执行

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "task-1"}}

# 执行中断后，可以从检查点恢复
try:
    result = graph.invoke({"task": "data_processing"}, config)
except Exception as e:
    print(f"执行中断: {e}")

    # 查看中断时的状态
    state = graph.get_state(config)
    print(f"下一步要执行的节点: {state.next}")
    print(f"当前状态: {state.values}")

    # 修复问题后，继续执行
    result = graph.invoke(None, config)  # 传 None 使用已保存的状态
```

## 14.7 动态图修改

运行时修改图的执行流程：

```python
def should_use_tools(state):
    """动态决定是否使用工具"""
    messages = state["messages"]
    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # 可以根据消息内容动态路由
    content = last.content.lower()
    if "搜索" in content or "查找" in content:
        return "search"

    return END
```

## 14.8 时间旅行调试

利用检查点回溯到之前的状态：

```python
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "debug-1"}}

# 执行几步
graph.invoke({"messages": [("user", "step 1")]}, config)
graph.invoke({"messages": [("user", "step 2")]}, config)

# 查看所有检查点
state_history = list(graph.get_state_history(config))
for state in state_history:
    print(f"检查点: {state.config}")
    print(f"下一步: {state.next}")
    print("---")

# 回到之前的检查点
old_state = state_history[-1]  # 最早的检查点
graph.invoke(None, old_state.config)  # 从该点继续
```

## 14.9 本章小结

- **状态持久化**：使用 `MemorySaver` 或 SQLite 保存执行状态
- **人机交互**：`interrupt_before/after` 在关键节点暂停，等待人工确认
- **子图**：将复杂流程拆分为可复用的子模块
- **并行执行**：多个节点从同一节点分叉实现并行
- **时间旅行**：利用检查点回溯和重放执行过程
- 这些高级特性让 LangGraph 能构建生产级的复杂 Agent 系统
