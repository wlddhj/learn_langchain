# 第14章：LangGraph 进阶

## 14.1 状态持久化 (Persistence)

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

## 14.2 人机交互 (Human-in-the-Loop)

在关键决策点暂停，等待人工确认后继续：

### interrupt_before / interrupt_after

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

## 14.3 子图 (Subgraph)

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

## 14.4 并行执行

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

## 14.5 错误恢复

```python
from langgraph.graph import StateGraph, START, END

def unreliable_node(state):
    """可能失败的节点"""
    import random
    if random.random() < 0.5:
        raise ValueError("随机失败")
    return {"result": "成功"}

class RobustState(TypedDict):
    attempts: int
    result: str
    error: str

def retry_node(state: RobustState):
    """重试节点"""
    attempts = state.get("attempts", 0) + 1
    if attempts >= 3:
        return {"error": "超过最大重试次数"}
    return {"attempts": attempts}

# 使用 checkpointer 的状态来恢复
def run_with_retry(graph, input_data, config, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = graph.invoke(input_data, config)
            return result
        except Exception as e:
            print(f"第 {attempt + 1} 次失败: {e}")
            # 从检查点恢复状态
            state = graph.get_state(config)
            if state:
                input_data = None  # 使用保存的状态继续
    return None
```

## 14.6 动态图修改

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

## 14.7 时间旅行调试

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

## 14.8 本章小结

- **状态持久化**：使用 `MemorySaver` 或 SQLite 保存执行状态
- **人机交互**：`interrupt_before/after` 在关键节点暂停，等待人工确认
- **子图**：将复杂流程拆分为可复用的子模块
- **并行执行**：多个节点从同一节点分叉实现并行
- **时间旅行**：利用检查点回溯和重放执行过程
- 这些高级特性让 LangGraph 能构建生产级的复杂 Agent 系统
