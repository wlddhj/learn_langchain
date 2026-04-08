# 第16章：综合实战 —— 构建智能研究助手

本章将综合运用前面 15 章的所有知识，从零构建一个完整的 **智能研究助手（Smart Research Assistant）**。这个助手能搜索信息、分析文档、生成报告，并支持多轮对话。

## 16.1 项目概述

### 功能需求

```
用户输入问题
    ↓
[问题分类] → 是简单问题？→ 直接回答
    ↓ 否
[研究流程]
    ├── 搜索相关信息（工具调用）
    ├── 检索本地知识库（RAG）
    ├── 综合分析（LLM 推理）
    └── 生成结构化报告（输出解析）
    ↓
[质量审查] → 不通过？→ 补充研究（循环）
    ↓ 通过
[输出结果] → 支持追问（多轮对话 + 记忆）
```

### 涉及的核心知识

| 章节 | 在本项目中的应用 |
|------|------------------|
| 第1-2章 | 环境搭建、LLM 调用 |
| 第3-4章 | Prompt 模板、结构化输出 |
| 第5章 | LCEL 管道操作 |
| 第6-7章 | 文档加载、向量检索（RAG） |
| 第8章 | 对话记忆管理 |
| 第9、12章 | 自定义工具（搜索、文件读写） |
| 第10-11章 | ReAct Agent 模式 |
| 第13-14章 | LangGraph 工作流、持久化 |
| 第15章 | 多 Agent 协作 |

## 16.2 项目结构

```
smart_research_assistant/
├── main.py              # 入口文件
├── config.py            # 配置文件
├── tools/
│   ├── __init__.py
│   ├── search.py        # 搜索工具
│   ├── file_ops.py      # 文件操作工具
│   └── knowledge.py     # 知识库工具（RAG）
├── agents/
│   ├── __init__.py
│   ├── supervisor.py    # 主管 Agent
│   ├── researcher.py    # 研究 Agent
│   ├── analyst.py       # 分析 Agent
│   └── writer.py        # 写作 Agent
├── graph.py             # LangGraph 工作流定义
├── state.py             # State 定义
└── data/
    └── knowledge/       # 本地知识库文档
```

## 16.3 第一步：配置与 State 定义

### config.py

```python
"""项目配置"""
import os

# LLM 配置
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# 检索配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# 向量存储
VECTOR_STORE_PATH = "./data/vector_store"

# 持久化
CHECKPOINT_DB_PATH = "./data/checkpoints.db"

# 最大研究轮次
MAX_RESEARCH_ITERATIONS = 3
```

### state.py

```python
"""State 定义 —— 贯穿整个工作流的核心数据结构"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """研究助手的全局状态"""
    # 对话
    messages: Annotated[list, add_messages]  # 消息历史

    # 研究过程
    question: str              # 原始问题
    question_type: str         # 问题类型: simple / research / report
    search_results: str        # 搜索结果
    knowledge_results: str     # 知识库检索结果
    research_notes: str        # 研究笔记
    analysis: str              # 分析结果
    report: str                # 最终报告

    # 控制
    iteration: int             # 当前迭代次数
    review_passed: bool        # 质量审查是否通过
    review_feedback: str       # 审查反馈
    next_agent: str            # 下一个 Agent
```

## 16.4 第二步：自定义工具

### tools/search.py —— 搜索工具

```python
"""搜索工具"""
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """使用搜索引擎搜索网络信息。
    当需要获取最新资讯、技术动态或公开信息时使用。
    query: 搜索关键词"""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "未找到相关结果"

        output = ""
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n"
            output += f"   摘要: {r['body'][:200]}\n\n"

        return output
    except Exception as e:
        return f"搜索失败: {e}"


@tool
def search_arxiv(query: str) -> str:
    """搜索 arXiv 学术论文。
    当需要查找学术论文、研究成果时使用。
    query: 论文搜索关键词"""
    try:
        import httpx

        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 5,
        }

        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params)

        # 简单解析 XML 结果
        entries = resp.text.split("<entry>")[1:]  # 跳过头部
        results = []
        for entry in entries[:5]:
            title = entry.split("<title>")[1].split("</title>")[0].strip()
            summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
            results.append(f"- {title}\n  摘要: {summary[:200]}")

        return "\n\n".join(results) if results else "未找到相关论文"
    except Exception as e:
        return f"搜索失败: {e}"
```

### tools/knowledge.py —— RAG 知识库工具

```python
"""本地知识库工具（RAG）"""
from langchain_core.tools import tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def build_vector_store(docs_dir: str, save_path: str):
    """从文档目录构建向量存储"""
    # 1. 加载文档
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.{txt,md}",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # 2. 切分文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    # 3. 创建向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=save_path,
    )
    return vectorstore


def get_retriever(docs_dir: str = "./data/knowledge", save_path: str = "./data/vector_store"):
    """获取检索器"""
    embeddings = OpenAIEmbeddings()

    # 尝试加载已有向量存储
    try:
        vectorstore = Chroma(
            persist_directory=save_path,
            embedding_function=embeddings,
        )
        # 检查是否有数据
        if vectorstore._collection.count() == 0:
            raise ValueError("空向量存储")
    except Exception:
        # 没有则重新构建
        vectorstore = build_vector_store(docs_dir, save_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


# 初始化全局检索器（延迟加载）
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@tool
def search_knowledge_base(query: str) -> str:
    """从本地知识库中检索相关信息。
    当需要查询已有文档、内部资料、历史记录时使用。
    query: 检索关键词"""
    try:
        retriever = _get_retriever()
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中未找到相关内容"

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            results.append(f"[文档{i}] 来源: {source}\n{doc.page_content}")

        return "\n\n".join(results)
    except Exception as e:
        return f"知识库检索失败: {e}"
```

### tools/file_ops.py —— 文件操作工具

```python
"""文件操作工具"""
import os
from langchain_core.tools import tool


@tool
def save_report(filename: str, content: str) -> str:
    """将报告保存为 Markdown 文件。
    filename: 文件名（不含路径，自动保存到 ./data/reports/）
    content: 报告内容（Markdown 格式）"""
    try:
        report_dir = "./data/reports"
        os.makedirs(report_dir, exist_ok=True)

        filepath = os.path.join(report_dir, filename)
        if not filepath.endswith(".md"):
            filepath += ".md"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return f"报告已保存: {filepath} ({len(content)} 字符)"
    except Exception as e:
        return f"保存失败: {e}"


@tool
def list_reports() -> str:
    """列出所有已保存的报告文件"""
    report_dir = "./data/reports"
    if not os.path.exists(report_dir):
        return "暂无报告"

    files = os.listdir(report_dir)
    if not files:
        return "暂无报告"

    result = "已保存的报告:\n"
    for f in sorted(files):
        size = os.path.getsize(os.path.join(report_dir, f))
        result += f"  - {f} ({size} bytes)\n"
    return result
```

### tools/__init__.py

```python
"""工具集合"""
from tools.search import search_web, search_arxiv
from tools.knowledge import search_knowledge_base
from tools.file_ops import save_report, list_reports

# 所有可用工具
ALL_TOOLS = [
    search_web,
    search_arxiv,
    search_knowledge_base,
    save_report,
    list_reports,
]
```

## 16.5 第三步：定义 Agent 节点

### agents/researcher.py —— 研究 Agent

```python
"""研究 Agent：负责信息收集"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from state import ResearchState


def researcher_node(state: ResearchState) -> dict:
    """研究节点：搜索并整理信息"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 如果之前有搜索结果或知识库结果，综合整理
    existing_info = ""
    if state.get("search_results"):
        existing_info += f"网络搜索结果:\n{state['search_results']}\n\n"
    if state.get("knowledge_results"):
        existing_info += f"知识库检索结果:\n{state['knowledge_results']}\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个研究专家。你的任务是：
1. 分析用户的问题
2. 整理已有信息
3. 总结关键发现
4. 指出信息缺口（如果有的话）

请用中文输出研究笔记，包含：
- 关键发现（要点列表）
- 信息来源评估
- 需要补充的信息（如有）"""),
        ("human", """问题: {question}

已有信息:
{info}

请整理研究笔记。"""),
    ])

    chain = prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "info": existing_info or "暂无已有信息",
    })

    return {"research_notes": result.content}
```

### agents/analyst.py —— 分析 Agent

```python
"""分析 Agent：负责深度分析"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from state import ResearchState


def analyst_node(state: ResearchState) -> dict:
    """分析节点：深度分析研究结果"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个分析专家。基于研究笔记，进行深度分析：
1. 提炼核心观点
2. 对比不同来源的信息
3. 发现趋势和模式
4. 给出专业见解

输出结构化的分析报告。"""),
        ("human", """问题: {question}

研究笔记:
{notes}

请进行深度分析。"""),
    ])

    chain = prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "notes": state.get("research_notes", "无研究笔记"),
    })

    return {"analysis": result.content}
```

### agents/writer.py —— 写作 Agent

```python
"""写作 Agent：负责生成最终报告"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from state import ResearchState


class ResearchReport(BaseModel):
    """研究报告结构"""
    title: str = Field(description="报告标题")
    summary: str = Field(description="一句话摘要")
    key_findings: list[str] = Field(description="关键发现列表")
    detailed_analysis: str = Field(description="详细分析内容")
    conclusion: str = Field(description="结论")
    references: list[str] = Field(description="参考来源")


def writer_node(state: ResearchState) -> dict:
    """写作节点：生成结构化报告"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ResearchReport)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个技术写作专家。基于研究和分析结果，撰写一份专业的报告。

{format_instructions}

要求：
- 语言简洁专业
- 观点有据可依
- 结构清晰完整"""),
        ("human", """问题: {question}

研究笔记:
{notes}

分析结果:
{analysis}

请撰写报告。"""),
    ])

    chain = prompt | llm | parser

    try:
        report: ResearchReport = chain.invoke({
            "question": state["question"],
            "notes": state.get("research_notes", ""),
            "analysis": state.get("analysis", ""),
            "format_instructions": parser.get_format_instructions(),
        })

        # 生成 Markdown 格式的报告
        md_report = f"""# {report.title}

> {report.summary}

## 关键发现

{chr(10).join(f'- {f}' for f in report.key_findings)}

## 详细分析

{report.detailed_analysis}

## 结论

{report.conclusion}

## 参考来源

{chr(10).join(f'- {r}' for r in report.references)}
"""
        return {"report": md_report}

    except Exception as e:
        # 解析失败时，使用纯文本回退
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是写作专家。基于以下信息写一份简洁的研究报告。"),
            ("human", "问题: {question}\n\n研究笔记:\n{notes}\n\n分析:\n{analysis}"),
        ])
        chain = simple_prompt | llm
        result = chain.invoke({
            "question": state["question"],
            "notes": state.get("research_notes", ""),
            "analysis": state.get("analysis", ""),
        })
        return {"report": result.content}
```

## 16.6 第四步：构建 LangGraph 工作流

### graph.py —— 核心工作流

```python
"""LangGraph 工作流定义"""
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from state import ResearchState
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from tools import ALL_TOOLS


# ===== 路由节点 =====

def classify_question(state: ResearchState) -> dict:
    """问题分类：判断问题类型"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    last_msg = state["messages"][-1]
    question = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    response = llm.invoke([
        SystemMessage(content="""根据用户问题判断类型，只回复一个词：

- "simple" — 简单的闲聊、常识问题、定义解释
- "research" — 需要搜索和分析的问题
- "report" — 需要生成完整报告的深度问题"""),
        HumanMessage(content=f"问题: {question}"),
    ])

    q_type = response.content.strip().lower()
    if q_type not in ("simple", "research", "report"):
        q_type = "research"  # 默认按研究处理

    return {
        "question": question,
        "question_type": q_type,
    }


def simple_answer(state: ResearchState) -> dict:
    """直接回答简单问题"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ===== 工具调用节点 =====

def should_use_tools(state: ResearchState) -> Literal["tools", "researcher"]:
    """判断是否需要调用工具搜索信息"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 检查是否已有足够信息
    if state.get("search_results") or state.get("knowledge_results"):
        return "researcher"

    # 让 LLM 决定是否需要搜索
    response = llm.invoke([
        SystemMessage(content="判断以下问题是否需要搜索外部信息。需要回复 YES，不需要回复 NO。"),
        HumanMessage(content=f"问题: {state.get('question', '')}"),
    ])

    if "YES" in response.content.upper():
        return "tools"

    return "researcher"


def tool_calling_node(state: ResearchState) -> dict:
    """工具调用节点：让 LLM 决定调用哪些工具"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    response = llm_with_tools.invoke([
        SystemMessage(content="你是搜索专家。请使用合适的工具搜索以下问题的相关信息。"),
        HumanMessage(content=state["question"]),
    ])

    return {"messages": [response]}


def process_tool_results(state: ResearchState) -> dict:
    """处理工具调用结果"""
    messages = state["messages"]
    results = []

    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            # 收集工具返回的结果
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                results.append(msg.content)

    # 合并搜索结果
    combined = "\n".join(results)

    if combined:
        return {"search_results": combined}

    return {}


# ===== 质量审查节点 =====

def review_node(state: ResearchState) -> dict:
    """质量审查：检查报告是否完整"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    iteration = state.get("iteration", 0)

    # 超过最大迭代次数，强制通过
    if iteration >= 3:
        return {"review_passed": True, "review_feedback": "已达到最大迭代次数，输出当前结果。"}

    response = llm.invoke([
        SystemMessage(content="""你是质量审查员。评估以下研究报告：

1. 是否回答了原始问题？
2. 内容是否充实（非空洞模板）？
3. 逻辑是否连贯？

回复格式：
PASS - 原因
或
REVISE - 需要改进的具体建议"""),
        HumanMessage(content=f"""原始问题: {state.get('question', '')}

报告内容:
{state.get('report', '无报告')}"""),
    ])

    content = response.content

    if "PASS" in content.upper()[:10]:
        return {"review_passed": True, "review_feedback": content}
    else:
        return {
            "review_passed": False,
            "review_feedback": content,
            "iteration": iteration + 1,
        }


# ===== 路由函数 =====

def route_by_question_type(state: ResearchState) -> str:
    """根据问题类型路由"""
    q_type = state.get("question_type", "research")
    if q_type == "simple":
        return "simple_answer"
    else:
        return "tool_calling"  # research 和 report 都先搜索信息


def route_after_tools(state: ResearchState) -> str:
    """工具调用后路由"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "researcher"


def route_after_review(state: ResearchState) -> str:
    """审查后路由"""
    if state.get("review_passed", False):
        return END
    return "researcher"  # 未通过，重新研究


# ===== 构建图 =====

def build_graph():
    """构建完整的工作流图"""
    builder = StateGraph(ResearchState)

    # 添加节点
    builder.add_node("classify", classify_question)
    builder.add_node("simple_answer", simple_answer)
    builder.add_node("tool_calling", tool_calling_node)
    builder.add_node("tool_executor", ToolNode(ALL_TOOLS))
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review", review_node)

    # 添加边
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_by_question_type)

    # 简单问题 → 直接回答 → 结束
    builder.add_edge("simple_answer", END)

    # 研究流程
    builder.add_edge("tool_calling", "tool_executor")
    builder.add_edge("tool_executor", "researcher")
    builder.add_conditional_edges("classify", route_by_question_type, {
        "simple_answer": "simple_answer",
        "tool_calling": "tool_calling",
    })
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", "review")
    builder.add_conditional_edges("review", route_after_review)

    # 添加持久化
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph
```

等一下，上面的图定义有重复的条件边。让我们重写一个更清晰的版本：

```python
"""LangGraph 工作流定义（最终版）"""
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from tools import ALL_TOOLS


# ===== 节点函数 =====

def classify_question(state: ResearchState) -> dict:
    """问题分类"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    last_msg = state["messages"][-1]
    question = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    response = llm.invoke([
        SystemMessage(content='''根据问题判断类型，只回复一个词：
- "simple" — 闲聊、常识、定义
- "research" — 需搜索分析的问题
- "report" — 需完整报告的深度问题'''),
        HumanMessage(content=f"问题: {question}"),
    ])

    q_type = response.content.strip().lower()
    if q_type not in ("simple", "research", "report"):
        q_type = "research"

    return {"question": question, "question_type": q_type}


def simple_answer(state: ResearchState) -> dict:
    """直接回答"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_calling_node(state: ResearchState) -> dict:
    """工具调用"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    response = llm_with_tools.invoke([
        SystemMessage(content="你是搜索专家。使用工具搜索以下问题的信息。"),
        HumanMessage(content=state["question"]),
    ])
    return {"messages": [response]}


def collect_results(state: ResearchState) -> dict:
    """收集工具结果"""
    results = []
    for msg in state["messages"]:
        # ToolMessage 有 content
        if hasattr(msg, "name") and msg.content:
            results.append(f"[{msg.name}] {msg.content[:500]}")

    combined = "\n\n".join(results)
    return {"search_results": combined}


def review_node(state: ResearchState) -> dict:
    """质量审查"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    iteration = state.get("iteration", 0)

    if iteration >= 3:
        return {"review_passed": True}

    response = llm.invoke([
        SystemMessage(content="""审查报告质量。回复 PASS 或 REVISE。"""),
        HumanMessage(content=f"问题: {state.get('question', '')}\n\n报告:\n{state.get('report', '')[:2000]}"),
    ])

    passed = "PASS" in response.content.upper()[:10]
    return {"review_passed": passed, "iteration": iteration + 1}


# ===== 路由函数 =====

def route_by_type(state: ResearchState) -> str:
    if state.get("question_type") == "simple":
        return "simple_answer"
    return "tool_calling"


def route_after_tools(state: ResearchState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "collect"


def route_after_review(state: ResearchState) -> str:
    if state.get("review_passed", False):
        return END
    return "researcher"


# ===== 构建图 =====

def build_graph():
    builder = StateGraph(ResearchState)

    # 节点
    builder.add_node("classify", classify_question)
    builder.add_node("simple_answer", simple_answer)
    builder.add_node("tool_calling", tool_calling_node)
    builder.add_node("tool_executor", ToolNode(ALL_TOOLS))
    builder.add_node("collect", collect_results)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review", review_node)

    # 边
    builder.add_edge(START, "classify")

    builder.add_conditional_edges("classify", route_by_type)

    builder.add_edge("simple_answer", END)

    builder.add_conditional_edges("tool_calling", route_after_tools)
    builder.add_edge("tool_executor", "collect")
    builder.add_edge("collect", "researcher")

    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "writer")
    builder.add_edge("writer", "review")

    builder.add_conditional_edges("review", route_after_review)

    # 编译
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph
```

### 图的执行流程

```
START
  ↓
[classify] ── simple ──→ [simple_answer] ──→ END
  ↓ research/report
[tool_calling]
  ↓ 有工具调用？
  ├── Yes → [tool_executor] → [collect]
  └── No  → [collect]
                ↓
           [researcher]
                ↓
            [analyst]
                ↓
            [writer]
                ↓
            [review] ── 未通过 ──→ [researcher]（循环）
                ↓ 通过
               END
```

## 16.7 第五步：主程序入口

### main.py

```python
"""智能研究助手 - 主程序"""
import sys
from graph import build_graph


def interactive_mode(graph):
    """交互式对话模式"""
    print("=" * 50)
    print("智能研究助手 v1.0")
    print("输入问题开始研究，输入 'quit' 退出")
    print("=" * 50)

    thread_id = "session-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 调用图
        result = graph.invoke(
            {"messages": [("user", user_input)]},
            config=config,
        )

        # 输出结果
        print("\n" + "=" * 40)

        if result.get("report"):
            print("研究报告:")
            print(result["report"])
        else:
            # 简单回答
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content:
                    # 只输出最后的 AI 回复
                    pass
            last_ai = None
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    last_ai = msg.content
            if last_ai:
                print(f"AI: {last_ai}")

        print("=" * 40)


def single_query(graph, question: str):
    """单次查询模式"""
    config = {"configurable": {"thread_id": "single"}}

    result = graph.invoke(
        {"messages": [("user", question)]},
        config=config,
    )

    if result.get("report"):
        print(result["report"])
    else:
        for msg in result["messages"]:
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                print(msg.content)


def stream_mode(graph, question: str):
    """流式输出模式"""
    config = {"configurable": {"thread_id": "stream"}}

    print(f"问题: {question}\n")
    print("处理中...")

    for event in graph.stream(
        {"messages": [("user", question)]},
        config=config,
    ):
        for node_name, output in event.items():
            print(f"\n[{node_name}]")
            if "report" in output:
                print(output["report"][:200] + "...")
            elif "research_notes" in output:
                print(f"研究笔记: {output['research_notes'][:200]}...")
            elif "analysis" in output:
                print(f"分析: {output['analysis'][:200]}...")

    print("\n完成!")


if __name__ == "__main__":
    graph = build_graph()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        single_query(graph, query)
    else:
        interactive_mode(graph)
```

## 16.8 第六步：运行与测试

### 准备知识库

```python
"""准备测试数据"""
import os

# 创建知识库目录
os.makedirs("./data/knowledge", exist_ok=True)

# 创建示例知识文档
with open("./data/knowledge/langchain-guide.md", "w", encoding="utf-8") as f:
    f.write("""# LangChain 开发指南

## 什么是 LangChain
LangChain 是一个用于开发大语言模型应用的开源框架。
它提供了模块化的组件，让开发者可以轻松构建 LLM 应用。

## 核心组件
- Model I/O: LLM 和 Chat Model 的接口
- Retrieval: RAG 检索增强生成的组件
- Chains: 链式调用
- Agents: 自主决策的 Agent
- Memory: 对话记忆

## LangGraph
LangGraph 是 LangChain 的扩展，专门用于构建有状态的、多角色的应用。
它基于图结构，支持循环、条件分支等复杂工作流。
""")

with open("./data/knowledge/agent-patterns.md", "w", encoding="utf-8") as f:
    f.write("""# Agent 设计模式

## ReAct 模式
ReAct (Reason + Act) 是最常用的 Agent 模式。
Agent 在每一步都先推理（Reason），然后行动（Act），
观察结果后再决定下一步。

## 多 Agent 模式
- Supervisor 模式：中心调度器分配任务
- Swarm 模式：Agent 间直接交接
- Hierarchical 模式：多层级管理

## 最佳实践
1. 工具描述要清晰，帮助 LLM 正确选择
2. 设置最大迭代次数，防止无限循环
3. 每个步骤的输入输出要可追踪
4. 使用 LangGraph 管理复杂工作流
""")
```

### 测试运行

```python
"""测试脚本"""
from graph import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "test-1"}}

# 测试1：简单问题
print("=== 测试1: 简单问题 ===")
result = graph.invoke(
    {"messages": [("user", "什么是 RAG？")]},
    config=config,
)

# 测试2：需要研究的问题
print("\n=== 测试2: 研究问题 ===")
result = graph.invoke(
    {"messages": [("user", "分析一下 LangChain 和 LangGraph 的关系和区别")]},
    config={"configurable": {"thread_id": "test-2"}},
)
if result.get("report"):
    print(result["report"])

# 测试3：需要搜索的深度问题
print("\n=== 测试3: 深度研究 ===")
result = graph.invoke(
    {"messages": [("user", "请研究 2024 年 AI Agent 领域的最新进展，并生成报告")]},
    config={"configurable": {"thread_id": "test-3"}},
)
if result.get("report"):
    print(result["report"][:500])

# 测试4：多轮对话（利用记忆）
print("\n=== 测试4: 多轮对话 ===")
config_multi = {"configurable": {"thread_id": "multi-1"}}

# 第一轮
graph.invoke(
    {"messages": [("user", "研究一下 Python 的异步编程")]},
    config=config_multi,
)

# 第二轮（追问，利用持久化的状态）
result = graph.invoke(
    {"messages": [("user", "基于之前的研究，详细解释 asyncio 的用法")]},
    config=config_multi,
)
```

## 16.9 项目优化：添加日志与监控

```python
"""日志与监控工具"""
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("research_assistant")


def log_node_execution(node_name: str):
    """装饰器：记录节点执行时间和结果"""
    def decorator(func):
        def wrapper(state):
            logger.info(f"[{node_name}] 开始执行")
            start = time.time()

            result = func(state)

            elapsed = time.time() - start
            logger.info(f"[{node_name}] 完成, 耗时 {elapsed:.2f}s")

            # 记录结果摘要
            for key, value in result.items():
                if isinstance(value, str):
                    logger.info(f"[{node_name}] 输出 {key}: {len(value)} 字符")
                elif isinstance(value, list):
                    logger.info(f"[{node_name}] 输出 {key}: {len(value)} 项")

            return result
        return wrapper
    return decorator


# 使用示例
@log_node_execution("researcher")
def researcher_node(state):
    # ... 原有逻辑
    pass


@log_node_execution("analyst")
def analyst_node(state):
    # ... 原有逻辑
    pass
```

## 16.10 项目优化：添加回调

```python
"""自定义回调：跟踪 Agent 执行过程"""
from langchain_core.callbacks import BaseCallbackHandler


class ResearchCallback(BaseCallbackHandler):
    """研究助手的自定义回调"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM 调用] 模型: {serialized.get('name', 'unknown')}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[工具调用] {serialized.get('name', 'unknown')}: {input_str[:100]}")

    def on_tool_end(self, output, **kwargs):
        print(f"[工具结果] {str(output)[:100]}")

    def on_chain_error(self, error, **kwargs):
        print(f"[错误] {error}")


# 使用回调
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[ResearchCallback()],
)
```

## 16.11 完整版：单文件快速体验

为了方便快速运行，以下是完整项目的单文件版本：

```python
"""
智能研究助手 - 单文件完整版
综合运用: LCEL、RAG、Tools、Agent、LangGraph、Memory
"""

# ============================================================
# 1. State 定义
# ============================================================
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    question_type: str
    search_results: str
    research_notes: str
    analysis: str
    report: str
    iteration: int
    review_passed: bool
    next_agent: str


# ============================================================
# 2. 工具定义
# ============================================================
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """搜索网络信息。query: 搜索关键词"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "未找到相关结果"
        return "\n\n".join(
            f"{i}. {r['title']}\n   {r['body'][:200]}"
            for i, r in enumerate(results, 1)
        )
    except Exception as e:
        return f"搜索失败: {e}"


@tool
def save_report(filename: str, content: str) -> str:
    """保存报告为 Markdown 文件"""
    import os
    os.makedirs("./data/reports", exist_ok=True)
    path = f"./data/reports/{filename}"
    if not path.endswith(".md"):
        path += ".md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已保存: {path}"


TOOLS = [search_web, save_report]


# ============================================================
# 3. 节点函数
# ============================================================
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


def classify(state: ResearchState) -> dict:
    """分类问题"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question = state["messages"][-1].content

    resp = llm.invoke([
        SystemMessage(content='回复一个词: simple/research/report'),
        HumanMessage(content=question),
    ])
    q_type = resp.content.strip().lower()
    if q_type not in ("simple", "research", "report"):
        q_type = "research"
    return {"question": question, "question_type": q_type}


def simple_answer(state: ResearchState) -> dict:
    """简单回答"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}


def tool_call(state: ResearchState) -> dict:
    """调用搜索工具"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_tools = llm.bind_tools(TOOLS)
    resp = llm_tools.invoke([
        SystemMessage(content="搜索以下问题的信息。"),
        HumanMessage(content=state["question"]),
    ])
    return {"messages": [resp]}


def collect_results(state: ResearchState) -> dict:
    """收集搜索结果"""
    results = []
    for msg in state["messages"]:
        if hasattr(msg, "name") and msg.content:
            results.append(f"[{msg.name}] {msg.content[:500]}")
    return {"search_results": "\n\n".join(results)}


def research(state: ResearchState) -> dict:
    """研究分析"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是研究专家。整理信息，输出研究笔记。"),
        ("human", "问题: {question}\n\n搜索结果:\n{search}"),
    ])
    chain = prompt | llm
    resp = chain.invoke({
        "question": state["question"],
        "search": state.get("search_results", "无"),
    })
    return {"research_notes": resp.content}


def analyze(state: ResearchState) -> dict:
    """深度分析"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是分析专家。基于研究笔记进行深度分析。"),
        ("human", "问题: {question}\n\n笔记:\n{notes}"),
    ])
    chain = prompt | llm
    resp = chain.invoke({
        "question": state["question"],
        "notes": state.get("research_notes", ""),
    })
    return {"analysis": resp.content}


def write_report(state: ResearchState) -> dict:
    """撰写报告"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是写作专家。撰写一份 Markdown 格式的研究报告。
包含: 标题、摘要、关键发现、详细分析、结论。"""),
        ("human", "问题: {question}\n\n笔记:\n{notes}\n\n分析:\n{analysis}"),
    ])
    chain = prompt | llm
    resp = chain.invoke({
        "question": state["question"],
        "notes": state.get("research_notes", ""),
        "analysis": state.get("analysis", ""),
    })
    return {"report": resp.content}


def review(state: ResearchState) -> dict:
    """质量审查"""
    iteration = state.get("iteration", 0)
    if iteration >= 3:
        return {"review_passed": True, "iteration": iteration}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = llm.invoke([
        SystemMessage(content="审查报告。回复 PASS 或 REVISE。"),
        HumanMessage(content=f"问题: {state['question']}\n\n报告:\n{state.get('report', '')[:2000]}"),
    ])
    passed = "PASS" in resp.content.upper()[:10]
    return {"review_passed": passed, "iteration": iteration + 1}


# ============================================================
# 4. 路由函数
# ============================================================

def route_type(state: ResearchState) -> str:
    return "simple_answer" if state.get("question_type") == "simple" else "tool_call"


def route_tools(state: ResearchState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "executor"
    return "collect"


def route_review(state: ResearchState) -> str:
    return END if state.get("review_passed") else "research"


# ============================================================
# 5. 构建并运行
# ============================================================

def build():
    builder = StateGraph(ResearchState)

    builder.add_node("classify", classify)
    builder.add_node("simple_answer", simple_answer)
    builder.add_node("tool_call", tool_call)
    builder.add_node("executor", ToolNode(TOOLS))
    builder.add_node("collect", collect_results)
    builder.add_node("research", research)
    builder.add_node("analyze", analyze)
    builder.add_node("write", write_report)
    builder.add_node("review", review)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_type)
    builder.add_edge("simple_answer", END)
    builder.add_conditional_edges("tool_call", route_tools)
    builder.add_edge("executor", "collect")
    builder.add_edge("collect", "research")
    builder.add_edge("research", "analyze")
    builder.add_edge("analyze", "write")
    builder.add_edge("write", "review")
    builder.add_conditional_edges("review", route_review)

    return builder.compile(checkpointer=MemorySaver())


# 运行
if __name__ == "__main__":
    graph = build()
    config = {"configurable": {"thread_id": "demo"}}

    # 简单问题
    print("=== 简单问题 ===")
    r = graph.invoke({"messages": [("user", "什么是 RAG？")]}, config)
    print(r["messages"][-1].content)

    # 研究问题
    print("\n=== 研究问题 ===")
    r = graph.invoke(
        {"messages": [("user", "研究 AI Agent 的最新进展并生成报告")]},
        config={"configurable": {"thread_id": "research-1"}},
    )
    if r.get("report"):
        print(r["report"])
```

## 16.12 关键设计回顾

本项目的架构贯穿了全书的核心概念：

```
┌─────────────────────────────────────────────────┐
│                  用户输入                         │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  LangGraph 工作流（第13-14章）                    │
│                                                   │
│  ┌──────────┐    ┌──────────┐                    │
│  │ classify │───→│ 简单回答  │ ← LCEL 管道（第5章） │
│  │ (路由)   │    └──────────┘                     │
│  └────┬─────┘                                     │
│       ↓                                           │
│  ┌──────────┐    ┌──────────┐                    │
│  │ 搜索工具  │←──→│ 工具执行  │ ← Tools（第9/12章） │
│  └────┬─────┘    └──────────┘                     │
│       ↓                                           │
│  ┌──────────┐                                     │
│  │ 研究节点  │ ← Prompt 模板（第3章）              │
│  └────┬─────┘                                     │
│       ↓                                           │
│  ┌──────────┐                                     │
│  │ 分析节点  │ ← LLM 调用（第2章）                 │
│  └────┬─────┘                                     │
│       ↓                                           │
│  ┌──────────┐                                     │
│  │ 写作节点  │ ← 输出解析（第4章）                 │
│  └────┬─────┘                                     │
│       ↓                                           │
│  ┌──────────┐                                     │
│  │ 质量审查  │──→ 未通过? → 回到研究（循环）        │
│  └────┬─────┘                                     │
│       ↓ 通过                                      │
└─────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│  输出结果 + 记忆持久化（第8/14章）                 │
└─────────────────────────────────────────────────┘
```

## 16.13 扩展方向

这个项目可以继续扩展：

| 方向 | 说明 | 涉及技术 |
|------|------|----------|
| **Web 界面** | 用 Streamlit/Gradio 构建前端 | Streamlit, Gradio |
| **数据库持久化** | 用 SQLite/PostgreSQL 存储对话 | LangGraph Checkpointer |
| **更多工具** | 接入代码执行、图片生成等 | Custom Tools |
| **多语言支持** | 支持中英文研究 | Prompt 设计 |
| **用户反馈** | 让用户评价报告质量 | Human-in-the-Loop |
| **邮件推送** | 自动发送研究报告 | 工具 + 工作流 |
| **API 服务** | 封装为 REST API | FastAPI |
| **部署** | Docker 容器化部署 | Docker, CI/CD |

## 16.14 本章小结

本章构建了一个完整的 **智能研究助手**，综合运用了：

1. **LangGraph**：定义复杂的工作流图，支持条件路由、循环、持久化
2. **自定义工具**：搜索、知识库检索、文件保存
3. **RAG**：本地知识库的向量检索
4. **LCEL 管道**：`prompt | llm | parser` 链式调用
5. **Prompt 工程**：针对不同角色的系统提示
6. **输出解析**：Pydantic 结构化输出
7. **状态管理**：自定义 State 在节点间传递数据
8. **质量审查循环**：确保输出质量

恭喜你完成了全部 16 章的学习！你已经具备了独立构建 AI Agent 应用的能力。接下来，选择一个你感兴趣的方向，开始构建自己的项目吧。
