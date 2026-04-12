"""
第15章 Demo 3：实战 —— 内容创作团队

多 Agent 协作完成：研究 → 大纲 → 写作 → 审核（循环修改）。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Literal, Annotated

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# State
# ============================================================

class TeamState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    research: str
    outline: str
    article: str
    feedback: str
    revision_count: int
    next: str


# ============================================================
# 团队成员节点
# ============================================================

def researcher(state: TeamState):
    """研究员：收集和分析信息"""
    response = llm.invoke([
        SystemMessage(content="你是资深研究员。对给定主题进行深入分析，列出关键要点和核心论据。用中文回答。"),
        HumanMessage(content=f"研究主题: {state['topic']}"),
    ])
    return {
        "research": response.content,
        "messages": [response],
    }


def outliner(state: TeamState):
    """大纲编辑：规划文章结构"""
    response = llm.invoke([
        SystemMessage(content="你是大纲编辑。基于研究笔记创建文章大纲，包含3-5个章节。用中文回答。"),
        HumanMessage(content=f"研究笔记:\n{state['research']}\n\n请创建文章大纲。"),
    ])
    return {
        "outline": response.content,
        "messages": [response],
    }


def writer(state: TeamState):
    """写作者：撰写完整文章"""
    feedback_ctx = ""
    if state.get("feedback"):
        feedback_ctx = f"\n\n审核反馈（请据此修改）:\n{state['feedback']}"

    response = llm.invoke([
        SystemMessage(content="你是专业写作者。根据大纲和研究笔记撰写文章，不超过300字。用中文回答。"),
        HumanMessage(content=f"大纲:\n{state['outline']}\n\n研究笔记:\n{state['research']}{feedback_ctx}"),
    ])
    return {
        "article": response.content,
        "revision_count": state.get("revision_count", 0) + 1,
        "messages": [response],
    }


def reviewer(state: TeamState):
    """审稿人：审核文章质量"""
    response = llm.invoke([
        SystemMessage(content="""你是严格的审稿人。审查文章质量，检查：
1. 内容是否完整覆盖主题
2. 结构是否清晰
3. 语言是否流畅

如果质量合格，回复 "PASS" 并给出简短评价。
如果需要修改，回复 "REVISE" 并列出需要改进的地方。"""),
        HumanMessage(content=f"审查以下文章:\n\n{state['article']}"),
    ])

    content = response.content
    decision = "PASS" if "PASS" in content.upper() and "REVISE" not in content.upper() else "REVISE"
    return {
        "feedback": content,
        "next": decision,
        "messages": [response],
    }


def review_decision(state: TeamState):
    """审核决策：通过则结束，需要修改则回到写作者"""
    if state.get("next") == "PASS":
        return END
    if state.get("revision_count", 0) >= 2:
        return END  # 最多修改2次
    return "writer"


# ============================================================
# 构建团队 Graph
# ============================================================

def build_team_graph():
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

    return builder.compile()


# ============================================================
# Demo 1: 完整创作流程
# ============================================================

def demo_full_workflow():
    print("=" * 50)
    print(f"内容创作团队 (1/2): 完整流程 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_team_graph()

    result = graph.invoke({
        "topic": "AI Agent 在软件开发中的应用与前景",
        "messages": [],
        "research": "",
        "outline": "",
        "article": "",
        "feedback": "",
        "revision_count": 0,
        "next": "",
    })

    print(f"主题: {result['topic']}")
    print()
    print(f"[研究员] 研究笔记:")
    print(f"  {preview(result['research'], 150)}")
    print()
    print(f"[大纲编辑] 文章大纲:")
    print(f"  {preview(result['outline'], 150)}")
    print()
    print(f"[写作者] 文章 (第{result['revision_count']}版):")
    print(f"  {preview(result['article'], 200)}")
    print()
    print(f"[审稿人] 反馈:")
    print(f"  {preview(result['feedback'], 150)}")
    print()
    print(f"最终状态: {result['next']}, 修改次数: {result['revision_count']}")
    print()


# ============================================================
# Demo 2: 追踪创作过程
# ============================================================

def demo_stream_workflow():
    print("=" * 50)
    print(f"内容创作团队 (2/2): 追踪创作过程 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_team_graph()
    topic = "Python 异步编程的最佳实践"

    print(f"主题: {topic}")
    print("-" * 40)

    node_names = {
        "researcher": "研究员",
        "outliner": "大纲编辑",
        "writer": "写作者",
        "reviewer": "审稿人",
    }

    for event in graph.stream({
        "topic": topic,
        "messages": [],
        "research": "",
        "outline": "",
        "article": "",
        "feedback": "",
        "revision_count": 0,
        "next": "",
    }):
        for node, output in event.items():
            cn_name = node_names.get(node, node)
            if node == "researcher":
                content = output.get("research", "")
            elif node == "outliner":
                content = output.get("outline", "")
            elif node == "writer":
                ver = output.get("revision_count", 0)
                content = f"(第{ver}版) {output.get('article', '')}"
            elif node == "reviewer":
                content = f"[{output.get('next', '?')}] {output.get('feedback', '')}"
            else:
                content = str(output)
            print(f"  [{cn_name}] {preview(content, 80)}")

    print()
    print("流程: 研究员 → 大纲编辑 → 写作者 → 审稿人 → (可能回到写作者)")
    print()


def preview(text, max_len):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


if __name__ == "__main__":
    demo_full_workflow()
    demo_stream_workflow()

    print("=" * 50)
    print("Demo 15-3 完成!")
    print()
    print("内容创作团队要点:")
    print("  分工明确      - 研究员/大纲编辑/写作者/审稿人各司其职")
    print("  循环修改      - 审稿人可让写作者反复修改")
    print("  修改次数限制  - 防止无限循环")
    print("  共享 State    - research/outline/article 传递信息")
    print("  conditional_edges - 审核结果决定是否继续")
    print("=" * 50)
