"""
第14章 Demo 3：子图与并行执行

演示子图复用、并行节点、结果合并。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 1. 子图：研究与写作
# ============================================================

class ResearchState(TypedDict):
    query: str
    findings: str


def research_search(state: ResearchState):
    """搜索信息"""
    response = llm.invoke([
        ("system", "你是研究专家。对给定主题进行简要分析，列出3个关键要点。"),
        ("human", f"研究主题: {state['query']}"),
    ])
    return {"findings": response.content}


def research_analyze(state: ResearchState):
    """深入分析"""
    response = llm.invoke([
        ("system", "基于搜索结果进行深入分析，给出总结（100字以内）。"),
        ("human", f"搜索结果: {state['findings']}"),
    ])
    return {"findings": response.content}


class WritingState(TypedDict):
    topic: str
    findings: str
    article: str


def writing_outline(state: WritingState):
    """拟定大纲"""
    response = llm.invoke([
        ("system", "基于研究发现，拟定文章大纲（3-5个要点）。"),
        ("human", f"主题: {state['topic']}\n研究: {state.get('findings', '无')}"),
    ])
    return {"article": f"大纲:\n{response.content}"}


def writing_content(state: WritingState):
    """撰写内容"""
    response = llm.invoke([
        ("system", "基于大纲撰写简短文章（150字以内）。"),
        ("human", f"{state.get('article', '')}"),
    ])
    return {"article": response.content}


def demo_subgraph():
    print("=" * 50)
    print(f"Demo 14-3 (1/3): 子图组合 [{QWEN_MODEL}]")
    print("=" * 50)

    # 研究子图
    research_builder = StateGraph(ResearchState)
    research_builder.add_node("search", research_search)
    research_builder.add_node("analyze", research_analyze)
    research_builder.add_edge(START, "search")
    research_builder.add_edge("search", "analyze")
    research_builder.add_edge("analyze", END)
    research_graph = research_builder.compile()

    # 写作子图
    writing_builder = StateGraph(WritingState)
    writing_builder.add_node("outline", writing_outline)
    writing_builder.add_node("write", writing_content)
    writing_builder.add_edge(START, "outline")
    writing_builder.add_edge("outline", "write")
    writing_builder.add_edge("write", END)
    writing_graph = writing_builder.compile()

    # 单独测试研究子图
    print("--- 测试研究子图 ---")
    result = research_graph.invoke({"query": "LangGraph 的优势", "findings": ""})
    print(f"研究发现: {preview(result['findings'], 120)}")
    print()

    # 单独测试写作子图
    print("--- 测试写作子图 ---")
    result = writing_graph.invoke({
        "topic": "LangGraph 的优势",
        "findings": "LangGraph 提供图结构工作流、状态持久化、条件路由",
        "article": "",
    })
    print(f"文章: {preview(result['article'], 150)}")
    print()

    print("子图可以独立编译和测试，也可以嵌入主图")
    print()


# ============================================================
# 2. 并行执行
# ============================================================

class ParallelState(TypedDict):
    question: str
    tech_analysis: str
    business_analysis: str
    final_summary: str


def tech_analyze(state: ParallelState):
    """技术分析（并行节点1）"""
    response = llm.invoke([
        ("system", "从技术角度分析，50字以内。"),
        ("human", state["question"]),
    ])
    return {"tech_analysis": response.content}


def business_analyze(state: ParallelState):
    """商业分析（并行节点2）"""
    response = llm.invoke([
        ("system", "从商业角度分析，50字以内。"),
        ("human", state["question"]),
    ])
    return {"business_analysis": response.content}


def combine_analysis(state: ParallelState):
    """合并分析结果"""
    response = llm.invoke([
        ("system", "综合技术和商业分析，给出最终总结（80字以内）。"),
        ("human", f"技术: {state['tech_analysis']}\n商业: {state['business_analysis']}"),
    ])
    return {"final_summary": response.content}


def demo_parallel():
    print("=" * 50)
    print(f"Demo 14-3 (2/3): 并行执行 [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(ParallelState)
    builder.add_node("tech", tech_analyze)
    builder.add_node("business", business_analyze)
    builder.add_node("combine", combine_analysis)

    # tech 和 business 从 START 并行开始
    builder.add_edge(START, "tech")
    builder.add_edge(START, "business")

    # 两者都完成后执行 combine
    builder.add_edge("tech", "combine")
    builder.add_edge("business", "combine")
    builder.add_edge("combine", END)

    graph = builder.compile()

    result = graph.invoke({
        "question": "AI Agent 技术在电商行业的应用前景如何？",
        "tech_analysis": "",
        "business_analysis": "",
        "final_summary": "",
    })

    print(f"问题: {result['question']}")
    print(f"技术分析: {preview(result['tech_analysis'], 100)}")
    print(f"商业分析: {preview(result['business_analysis'], 100)}")
    print(f"综合总结: {preview(result['final_summary'], 120)}")
    print()

    print("并行执行: tech 和 business 同时运行，完成后 combine 合并")
    print()


# ============================================================
# 3. stream 追踪并行执行
# ============================================================

def demo_parallel_stream():
    print("=" * 50)
    print(f"Demo 14-3 (3/3): 并行执行追踪 [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(ParallelState)
    builder.add_node("tech", tech_analyze)
    builder.add_node("business", business_analyze)
    builder.add_node("combine", combine_analysis)

    builder.add_edge(START, "tech")
    builder.add_edge(START, "business")
    builder.add_edge("tech", "combine")
    builder.add_edge("business", "combine")
    builder.add_edge("combine", END)

    graph = builder.compile()

    question = "大语言模型在企业中的落地挑战有哪些？"
    print(f"问题: {question}")
    print("-" * 40)

    for event in graph.stream({
        "question": question,
        "tech_analysis": "",
        "business_analysis": "",
        "final_summary": "",
    }):
        for node, output in event.items():
            if node == "tech":
                print(f"  [技术分析] {preview(output.get('tech_analysis', ''), 80)}")
            elif node == "business":
                print(f"  [商业分析] {preview(output.get('business_analysis', ''), 80)}")
            elif node == "combine":
                print(f"  [综合总结] {preview(output.get('final_summary', ''), 100)}")

    print()
    print("可以看到 tech 和 business 可能以任意顺序完成（并行）")
    print()


def preview(text, max_len):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


if __name__ == "__main__":
    demo_subgraph()
    demo_parallel()
    demo_parallel_stream()

    print("=" * 50)
    print("Demo 14-3 完成!")
    print()
    print("子图与并行要点:")
    print("  子图           - 独立编译和测试的模块化 Graph")
    print("  add_node(子图)  - 将子图作为节点嵌入主图")
    print("  并行边         - 多个节点从同一分叉点出发")
    print("  合并节点       - 等待所有并行节点完成后执行")
    print("  stream()       - 追踪并行节点的执行顺序")
    print("=" * 50)
