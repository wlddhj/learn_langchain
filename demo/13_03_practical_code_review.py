"""
第13章 Demo 3：实战 —— 代码审查 Graph

多节点串联工作流：查找问题 → 给出建议 → 改进代码。
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
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# State 定义
# ============================================================

class CodeReviewState(TypedDict):
    code: str
    language: str
    issues: str
    suggestions: str
    improved_code: str
    quality_score: str


# ============================================================
# 节点函数
# ============================================================

def find_issues(state: CodeReviewState):
    """查找代码问题"""
    prompt = ChatPromptTemplate.from_template(
        "分析以下 {language} 代码的问题。列出所有发现的问题（格式、安全、性能、可读性等）。\n"
        "如果没有问题，回复'代码质量良好'。\n\n"
        "```{language}\n{code}\n```"
    )
    chain = prompt | llm
    result = chain.invoke({"language": state["language"], "code": state["code"]})
    return {"issues": result.content}


def suggest_fixes(state: CodeReviewState):
    """给出修复建议"""
    prompt = ChatPromptTemplate.from_template(
        "基于以下代码问题，给出具体的修复建议。\n\n"
        "原代码:\n```{language}\n{code}\n```\n\n"
        "发现的问题:\n{issues}\n\n"
        "请给出修复建议（每条建议对应一个问题）。"
    )
    chain = prompt | llm
    result = chain.invoke({
        "language": state["language"],
        "code": state["code"],
        "issues": state["issues"],
    })
    return {"suggestions": result.content}


def improve_code(state: CodeReviewState):
    """改进代码"""
    prompt = ChatPromptTemplate.from_template(
        "根据建议改进代码。只输出改进后的完整代码，不要解释。\n\n"
        "原代码:\n```{language}\n{code}\n```\n\n"
        "修复建议:\n{suggestions}"
    )
    chain = prompt | llm
    result = chain.invoke({
        "language": state["language"],
        "code": state["code"],
        "suggestions": state["suggestions"],
    })
    return {"improved_code": result.content}


def quality_check(state: CodeReviewState):
    """质量评估"""
    prompt = ChatPromptTemplate.from_template(
        "评估代码改进的质量。给出一个评分（A/B/C/D）和一句话评价。\n\n"
        "原代码:\n{code}\n\n"
        "改进后:\n{improved_code}\n\n"
        "请回复格式: '评分: X - 一句话评价'"
    )
    chain = prompt | llm
    result = chain.invoke({
        "code": state["code"],
        "improved_code": state["improved_code"],
    })
    return {"quality_score": result.content}


# ============================================================
# 构建审查 Graph
# ============================================================

def build_review_graph():
    builder = StateGraph(CodeReviewState)
    builder.add_node("find_issues", find_issues)
    builder.add_node("suggest_fixes", suggest_fixes)
    builder.add_node("improve_code", improve_code)
    builder.add_node("quality_check", quality_check)

    builder.add_edge(START, "find_issues")
    builder.add_edge("find_issues", "suggest_fixes")
    builder.add_edge("suggest_fixes", "improve_code")
    builder.add_edge("improve_code", "quality_check")
    builder.add_edge("quality_check", END)

    return builder.compile()


# ============================================================
# 测试代码
# ============================================================

CODE_SAMPLES = {
    "python_bad": {
        "language": "python",
        "code": """def get_user_data(id):
    import pymysql
    conn = pymysql.connect(host='localhost', user='root', password='123456', db='users')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = " + str(id))
    result = cursor.fetchone()
    conn.close()
    return result

def calc(a,b,op):
    if op==1: return a+b
    if op==2: return a-b
    if op==3: return a*b
    if op==4: return a/b
""",
        "desc": "包含SQL注入、硬编码密码、缺少异常处理、变量命名差",
    },
    "python_medium": {
        "language": "python",
        "code": """def process_data(data):
    results = []
    for item in data:
        if item['status'] == 'active':
            if item['score'] > 80:
                results.append({'name': item['name'], 'level': 'A'})
            elif item['score'] > 60:
                results.append({'name': item['name'], 'level': 'B'})
            else:
                results.append({'name': item['name'], 'level': 'C'})
    return results
""",
        "desc": "嵌套过深、缺少类型注解、可简化",
    },
}


# ============================================================
# Demo 1: 完整代码审查流程
# ============================================================

def demo_full_review():
    print("=" * 50)
    print(f"代码审查 Graph (1/2): 完整流程 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_review_graph()
    sample = CODE_SAMPLES["python_bad"]

    print(f"说明: {sample['desc']}")
    print("-" * 40)

    result = graph.invoke({
        "code": sample["code"],
        "language": sample["language"],
        "issues": "",
        "suggestions": "",
        "improved_code": "",
        "quality_score": "",
    })

    print(f"[1] 发现的问题:")
    print(preview(result["issues"], 300))
    print()
    print(f"[2] 修复建议:")
    print(preview(result["suggestions"], 300))
    print()
    print(f"[3] 改进后代码:")
    print(preview(result["improved_code"], 300))
    print()
    print(f"[4] 质量评估: {preview(result['quality_score'], 100)}")
    print()


# ============================================================
# Demo 2: stream 追踪审查过程
# ============================================================

def demo_stream_review():
    print("=" * 50)
    print(f"代码审查 Graph (2/2): stream 追踪 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_review_graph()
    sample = CODE_SAMPLES["python_medium"]

    print(f"说明: {sample['desc']}")
    print("-" * 40)

    node_names = {
        "find_issues": "查找问题",
        "suggest_fixes": "修复建议",
        "improve_code": "改进代码",
        "quality_check": "质量评估",
    }

    for event in graph.stream({
        "code": sample["code"],
        "language": sample["language"],
        "issues": "",
        "suggestions": "",
        "improved_code": "",
        "quality_score": "",
    }):
        for node, output in event.items():
            cn_name = node_names.get(node, node)
            if node == "quality_check":
                content = output.get("quality_score", "")
            elif node == "find_issues":
                content = output.get("issues", "")
            elif node == "suggest_fixes":
                content = output.get("suggestions", "")
            elif node == "improve_code":
                content = output.get("improved_code", "")
            else:
                content = str(output)
            print(f"  [{cn_name}] {preview(content, 80)}")

    print()
    print("执行流程: find_issues → suggest_fixes → improve_code → quality_check")
    print()


def preview(text, max_len):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


if __name__ == "__main__":
    demo_full_review()
    demo_stream_review()

    print("=" * 50)
    print("Demo 13-3 完成!")
    print()
    print("代码审查 Graph 要点:")
    print("  StateGraph      - 定义多步骤工作流")
    print("  自定义 State    - 传递 code/issues/suggestions 等")
    print("  节点串联        - 每个节点专注一个任务")
    print("  ChatPromptTemplate - 结构化 prompt 模板")
    print("  stream()        - 逐步追踪每个节点的输出")
    print("=" * 50)
