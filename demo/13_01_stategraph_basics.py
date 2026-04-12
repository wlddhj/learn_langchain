"""
第13章 Demo 1：StateGraph 基础

演示 State、Node、Edge、消息状态的传递。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 1. 最简单的 StateGraph
# ============================================================

class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: SimpleState):
    """聊天机器人节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def demo_simple_graph():
    print("=" * 50)
    print(f"Demo 13-1 (1/3): 最简单的 StateGraph [{QWEN_MODEL}]")
    print("=" * 50)

    # 构建图
    builder = StateGraph(SimpleState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile()

    # 运行
    result = graph.invoke({"messages": [("user", "用一句话介绍 LangGraph")]})
    print(f"用户: 用一句话介绍 LangGraph")
    print(f"AI:   {result['messages'][-1].content}")
    print()

    # 查看所有消息
    print(f"消息链: {len(result['messages'])} 条")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        content = msg.content[:80]
        print(f"  [{role}] {content}")
    print()


# ============================================================
# 2. 多节点串联
# ============================================================

class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    summary: str
    translation: str


def summarize(state: PipelineState):
    """总结节点"""
    response = llm.invoke([
        ("system", "你是一个总结专家。用2-3句话总结用户输入的内容。"),
        ("human", f"请总结：{state['topic']}"),
    ])
    return {"summary": response.content, "messages": [response]}


def translate(state: PipelineState):
    """翻译节点"""
    response = llm.invoke([
        ("system", "你是一个翻译专家。将中文翻译成简洁的英文。"),
        ("human", f"翻译以下内容：{state['summary']}"),
    ])
    return {"translation": response.content, "messages": [response]}


def demo_pipeline():
    print("=" * 50)
    print(f"Demo 13-1 (2/3): 多节点串联 [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(PipelineState)
    builder.add_node("summarize", summarize)
    builder.add_node("translate", translate)

    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "translate")
    builder.add_edge("translate", END)

    graph = builder.compile()

    result = graph.invoke({
        "topic": "LangGraph 是一个用于构建有状态、多参与者 AI 应用的框架。"
                 "它基于图结构定义工作流，支持条件路由、循环、并行执行和状态持久化。"
                 "与 LangChain Agent 相比，LangGraph 提供了对执行流程的完全控制。",
        "messages": [],
    })

    print(f"输入: LangGraph 是一个用于构建有状态...")
    print(f"总结: {state_preview(result['summary'], 100)}")
    print(f"翻译: {state_preview(result['translation'], 150)}")
    print()
    print(f"节点执行顺序: START → summarize → translate → END")
    print(f"总消息数: {len(result['messages'])}")
    print()


# ============================================================
# 3. 自定义 State（非消息类型）
# ============================================================

class AnalysisState(TypedDict):
    question: str
    thinking: str
    answer: str
    confidence: str


def think(state: AnalysisState):
    """思考节点"""
    response = llm.invoke([
        ("system", "你是一个分析助手。先分析问题，给出思考过程（不超过50字）。"),
        ("human", state["question"]),
    ])
    return {"thinking": response.content}


def answer(state: AnalysisState):
    """回答节点"""
    response = llm.invoke([
        ("system", "基于思考过程给出简洁回答（不超过80字），并评估置信度（高/中/低）。"),
        ("human", f"问题: {state['question']}\n思考: {state['thinking']}"),
    ])
    return {"answer": response.content, "confidence": "高"}


def demo_custom_state():
    print("=" * 50)
    print(f"Demo 13-1 (3/3): 自定义 State [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(AnalysisState)
    builder.add_node("think", think)
    builder.add_node("answer", answer)

    builder.add_edge(START, "think")
    builder.add_edge("think", "answer")
    builder.add_edge("answer", END)

    graph = builder.compile()

    result = graph.invoke({
        "question": "Python 和 Go 语言各有什么优缺点？",
        "thinking": "",
        "answer": "",
        "confidence": "",
    })

    print(f"问题: {result['question']}")
    print(f"思考: {state_preview(result['thinking'], 100)}")
    print(f"回答: {state_preview(result['answer'], 150)}")
    print(f"置信度: {result['confidence']}")
    print()


def state_preview(text, max_len):
    """截取文本预览"""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


if __name__ == "__main__":
    demo_simple_graph()
    demo_pipeline()
    demo_custom_state()

    print("=" * 50)
    print("Demo 13-1 完成!")
    print()
    print("StateGraph 基础要点:")
    print("  State       - 定义图的状态结构（TypedDict）")
    print("  add_messages - 自动处理消息追加的 reducer")
    print("  add_node    - 添加处理节点（函数）")
    print("  add_edge    - 添加节点间的固定边")
    print("  START/END   - 图的入口和出口")
    print("  compile()   - 编译图为可执行对象")
    print("  invoke()    - 运行图并获取结果")
    print("=" * 50)
