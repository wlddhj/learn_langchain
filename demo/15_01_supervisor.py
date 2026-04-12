"""
第15章 Demo 1：Supervisor 模式多 Agent

演示中心调度器、专业 Agent 分工协作。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# State
# ============================================================

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str


# ============================================================
# 专业 Agent 节点
# ============================================================

def research_agent(state: SupervisorState):
    """研究 Agent：搜索和分析信息"""
    response = llm.invoke([
        SystemMessage(content="你是研究专家。根据用户的问题进行深入分析，给出详细的研究结果。用中文回答，不超过200字。"),
        *state["messages"][-3:],  # 只看最近3条消息
    ])
    return {"messages": [AIMessage(content=f"[研究专家] {response.content}")]}


def writing_agent(state: SupervisorState):
    """写作 Agent：撰写和编辑文本"""
    response = llm.invoke([
        SystemMessage(content="你是写作专家。基于已有信息撰写优美的文章段落。用中文回答，不超过200字。"),
        *state["messages"][-3:],
    ])
    return {"messages": [AIMessage(content=f"[写作专家] {response.content}")]}


def summary_agent(state: SupervisorState):
    """总结 Agent：精炼和总结内容"""
    response = llm.invoke([
        SystemMessage(content="你是总结专家。将已有信息提炼成简洁的总结。用中文回答，不超过100字。"),
        *state["messages"][-3:],
    ])
    return {"messages": [AIMessage(content=f"[总结专家] {response.content}")]}


# ============================================================
# Supervisor 调度器
# ============================================================

def supervisor(state: SupervisorState):
    """主管 Agent：决定下一步由谁处理"""
    messages_text = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            messages_text += f"用户: {msg.content[:100]}\n"
        elif isinstance(msg, AIMessage):
            messages_text += f"AI: {msg.content[:100]}\n"

    response = llm.invoke([
        SystemMessage(content="""你是任务调度器。根据对话内容决定下一步：

- 需要搜索、研究、分析信息 → 回复 "researcher"
- 需要撰写、编辑、创作文本 → 回复 "writer"
- 需要总结、提炼、归纳内容 → 回复 "summarizer"
- 任务已完成 → 回复 "FINISH"

只回复一个词。"""),
        HumanMessage(content=f"对话历史:\n{messages_text}\n\n下一步应该交给谁？"),
    ])

    decision = response.content.strip().lower()

    if "researcher" in decision:
        return {"next_agent": "researcher"}
    elif "writer" in decision:
        return {"next_agent": "writer"}
    elif "summarizer" in decision:
        return {"next_agent": "summarizer"}
    else:
        return {"next_agent": "FINISH"}


def route_to_agent(state: SupervisorState):
    if state["next_agent"] == "FINISH":
        return END
    return state["next_agent"]


# ============================================================
# 构建多 Agent Graph
# ============================================================

def build_supervisor_graph():
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", research_agent)
    builder.add_node("writer", writing_agent)
    builder.add_node("summarizer", summary_agent)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_to_agent)
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("summarizer", "supervisor")

    return builder.compile()


# ============================================================
# Demo 1: 单 Agent 调度
# ============================================================

def demo_single_dispatch():
    print("=" * 50)
    print(f"Demo 15-1 (1/3): Supervisor 调度 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_supervisor_graph()

    questions = [
        "帮我研究一下 Python GIL 的工作原理",
        "帮我写一段描述春天的优美文字",
        "用一句话总结：AI Agent 是 LLM + 工具 + 自主决策循环",
    ]

    for q in questions:
        result = graph.invoke({
            "messages": [HumanMessage(content=q)],
            "next_agent": "",
        })

        print(f"用户: {q}")
        # 找到最后一条 AI 消息
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"回答: {msg.content[:150]}")
                break
        print()


# ============================================================
# Demo 2: 多 Agent 协作
# ============================================================

def demo_collaboration():
    print("=" * 50)
    print(f"Demo 15-1 (2/3): 多 Agent 协作 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_supervisor_graph()
    import time

    task = "研究 LangGraph 的核心特性，然后写一段介绍，最后总结"
    print(f"用户: {task}")
    print("-" * 40)

    result = graph.invoke({
        "messages": [HumanMessage(content=task)],
        "next_agent": "",
    })

    # 打印所有 Agent 的输出
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            agent_name = ""
            if "[研究专家]" in content:
                agent_name = "研究专家"
                content = content.replace("[研究专家] ", "")
            elif "[写作专家]" in content:
                agent_name = "写作专家"
                content = content.replace("[写作专家] ", "")
            elif "[总结专家]" in content:
                agent_name = "总结专家"
                content = content.replace("[总结专家] ", "")
            else:
                continue

            print(f"[{agent_name}] {content[:120]}")

    print()

    # 统计调度次数
    supervisor_calls = sum(1 for msg in result["messages"]
                          if isinstance(msg, AIMessage) and
                          msg.content and not msg.content.startswith("["))
    agent_outputs = sum(1 for msg in result["messages"]
                       if isinstance(msg, AIMessage) and
                       msg.content and msg.content.startswith("["))
    print(f"调度次数: {supervisor_calls}, Agent 输出: {agent_outputs}")
    print()


# ============================================================
# Demo 3: stream 追踪调度过程
# ============================================================

def demo_stream_trace():
    print("=" * 50)
    print(f"Demo 15-1 (3/3): 追踪调度过程 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_supervisor_graph()

    task = "分析 Python 和 Go 的区别"
    print(f"用户: {task}")
    print("-" * 40)

    for event in graph.stream({
        "messages": [HumanMessage(content=task)],
        "next_agent": "",
    }):
        for node, output in event.items():
            if node == "supervisor":
                next_a = output.get("next_agent", "?")
                print(f"  [Supervisor] → 调度给: {next_a}")
            elif node == "researcher":
                msg = output["messages"][-1]
                print(f"  [研究专家]   {msg.content[:80]}")
            elif node == "writer":
                msg = output["messages"][-1]
                print(f"  [写作专家]   {msg.content[:80]}")
            elif node == "summarizer":
                msg = output["messages"][-1]
                print(f"  [总结专家]   {msg.content[:80]}")

    print()


if __name__ == "__main__":
    demo_single_dispatch()
    demo_collaboration()
    demo_stream_trace()

    print("=" * 50)
    print("Demo 15-1 完成!")
    print()
    print("Supervisor 模式要点:")
    print("  Supervisor   - 中心调度器，决定任务分配")
    print("  专业 Agent   - 各有专长，独立处理任务")
    print("  循环调度     - Agent 完成后回到 Supervisor")
    print("  FINISH       - Supervisor 判断任务完成")
    print("  共享 State   - 通过 messages 传递上下文")
    print("=" * 50)
