"""
第14章 Demo 1：状态持久化

演示 MemorySaver、thread_id 隔离、状态恢复。
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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# Graph 定义
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: ChatState):
    response = llm.invoke([
        ("system", "你是一个有帮助的助手，能记住之前的对话。用中文简洁回答。"),
        *state["messages"],
    ])
    return {"messages": [response]}


def build_chat_graph():
    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # 使用 MemorySaver 做检查点
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ============================================================
# 1. 持久化多轮对话
# ============================================================

def demo_persistent_chat():
    print("=" * 50)
    print(f"Demo 14-1 (1/3): 持久化多轮对话 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_chat_graph()
    config = {"configurable": {"thread_id": "chat-1"}}

    conversations = [
        "你好，我叫小明，我是一名 Python 开发者。",
        "我擅长 Django 和 FastAPI。",
        "我叫什么名字？我擅长什么？",  # 测试记忆
        "推荐一个适合我的学习路线。",      # 基于上下文推荐
    ]

    for msg in conversations:
        result = graph.invoke({"messages": [("user", msg)]}, config)
        ai_response = result["messages"][-1].content
        print(f"用户: {msg}")
        print(f"AI:   {ai_response[:120]}")
        print()

    print("MemorySaver 自动保存对话状态，同 thread_id 内共享记忆")
    print()


# ============================================================
# 2. thread_id 会话隔离
# ============================================================

def demo_session_isolation():
    print("=" * 50)
    print("Demo 14-1 (2/3): thread_id 会话隔离")
    print("=" * 50)

    graph = build_chat_graph()

    # 用户 A
    config_a = {"configurable": {"thread_id": "user-A"}}
    r1 = graph.invoke({"messages": [("user", "我叫小红，我喜欢画画")]}, config_a)
    print(f"[用户A] 说: 我叫小红，我喜欢画画")
    print(f"[用户A] AI: {r1['messages'][-1].content[:80]}")

    # 用户 B（不同 thread_id）
    config_b = {"configurable": {"thread_id": "user-B"}}
    r2 = graph.invoke({"messages": [("user", "我叫小蓝，我喜欢编程")]}, config_b)
    print(f"\n[用户B] 说: 我叫小蓝，我喜欢编程")
    print(f"[用户B] AI: {r2['messages'][-1].content[:80]}")

    # 用户 A 追问
    r3 = graph.invoke({"messages": [("user", "我叫什么？我喜欢什么？")]}, config_a)
    print(f"\n[用户A] 说: 我叫什么？我喜欢什么？")
    print(f"[用户A] AI: {r3['messages'][-1].content[:100]}")

    # 用户 B 追问
    r4 = graph.invoke({"messages": [("user", "我叫什么？我喜欢什么？")]}, config_b)
    print(f"\n[用户B] 说: 我叫什么？我喜欢什么？")
    print(f"[用户B] AI: {r4['messages'][-1].content[:100]}")
    print()

    print("结论: 不同 thread_id 的对话状态完全隔离")
    print()


# ============================================================
# 3. 状态检查与历史
# ============================================================

def demo_state_inspection():
    print("=" * 50)
    print(f"Demo 14-1 (3/3): 状态检查 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_chat_graph()
    config = {"configurable": {"thread_id": "inspect-1"}}

    # 执行几轮
    graph.invoke({"messages": [("user", "第一轮：记住数字 42")]}, config)
    graph.invoke({"messages": [("user", "第二轮：记住颜色 蓝色")]}, config)
    graph.invoke({"messages": [("user", "我让你记住的数字和颜色是什么？")]}, config)

    # 查看当前状态
    state = graph.get_state(config)
    print(f"当前状态消息数: {len(state.values.get('messages', []))}")
    print(f"下一步节点: {state.next}")
    print()

    # 查看检查点历史
    print("--- 检查点历史 ---")
    history = list(graph.get_state_history(config))
    for i, snap in enumerate(history):
        msg_count = len(snap.values.get("messages", []))
        print(f"  检查点 {i + 1}: {msg_count} 条消息, 下一步: {snap.next}")

    print()
    print("get_state() 查看当前状态，get_state_history() 回溯所有检查点")
    print()


if __name__ == "__main__":
    demo_persistent_chat()
    demo_session_isolation()
    demo_state_inspection()

    print("=" * 50)
    print("Demo 14-1 完成!")
    print()
    print("状态持久化要点:")
    print("  MemorySaver       - 内存检查点存储")
    print("  checkpointer      - 编译时传入，自动保存状态")
    print("  thread_id         - 隔离不同的对话/工作流")
    print("  get_state()       - 查看当前状态")
    print("  get_state_history() - 回溯所有检查点")
    print("  生产环境推荐      - SqliteSaver 或 PostgresSaver")
    print("=" * 50)
