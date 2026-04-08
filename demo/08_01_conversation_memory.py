"""
第8章 Demo 1：对话记忆管理

演示 RunnableWithMessageHistory、会话隔离、多轮对话记忆。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 会话存储
# ============================================================

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取指定会话的历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ============================================================
# Demo 1: 基础多轮对话
# ============================================================

def demo_basic_memory():
    print("=" * 50)
    print(f"Demo 8-1 (1/3): 多轮对话记忆 [{QWEN_MODEL}]")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的 AI 助手，能记住之前的对话内容。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "demo-user-1"}}

    # 多轮对话
    conversations = [
        "你好，我叫小明，我是一名 Python 开发者。",
        "我擅长 Django 和 FastAPI 框架。",
        "我叫什么名字？我擅长什么技术？",
        "推荐一个适合我的 Web 框架学习路线。",
    ]

    for user_msg in conversations:
        response = chain_with_history.invoke(
            {"question": user_msg},
            config=config,
        )
        print(f"用户: {user_msg}")
        print(f"AI:   {response}")
        print()

    # 查看历史消息数
    history = get_session_history("demo-user-1")
    print(f"当前历史消息数: {len(history.messages)}")
    print()


# ============================================================
# Demo 2: 会话隔离
# ============================================================

def demo_session_isolation():
    print("=" * 50)
    print("Demo 8-1 (2/3): 会话隔离")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    # 用户 A
    config_a = {"configurable": {"session_id": "user-A"}}
    r1 = chain_with_history.invoke({"question": "我叫小红，我喜欢画画"}, config=config_a)
    print(f"[用户A] 说: 我叫小红，我喜欢画画")
    print(f"[用户A] AI: {r1[:80]}")

    # 用户 B（不同会话）
    config_b = {"configurable": {"session_id": "user-B"}}
    r2 = chain_with_history.invoke({"question": "我叫小蓝，我喜欢编程"}, config=config_b)
    print(f"\n[用户B] 说: 我叫小蓝，我喜欢编程")
    print(f"[用户B] AI: {r2[:80]}")

    # 用户 A 追问
    r3 = chain_with_history.invoke({"question": "我叫什么？我喜欢什么？"}, config=config_a)
    print(f"\n[用户A] 说: 我叫什么？我喜欢什么？")
    print(f"[用户A] AI: {r3[:100]}")

    # 用户 B 追问
    r4 = chain_with_history.invoke({"question": "我叫什么？我喜欢什么？"}, config=config_b)
    print(f"\n[用户B] 说: 我叫什么？我喜欢什么？")
    print(f"[用户B] AI: {r4[:100]}")
    print()

    print("结论: 不同 session_id 的对话历史完全隔离，互不影响")
    print()


# ============================================================
# Demo 3: 带 System 角色的记忆
# ============================================================

def demo_role_memory():
    print("=" * 50)
    print("Demo 8-1 (3/3): 带角色设定的记忆对话")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深的技术面试官，正在面试 Python 开发者。请用专业但友好的语气提问和回应。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "interview-1"}}

    steps = [
        "你好，我是来面试 Python 开发岗位的。",
        "我有3年的 Django 开发经验。",
        "我最近在学习 FastAPI 和异步编程。",
        "根据我的背景，你觉得我应该重点准备哪些面试题？",
    ]

    for msg in steps:
        response = chain_with_history.invoke({"question": msg}, config=config)
        print(f"候选人: {msg}")
        print(f"面试官: {response[:150]}")
        print()


if __name__ == "__main__":
    demo_basic_memory()
    demo_session_isolation()
    demo_role_memory()

    print("=" * 50)
    print("Demo 8-1 完成!")
    print()
    print("对话记忆要点:")
    print("  RunnableWithMessageHistory - 自动管理对话历史")
    print("  MessagesPlaceholder        - 在 prompt 中预留历史位置")
    print("  session_id                  - 隔离不同用户/会话")
    print("  InMemoryChatMessageHistory  - 内存存储（开发用）")
    print("=" * 50)
