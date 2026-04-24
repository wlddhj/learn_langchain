"""
第25章 Demo 1：Streamlit 聊天界面

演示 Streamlit 聊天界面的基本实现。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
import asyncio

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")


def demo_streamlit_code():
    """Streamlit 代码示例"""
    print("=" * 60)
    print("Demo 25-1: Streamlit 聊天界面")
    print("=" * 60)
    print()

    code = """
import streamlit as st
from langchain_openai import ChatOpenAI

# 页面配置
st.set_page_config(page_title="AI 助手", page_icon="🤖")

st.title("🤖 AI 聊天助手")

# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 初始化历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("输入消息"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 回复（流式）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 运行: streamlit run app.py
"""

    print("Streamlit 聊天代码：")
    print("-" * 60)
    print(code)


async def demo_streamlit_simulation():
    """模拟 Streamlit 流式输出"""
    print("=" * 60)
    print(f"Demo 25-1 (2/3): 模拟流式输出 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    print("模拟 Streamlit 流式输出效果：")
    print("-" * 60)
    print()

    print("用户: 你好")
    print("AI: ", end="", flush=True)

    full_response = ""
    for chunk in llm.stream("你好"):
        if chunk.content:
            full_response += chunk.content
            print(chunk.content, end="", flush=True)

    print()
    print()
    print(f"完整回复: {full_response}")


def demo_streamlit_features():
    """Streamlit 功能"""
    print("=" * 60)
    print("Demo 25-1 (3/3): Streamlit 功能")
    print("=" * 60)
    print()

    print("Streamlit 常用功能：")
    print("-" * 60)
    print("""
| 功能 | 代码 | 用途 |
|------|------|------|
| 聊天输入 | st.chat_input | 用户消息输入 |
| 聊天消息 | st.chat_message | 显示对话消息 |
| 空占位符 | st.empty() | 流式输出更新 |
| 会话状态 | st.session_state | 保存对话历史 |
| 侧边栏 | st.sidebar | 配置面板 |
| 缓存 | @st.cache_resource | 缓存模型初始化 |
""")
    print()

    print("运行命令：")
    print("  streamlit run app.py")
    print("  streamlit run app.py --server.port 8080")


async def main():
    demo_streamlit_code()
    await demo_streamlit_simulation()
    demo_streamlit_features()

    print("=" * 60)
    print("Demo 25-1 完成!")
    print()
    print("Streamlit 优势：")
    print("  - 简单快速：几行代码创建界面")
    print("  - 内置聊天组件：chat_input/chat_message")
    print("  - 流式支持：实时显示输出")
    print("  - 会话管理：session_state 保存历史")
    print()
    print("部署选项：")
    print("  - Streamlit Cloud（免费）")
    print("  - 本地运行")
    print("  - Docker 容器")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())