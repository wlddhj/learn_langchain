"""
第31章 Demo 1：LangServe 基础

演示 LangServe 的基本使用方式。
此 demo 展示服务端代码结构，需要实际运行服务才能测试。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def demo_langserve_intro():
    """LangServe 简介"""
    print("=" * 60)
    print("Demo 31-1: LangServe 基础介绍")
    print("=" * 60)
    print()

    print("LangServe 是 LangChain 官方的服务化工具：")
    print("-" * 60)
    print("""
核心特性：
- 快速部署：几行代码创建 REST API
- 类型安全：Pydantic 自动生成 Schema
- 流式支持：内置 SSE 流式响应
- Playground：内置交互式 UI
- 兼容 LangChain：直接集成 Runnable
- 异步支持：高性能异步处理
""")
    print()

    print("安装：")
    print("  pip install langserve")
    print("  pip install langserve[all]  # 安装所有依赖")
    print()


def demo_basic_server_code():
    """基础服务代码"""
    print("=" * 60)
    print("Demo 31-1 (2/4): 基础服务代码")
    print("=" * 60)
    print()

    code = """
# server.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 创建 FastAPI 应用
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="简单的 LangChain API 服务",
)

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 添加路由 - 一行代码即可
add_routes(app, llm, path="/chat")

# 运行: uvicorn server:app --reload
"""
    print("最简服务代码：")
    print("-" * 60)
    print(code)


def demo_api_endpoints():
    """API 端点说明"""
    print("=" * 60)
    print("Demo 31-1 (3/4): API 端点")
    print("=" * 60)
    print()

    print("LangServe 自动生成的端点：")
    print("-" * 60)
    print("""
| 端点 | 说明 | 用途 |
|------|------|------|
| /chat/invoke | 单次调用 | 标准请求响应 |
| /chat/stream | 流式调用 | SSE 实时输出 |
| /chat/batch | 批量调用 | 多请求并行 |
| /chat/playground | 交互界面 | 可视化测试 |

调用示例：

# invoke
curl -X POST http://localhost:8000/chat/invoke \\
  -H "Content-Type: application/json" \\
  -d '{"input": "你好"}'

# stream
curl -X POST http://localhost:8000/chat/stream \\
  -H "Content-Type: application/json" \\
  -d '{"input": "你好"}'

# batch
curl -X POST http://localhost:8000/chat/batch \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": ["你好", "再见"]}'
""")
    print()


def demo_client_usage():
    """客户端使用"""
    print("=" * 60)
    print("Demo 31-1 (4/4): Python 客户端")
    print("=" * 60)
    print()

    code = """
from langserve.client import RemoteRunnable

# 创建客户端
client = RemoteRunnable("http://localhost:8000/chat")

# invoke 调用
result = client.invoke("你好")
print(result)

# stream 调用
for chunk in client.stream("讲一个故事"):
    print(chunk.content, end="", flush=True)

# batch 调用
results = client.batch(["你好", "再见"])
for r in results:
    print(r)

# 异步调用
async def async_chat():
    result = await client.ainvoke("你好")
    print(result)

    async for chunk in client.astream("讲一个故事"):
        print(chunk.content)
"""
    print("Python 客户端代码：")
    print("-" * 60)
    print(code)


def demo_chain_routes():
    """Chain 服务化"""
    print("=" * 60)
    print("Chain 服务化示例")
    print("=" * 60)
    print()

    code = """
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建 Chain
prompt = ChatPromptTemplate.from_template("翻译成{language}: {text}")
chain = prompt | llm | StrOutputParser()

# 添加 Chain 路由
add_routes(app, chain, path="/translate")

# 多个 Chain
summary_chain = (
    ChatPromptTemplate.from_template("用一句话总结: {text}")
    | llm
    | StrOutputParser()
)
add_routes(app, summary_chain, path="/summary")
"""
    print("Chain 服务化：")
    print("-" * 60)
    print(code)


if __name__ == "__main__":
    demo_langserve_intro()
    demo_basic_server_code()
    demo_api_endpoints()
    demo_client_usage()
    demo_chain_routes()

    print("=" * 60)
    print("Demo 31-1 完成!")
    print()
    print("LangServe 核心概念：")
    print("  - add_routes(): 一行代码添加路由")
    print("  - 自动生成 invoke/stream/batch/playground 端点")
    print("  - RemoteRunnable: Python 客户端")
    print()
    print("使用流程：")
    print("  1. 创建 FastAPI app")
    print("  2. 创建 LangChain Runnable (LLM/Chain/Agent)")
    print("  3. add_routes(app, runnable, path)")
    print("  4. uvicorn server:app 运行")
    print("=" * 60)