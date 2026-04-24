"""
第18章 Demo 1：异步编程基础

演示 ainvoke/abatch/astream 的基本使用。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
import asyncio
import time

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


async def demo_ainvoke():
    """异步单次调用"""
    print("=" * 60)
    print(f"Demo 18-1 (1/3): ainvoke 异步调用 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 同步调用
    start = time.time()
    sync_result = llm.invoke("你好")
    sync_time = time.time() - start

    # 异步调用
    start = time.time()
    async_result = await llm.ainvoke("你好")
    async_time = time.time() - start

    print(f"同步调用结果: {sync_result.content[:50]}...")
    print(f"同步调用耗时: {sync_time:.2f}s")
    print()
    print(f"异步调用结果: {async_result.content[:50]}...")
    print(f"异步调用耗时: {async_time:.2f}s")


async def demo_abatch():
    """异步批量调用"""
    print("=" * 60)
    print(f"Demo 18-1 (2/3): abatch 异步批量调用 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    prompts = ["你好", "介绍一下 Python", "什么是 RAG", "LangChain 是什么"]

    # 同步批量
    start = time.time()
    sync_results = llm.batch(prompts)
    sync_time = time.time() - start

    # 异步批量
    start = time.time()
    async_results = await llm.abatch(prompts)
    async_time = time.time() - start

    print(f"批量调用 {len(prompts)} 个请求：")
    print("-" * 60)
    print(f"同步批量耗时: {sync_time:.2f}s")
    print(f"异步批量耗时: {async_time:.2f}s")
    print()
    print("结果预览:")
    for i, result in enumerate(async_results):
        print(f"  [{i+1}] {result.content[:30]}...")


async def demo_astream():
    """异步流式调用"""
    print("=" * 60)
    print(f"Demo 18-1 (3/3): astream 异步流式 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    print("流式输出:")
    print("-" * 60)

    full_content = ""
    async for chunk in llm.astream("讲一个简短的故事"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_content += chunk.content

    print()
    print()
    print(f"完整内容长度: {len(full_content)} 字符")


async def main():
    await demo_ainvoke()
    await demo_abatch()
    await demo_astream()

    print("=" * 60)
    print("Demo 18-1 完成!")
    print()
    print("异步方法对比：")
    print("  | 方法 | 同步版本 | 异步版本 |")
    print("  |------|---------|---------|")
    print("  | 单次 | invoke | ainvoke |")
    print("  | 批量 | batch | abatch |")
    print("  | 流式 | stream | astream |")
    print()
    print("优势：")
    print("  - 并发处理多个请求")
    print("  - 不阻塞主线程")
    print("  - 适合高并发场景")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())