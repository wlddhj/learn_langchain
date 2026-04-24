"""
第18章 Demo 2：并发控制与性能对比

演示 asyncio 并发控制和同步/异步性能对比。
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

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


async def demo_concurrency_comparison():
    """并发对比"""
    print("=" * 60)
    print(f"Demo 18-2 (1/3): 同步 vs 异步并发 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    prompts = ["问题1", "问题2", "问题3", "问题4"]

    # 同步顺序调用
    print("同步顺序调用:")
    start = time.time()
    for prompt in prompts:
        llm.invoke(prompt)
    sync_time = time.time() - start
    print(f"  耗时: {sync_time:.2f}s")
    print()

    # 异步并发调用
    print("异步并发调用:")
    start = time.time()
    await llm.abatch(prompts)
    async_time = time.time() - start
    print(f"  耗时: {async_time:.2f}s")
    print()

    print(f"性能提升: {sync_time / async_time:.1f}x")


async def demo_semaphore_control():
    """并发数限制"""
    print("=" * 60)
    print(f"Demo 18-2 (2/3): 并发数限制 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    prompts = [f"问题{i}" for i in range(10)]

    # 使用 Semaphore 限制并发数
    semaphore = asyncio.Semaphore(3)  # 最多3个并发

    async def limited_call(prompt: str):
        async with semaphore:
            print(f"  开始: {prompt}")
            result = await llm.ainvoke(prompt)
            print(f"  完成: {prompt}")
            return result

    print("限制并发数为 3:")
    print("-" * 60)

    start = time.time()
    results = await asyncio.gather(*[limited_call(p) for p in prompts])
    elapsed = time.time() - start

    print()
    print(f"总耗时: {elapsed:.2f}s")
    print(f"完成数量: {len(results)}")


async def demo_async_chain():
    """异步 Chain"""
    print("=" * 60)
    print(f"Demo 18-2 (3/3): 异步 Chain [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template("翻译成英文: {text}")
    chain = prompt | llm | StrOutputParser()

    # 同步调用
    sync_result = chain.invoke({"text": "你好世界"})
    print(f"同步结果: {sync_result}")

    # 异步调用
    async_result = await chain.ainvoke({"text": "你好世界"})
    print(f"异步结果: {async_result}")

    # 异步批量
    texts = ["你好", "世界", "谢谢"]
    async_results = await chain.abatch([{"text": t} for t in texts])
    print(f"批量结果: {async_results}")


async def main():
    await demo_concurrency_comparison()
    await demo_semaphore_control()
    await demo_async_chain()

    print("=" * 60)
    print("Demo 18-2 完成!")
    print()
    print("并发控制要点：")
    print("  - asyncio.gather: 并发执行多个任务")
    print("  - asyncio.Semaphore: 限制并发数")
    print("  - abatch: LangChain 内置批量并发")
    print()
    print("最佳实践：")
    print("  - 根据 API 限制设置并发数")
    print("  - 避免过多并发导致限流")
    print("  - 监控并发状态和错误")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())