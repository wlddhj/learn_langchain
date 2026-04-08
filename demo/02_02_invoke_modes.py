"""
第2章 Demo 2：调用方式对比 —— 同步、批量、流式

演示 invoke / batch / stream 三种调用方式的区别和适用场景。
可独立运行，需要 GLM_API_KEY。
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GLM_API_KEY") or os.environ["GLM_API_KEY"].startswith("your-"):
    print("错误: 未设置 GLM_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(
    model=GLM_MODEL,
    temperature=0,
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
)


def demo_sync_invoke():
    """同步调用：最基本的方式"""
    print("=" * 50)
    print(f"Demo 2-2 (1/3): 同步调用 (invoke) [{GLM_MODEL}]")
    print("=" * 50)
    print("适用场景: 脚本、简单工具、交互式命令行")
    print()

    questions = [
        "什么是 API？用一句话回答。",
        "什么是 JSON？用一句话回答。",
        "什么是 REST？用一句话回答。",
    ]

    for q in questions:
        start = time.time()
        response = llm.invoke(q)
        elapsed = time.time() - start
        print(f"Q: {q}")
        print(f"A: {response.content.strip()}")
        print(f"   耗时: {elapsed:.2f}s")
        print()


def demo_batch_invoke():
    """批量调用：一次发送多个请求"""
    print("=" * 50)
    print(f"Demo 2-2 (2/3): 批量调用 (batch) [{GLM_MODEL}]")
    print("=" * 50)
    print("适用场景: 需要同时处理多个独立请求")
    print()

    questions = [
        "Python 是什么？一句话回答。",
        "JavaScript 是什么？一句话回答。",
        "Rust 是什么？一句话回答。",
    ]

    print(f"发送 {len(questions)} 个请求...")

    start = time.time()
    responses = llm.batch(questions)
    elapsed = time.time() - start

    print(f"批量完成，总耗时: {elapsed:.2f}s")
    print()

    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r.content.strip()}")
        print()

    print("对比: batch 比逐个 invoke 更高效（并行处理）")
    print()


def demo_streaming():
    """流式输出：逐 token 输出"""
    print("=" * 50)
    print(f"Demo 2-2 (3/3): 流式输出 (stream) [{GLM_MODEL}]")
    print("=" * 50)
    print("适用场景: 聊天界面、实时响应、长文本生成")
    print()

    question = "用三句话解释什么是深度学习。"

    print(f"Q: {question}")
    print("A: ", end="", flush=True)

    start = time.time()
    first_token_time = None
    token_count = 0

    for chunk in llm.stream(question):
        if chunk.content:
            if first_token_time is None:
                first_token_time = time.time()
            print(chunk.content, end="", flush=True)
            token_count += 1

    elapsed = time.time() - start
    print()
    print()
    if first_token_time:
        print(f"首个 token 延迟: {first_token_time - start:.2f}s")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"收到 {token_count} 个 chunk")
    print()
    print("优势: 用户无需等待全部生成完毕即可看到输出")


async def demo_async():
    """异步调用（补充演示）"""
    print()
    print("=" * 50)
    print(f"补充: 异步调用 (ainvoke) [{GLM_MODEL}]")
    print("=" * 50)
    print("适用场景: Web 服务 (FastAPI)、高并发场景")
    print()

    async def single_call(question: str):
        response = await llm.ainvoke(question)
        return response.content.strip()

    start = time.time()
    results = await asyncio.gather(
        single_call("什么是 CPU？一句话回答。"),
        single_call("什么是 GPU？一句话回答。"),
        single_call("什么是 TPU？一句话回答。"),
    )
    elapsed = time.time() - start

    for q, a in zip(["CPU", "GPU", "TPU"], results):
        print(f"{q}: {a}")

    print(f"\n3 个并发请求总耗时: {elapsed:.2f}s")
    print("异步并发比串行快得多！")


if __name__ == "__main__":
    demo_sync_invoke()
    demo_batch_invoke()
    demo_streaming()

    # 异步演示
    print()
    asyncio.run(demo_async())

    print()
    print("=" * 50)
    print("Demo 2-2 完成!")
    print()
    print("调用方式总结:")
    print("  invoke  - 单次同步调用，适合脚本")
    print("  batch   - 批量调用，适合多请求处理")
    print("  stream  - 流式输出，适合实时交互")
    print("  ainvoke - 异步调用，适合 Web 服务")
    print("=" * 50)
