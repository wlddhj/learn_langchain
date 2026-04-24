"""
第19章 Demo 1：错误处理基础

演示 with_retry 和 with_fallbacks 的基本使用。
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


def demo_retry_basic():
    """重试机制基础"""
    print("=" * 60)
    print(f"Demo 19-1 (1/3): 重试机制 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    # 添加重试
    retry_llm = llm.with_retry(
        stop_after_attempt=3,  # 最多重试3次
        wait_to_exponentially_increase_timeout_initial=1,  # 初始等待1秒
        max_wait=10,  # 最大等待10秒
    )

    print("重试配置：")
    print("  - 最大重试次数: 3")
    print("  - 初始等待: 1秒")
    print("  - 最大等待: 10秒")
    print("  - 策略: 指数退避")
    print()

    try:
        result = retry_llm.invoke("你好")
        print(f"结果: {result.content[:50]}...")
    except Exception as e:
        print(f"最终失败: {e}")


def demo_fallback_basic():
    """回退机制"""
    print("=" * 60)
    print(f"Demo 19-1 (2/3): 回退机制 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 主模型
    primary_llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )

    # 回退模型（可以是不同模型）
    fallback_llm = ChatOpenAI(
        model=QWEN_MODEL,  # 实际可换成其他模型
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )

    # 配置回退
    llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

    print("回退配置：")
    print("  - 主模型: qwen-plus")
    print("  - 回退模型: qwen-plus（演示用相同模型）")
    print("  - 触发条件: 主模型失败")
    print()

    try:
        result = llm_with_fallback.invoke("你好")
        print(f"结果: {result.content[:50]}...")
    except Exception as e:
        print(f"所有模型都失败: {e}")


async def demo_async_retry():
    """异步重试"""
    print("=" * 60)
    print(f"Demo 19-1 (3/3): 异步重试 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    retry_llm = llm.with_retry(stop_after_attempt=2)

    print("异步重试配置：")
    print("  - 最大重试次数: 2")
    print()

    try:
        result = await retry_llm.ainvoke("你好")
        print(f"结果: {result.content[:50]}...")
    except Exception as e:
        print(f"最终失败: {e}")


async def main():
    demo_retry_basic()
    demo_fallback_basic()
    await demo_async_retry()

    print("=" * 60)
    print("Demo 19-1 完成!")
    print()
    print("错误处理核心方法：")
    print("  - with_retry: 自动重试失败请求")
    print("  - with_fallbacks: 主模型失败时切换备用模型")
    print()
    print("重试参数：")
    print("  - stop_after_attempt: 最大重试次数")
    print("  - wait_to_exponentially_increase: 指数退避等待")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())