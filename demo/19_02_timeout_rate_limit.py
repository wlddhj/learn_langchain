"""
第19章 Demo 2：超时控制与速率限制

演示超时处理和简单的速率限制。
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


async def demo_timeout():
    """超时控制"""
    print("=" * 60)
    print(f"Demo 19-2 (1/3): 超时控制 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    print("使用 asyncio.wait_for 设置超时:")
    print("  - 超时时间: 30秒")
    print()

    try:
        # 设置30秒超时
        result = await asyncio.wait_for(
            llm.ainvoke("你好"),
            timeout=30.0
        )
        print(f"结果: {result.content[:50]}...")
        print("调用成功，未超时")
    except asyncio.TimeoutError:
        print("请求超时！")


async def demo_rate_limit_simulation():
    """速率限制模拟"""
    print("=" * 60)
    print(f"Demo 19-2 (2/3): 速率限制模拟 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    # 简单的速率限制器（模拟）
    class SimpleRateLimiter:
        def __init__(self, max_per_minute=10):
            self.max_per_minute = max_per_minute
            self.calls = []

        def wait_if_needed(self):
            """如果需要则等待"""
            now = time.time()
            # 清理超过1分钟的记录
            self.calls = [c for c in self.calls if now - c < 60]

            if len(self.calls) >= self.max_per_minute:
                # 需要等待
                oldest = min(self.calls)
                wait_time = 60 - (now - oldest)
                print(f"  达到速率限制，等待 {wait_time:.1f}s")
                time.sleep(wait_time)
                self.calls = [c for c in self.calls if time.time() - c < 60]

            self.calls.append(time.time())

    rate_limiter = SimpleRateLimiter(max_per_minute=5)

    print("速率限制配置：")
    print("  - 每分钟最多: 5次")
    print()

    for i in range(6):
        rate_limiter.wait_if_needed()
        print(f"  调用 #{i+1}")
        llm.invoke(f"问题{i}")


async def demo_error_logging():
    """错误日志记录"""
    print("=" * 60)
    print(f"Demo 19-2 (3/3): 错误日志记录 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    error_log = []

    async def safe_call(prompt: str):
        """带错误记录的安全调用"""
        try:
            result = await llm.ainvoke(prompt)
            return {"success": True, "result": result.content}
        except Exception as e:
            error_log.append({
                "prompt": prompt,
                "error": str(e),
                "timestamp": time.time(),
            })
            return {"success": False, "error": str(e)}

    result = await safe_call("你好")
    print(f"调用结果: {result}")

    print()
    print("错误日志:")
    if error_log:
        for err in error_log:
            print(f"  - {err}")
    else:
        print("  无错误")


async def main():
    await demo_timeout()
    await demo_rate_limit_simulation()
    await demo_error_logging()

    print("=" * 60)
    print("Demo 19-2 完成!")
    print()
    print("超时与速率限制要点：")
    print("  - asyncio.wait_for: 设置超时")
    print("  - 速率限制: 防止 API 过载")
    print("  - 错误日志: 记录失败原因")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())