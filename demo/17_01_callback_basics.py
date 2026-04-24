"""
第17章 Demo 1：回调系统基础

演示 BaseCallbackHandler 的基本使用。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")


class SimpleCallbackHandler(BaseCallbackHandler):
    """简单的回调处理器"""

    def __init__(self):
        self.token_count = 0
        self.call_count = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """LLM 开始调用"""
        self.call_count += 1
        print(f"[回调] LLM 开始调用 #{self.call_count}")
        print(f"[回调] Prompt: {prompts[0][:50]}...")

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        """LLM 结束调用"""
        # 获取 token 使用情况
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.token_count += usage.get("total_tokens", 0)
            print(f"[回调] Token 使用: {usage}")
        print(f"[回调] LLM 结束调用")

    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> None:
        """LLM 调用出错"""
        print(f"[回调] LLM 错误: {error}")

    def get_stats(self) -> dict:
        """获取统计"""
        return {
            "call_count": self.call_count,
            "token_count": self.token_count,
        }


def demo_basic_callback():
    """基础回调演示"""
    print("=" * 60)
    print(f"Demo 17-1 (1/3): 基础回调 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 创建回调处理器
    handler = SimpleCallbackHandler()

    # 创建带回调的 LLM
    llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        callbacks=[handler],
    )

    # 调用
    response = llm.invoke("你好，介绍一下自己")

    print(f"回答: {response.content[:100]}...")
    print()

    stats = handler.get_stats()
    print(f"统计: 调用次数={stats['call_count']}, Token数={stats['token_count']}")


def demo_streaming_callback():
    """流式回调"""
    print("=" * 60)
    print(f"Demo 17-1 (2/3): 流式回调 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    class StreamCallbackHandler(BaseCallbackHandler):
        def on_llm_start(self, serialized, prompts, **kwargs):
            print("[流式] 开始生成...")

        def on_llm_new_token(self, token: str, **kwargs):
            """每个新 token"""
            print(token, end="", flush=True)

        def on_llm_end(self, response, **kwargs):
            print("\n[流式] 完成")

    handler = StreamCallbackHandler()
    llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        callbacks=[handler],
        streaming=True,
    )

    print("流式输出:")
    llm.invoke("讲一个简短的故事")
    print()


def demo_multiple_callbacks():
    """多个回调处理器"""
    print("=" * 60)
    print(f"Demo 17-1 (3/3): 多个回调处理器 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    class LogCallback(BaseCallbackHandler):
        """日志回调"""
        def on_llm_start(self, serialized, prompts, **kwargs):
            print("[日志] 记录开始时间")

        def on_llm_end(self, response, **kwargs):
            print("[日志] 记录结束时间")

    class TokenCallback(BaseCallbackHandler):
        """Token 统计回调"""
        def __init__(self):
            self.tokens = 0

        def on_llm_end(self, response, **kwargs):
            if response.llm_output:
                self.tokens += response.llm_output.get("token_usage", {}).get("total_tokens", 0)
                print(f"[Token] 累计: {self.tokens}")

    # 多个回调
    log_handler = LogCallback()
    token_handler = TokenCallback()

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        callbacks=[log_handler, token_handler],
    )

    llm.invoke("你好")
    print()


if __name__ == "__main__":
    demo_basic_callback()
    demo_streaming_callback()
    demo_multiple_callbacks()

    print("=" * 60)
    print("Demo 17-1 完成!")
    print()
    print("回调系统核心概念：")
    print("  - BaseCallbackHandler: 回调基类")
    print("  - on_llm_start: LLM 开始调用")
    print("  - on_llm_end: LLM 结束调用")
    print("  - on_llm_new_token: 流式 token")
    print("  - callbacks 参数: 传入 LLM 或 Chain")
    print("=" * 60)