"""
第17章 Demo 2：Token 追踪与日志记录

演示完整的 Token 追踪和日志记录系统。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import json

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


class TokenTracker(BaseCallbackHandler):
    """Token 追踪器"""

    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_records = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """记录 token 使用"""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]

            record = {
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            self.call_records.append(record)
            self.total_tokens += record["total_tokens"]
            self.prompt_tokens += record["prompt_tokens"]
            self.completion_tokens += record["completion_tokens"]

    def get_report(self) -> dict:
        """生成报告"""
        return {
            "total_calls": len(self.call_records),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_tokens_per_call": self.total_tokens / len(self.call_records) if self.call_records else 0,
            "records": self.call_records,
        }

    def save_log(self, filepath: str):
        """保存日志"""
        with open(filepath, "w") as f:
            json.dump(self.get_report(), f, indent=2, ensure_ascii=False)


class TimingTracker(BaseCallbackHandler):
    """时间追踪器"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.latency_records = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """记录开始时间"""
        self.start_time = datetime.now()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """记录结束时间"""
        self.end_time = datetime.now()
        latency = (self.end_time - self.start_time).total_seconds()
        self.latency_records.append(latency)

    def get_stats(self) -> dict:
        """统计"""
        if not self.latency_records:
            return {}
        return {
            "total_calls": len(self.latency_records),
            "avg_latency": sum(self.latency_records) / len(self.latency_records),
            "max_latency": max(self.latency_records),
            "min_latency": min(self.latency_records),
        }


def demo_token_tracking():
    """Token 追踪演示"""
    print("=" * 60)
    print(f"Demo 17-2 (1/3): Token 追踪 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    tracker = TokenTracker()

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        callbacks=[tracker],
    )

    # 多次调用
    prompts = ["你好", "介绍一下 Python", "什么是 RAG"]
    for prompt in prompts:
        llm.invoke(prompt)
        print(f"调用: {prompt}")

    print()
    report = tracker.get_report()
    print("Token 使用报告：")
    print("-" * 60)
    print(f"调用次数: {report['total_calls']}")
    print(f"总 Token: {report['total_tokens']}")
    print(f"Prompt Token: {report['prompt_tokens']}")
    print(f"Completion Token: {report['completion_tokens']}")
    print(f"平均每次: {report['avg_tokens_per_call']:.1f}")


def demo_timing_tracking():
    """时间追踪演示"""
    print("=" * 60)
    print(f"Demo 17-2 (2/3): 时间追踪 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    token_tracker = TokenTracker()
    timing_tracker = TimingTracker()

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        callbacks=[token_tracker, timing_tracker],
    )

    # 多次调用
    for i in range(3):
        llm.invoke(f"问题 {i+1}")

    print("时间统计：")
    print("-" * 60)
    timing_stats = timing_tracker.get_stats()
    print(f"调用次数: {timing_stats['total_calls']}")
    print(f"平均延迟: {timing_stats['avg_latency']:.2f}s")
    print(f"最大延迟: {timing_stats['max_latency']:.2f}s")
    print(f"最小延迟: {timing_stats['min_latency']:.2f}s")


def demo_cost_calculation():
    """成本计算示例"""
    print("=" * 60)
    print(f"Demo 17-2 (3/3): 成本估算")
    print("=" * 60)
    print()

    tracker = TokenTracker()

    llm = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        callbacks=[tracker],
    )

    llm.invoke("介绍一下 LangChain")

    report = tracker.get_report()

    # 模型定价（假设）
    pricing = {
        "gpt-4o-mini": {"prompt": 0.15 / 1000, "completion": 0.60 / 1000},
        "qwen-plus": {"prompt": 0.004 / 1000, "completion": 0.004 / 1000},
    }

    model_price = pricing.get(QWEN_MODEL, {"prompt": 0.01, "completion": 0.01})

    prompt_cost = report["prompt_tokens"] * model_price["prompt"]
    completion_cost = report["completion_tokens"] * model_price["completion"]
    total_cost = prompt_cost + completion_cost

    print("成本估算：")
    print("-" * 60)
    print(f"模型: {QWEN_MODEL}")
    print(f"Prompt Token: {report['prompt_tokens']} → ¥{prompt_cost:.4f}")
    print(f"Completion Token: {report['completion_tokens']} → ¥{completion_cost:.4f}")
    print(f"总成本: ¥{total_cost:.4f}")


if __name__ == "__main__":
    demo_token_tracking()
    demo_timing_tracking()
    demo_cost_calculation()

    print("=" * 60)
    print("Demo 17-2 完成!")
    print()
    print("Token 追踪核心功能：")
    print("  - 记录每次调用的 token 使用")
    print("  - 统计总 token 和平均 token")
    print("  - 结合定价计算成本")
    print()
    print("扩展建议：")
    print("  - 添加用户级 token 统计")
    print("  - 设置 token 预算限制")
    print("  - 实现成本告警机制")
    print("=" * 60)