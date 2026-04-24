"""
第20章 Demo 1：Token 成本优化基础

演示模型选择、Prompt 优化等成本节约方法。
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def demo_model_comparison():
    """模型成本对比"""
    print("=" * 60)
    print("Demo 20-1 (1/3): 模型成本对比")
    print("=" * 60)
    print()

    # 不同模型定价（假设，单位：元/千token）
    pricing = {
        "qwen-turbo": {"prompt": 0.002, "completion": 0.002},
        "qwen-plus": {"prompt": 0.004, "completion": 0.004},
        "qwen-max": {"prompt": 0.02, "completion": 0.06},
    }

    print("模型定价对比：")
    print("-" * 60)
    print(f"{'模型':<15} {'Prompt价格':<15} {'Completion价格':<15}")
    print("-" * 60)
    for model, price in pricing.items():
        print(f"{model:<15} ¥{price['prompt']}/千token {'':<5} ¥{price['completion']}/千token")

    print()
    print("选择建议：")
    print("  - 简单任务 → qwen-turbo（成本最低）")
    print("  - 中等复杂 → qwen-plus（平衡性价比）")
    print("  - 高复杂任务 → qwen-max（能力最强）")


def demo_prompt_optimization():
    """Prompt 优化"""
    print("=" * 60)
    print("Demo 20-1 (2/3): Prompt 优化")
    print("=" * 60)
    print()

    # 长 Prompt
    long_prompt = """
请详细分析以下问题，从多个角度进行阐述：
1. 首先介绍背景和历史
2. 然后分析当前状况
3. 接着讨论未来趋势
4. 最后给出总结和建议

请确保回答详尽、完整、专业。

问题：什么是 Python？
"""

    # 短 Prompt
    short_prompt = "用一句话解释 Python"

    print("Prompt 长度对比：")
    print("-" * 60)
    print(f"长 Prompt: {len(long_prompt)} 字符")
    print(f"短 Prompt: {len(short_prompt)} 字符")
    print()
    print("优化建议：")
    print("  - 去除不必要的指导语")
    print("  - 明确问题，避免冗长")
    print("  - 根据需求选择详细程度")


async def demo_max_tokens():
    """max_tokens 控制"""
    print("=" * 60)
    print(f"Demo 20-1 (3/3): max_tokens 控制 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

    # 无限制
    llm_unlimited = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
    )

    # 限制 token
    llm_limited = ChatOpenAI(
        model=QWEN_MODEL,
        temperature=0,
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        max_tokens=50,  # 限制输出50 token
    )

    prompt = "介绍一下 Python 语言"

    result_unlimited = await llm_unlimited.ainvoke(prompt)
    result_limited = await llm_limited.ainvoke(prompt)

    print("无限制输出:")
    print(f"  {result_unlimited.content[:100]}...")
    print()
    print("限制50 token 输出:")
    print(f"  {result_limited.content}")
    print()
    print("控制要点：")
    print("  - max_tokens 限制输出长度")
    print("  - 减少不必要的长输出")
    print("  - 简短回答场景特别适用")


async def main():
    demo_model_comparison()
    demo_prompt_optimization()
    await demo_max_tokens()

    print("=" * 60)
    print("Demo 20-1 完成!")
    print()
    print("成本优化策略：")
    print("  1. 模型选择：根据任务复杂度选合适模型")
    print("  2. Prompt 优化：简洁明确，减少冗余")
    print("  3. max_tokens：限制输出长度")
    print("  4. 缓存机制：避免重复调用")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())