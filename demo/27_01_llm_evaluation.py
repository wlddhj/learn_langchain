"""
第27章 Demo：LLM 评估体系演示

展示基础评估方法：Token评估、响应时间、关键词检查。
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


def demo_evaluation_types():
    """评估方法总览"""
    print("=" * 60)
    print("Demo 27-1 (1/3): 评估方法总览")
    print("=" * 60)
    print()

    print("评估方法金字塔：")
    print("-" * 60)
    print("""
┌─────────────────────────────────────────────────┐
│              评估方法金字塔                        │
│                                                 │
│           人工评估（最准确但成本高）               │
│         ─────────────────────                   │
│        LLM-as-Judge（自动评估）                  │
│       ───────────────────────────               │
│      规则评估（确定性指标）                       │
│    ───────────────────────────────              │
│   自动指标评估（基础指标）                        │
└─────────────────────────────────────────────────┘

推荐策略：
├── 开发阶段：自动指标 + LLM-as-Judge（快速迭代）
├── 测试阶段：规则评估 + LLM-as-Judge（覆盖场景）
└── 上线阶段：人工抽样评估（质量把关）
"")


async def demo_token_evaluation():
    """Token 用量评估"""
    print("=" * 60)
    print(f"Demo 27-1 (2/3): Token 用量评估 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    print("Token 用量评估：")
    print("-" * 60)

    test_prompt = "请用简洁的语言解释什么是机器学习"
    response = await llm.ainvoke(test_prompt)

    # 获取 token 用量
    usage = response.response_metadata.get("token_usage", {})

    print(f"问题: {test_prompt}")
    print(f"回答: {response.content[:100]}...")
    print()
    print(f"Token 用量：")
    print(f"  输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
    print(f"  输出 tokens: {usage.get('completion_tokens', 'N/A')}")
    print(f"  总 tokens: {usage.get('total_tokens', 'N/A')}")

    print()
    print("成本估算函数：")
    code = """
def calculate_cost(usage, model="qwen-plus"):
    # 价格参考（人民币/1M tokens）
    prices = {
        "qwen-plus": {"input": 0.4, "output": 2.0},
        "qwen-turbo": {"input": 0.3, "output": 1.2},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    price = prices.get(model, {"input": 0, "output": 0})

    input_cost = usage['prompt_tokens'] * price["input"] / 1_000_000
    output_cost = usage['completion_tokens'] * price["output"] / 1_000_000

    return input_cost + output_cost
"""
    print(code)


async def demo_latency_evaluation():
    """响应时间评估"""
    print("=" * 60)
    print(f"Demo 27-1 (3/3): 响应时间评估 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    print("响应延迟评估：")
    print("-" * 60)

    test_prompt = "什么是 LangChain？"
    num_runs = 3

    latencies = []
    for i in range(num_runs):
        start = time.time()
        response = await llm.ainvoke(test_prompt)
        elapsed = time.time() - start
        latencies.append(elapsed)
        print(f"第{i+1}次: {elapsed:.2f}s")

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print()
    print(f"统计结果：")
    print(f"  平均延迟: {avg_latency:.2f}s")
    print(f"  最小延迟: {min_latency:.2f}s")
    print(f"  最大延迟: {max_latency:.2f}s")

    print()
    print("延迟评估函数：")
    code = """
def evaluate_latency(llm, prompt, num_runs=5):
    latencies = []
    for i in range(num_runs):
        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        latencies.append(elapsed)

    return {
        "avg_latency": sum(latencies) / len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
    }
"""
    print(code)


def demo_keyword_check():
    """关键词检查"""
    print("=" * 60)
    print("Demo 27-1 (补充): 关键词覆盖检查")
    print("=" * 60)
    print()

    print("关键词覆盖评估：")
    print("-" * 60)

    # 模拟评估
    test_answer = "机器学习是人工智能的一个分支，它使计算机能够从数据中自动学习模式和规律，无需明确编程。"
    required_keywords = ["机器学习", "人工智能", "数据"]

    content = test_answer.lower()
    covered = [kw for kw in required_keywords if kw.lower() in content]
    missing = [kw for kw in required_keywords if kw.lower() not in content]

    coverage_rate = len(covered) / len(required_keywords) if required_keywords else 1.0

    print(f"回答: {test_answer}")
    print(f"必需关键词: {required_keywords}")
    print(f"覆盖关键词: {covered}")
    print(f"缺失关键词: {missing}")
    print(f"覆盖率: {coverage_rate:.2%}")

    print()
    print("关键词检查函数：")
    code = """
def evaluate_keywords(response, required_keywords, prohibited_keywords=None):
    content = response.content.lower()

    covered = [kw for kw in required_keywords if kw.lower() in content]
    missing = [kw for kw in required_keywords if kw.lower() not in content]

    violations = []
    if prohibited_keywords:
        violations = [kw for kw in prohibited_keywords if kw.lower() in content]

    coverage_rate = len(covered) / len(required_keywords)

    return {
        "coverage_rate": coverage_rate,
        "covered_keywords": covered,
        "missing_keywords": missing,
        "violations": violations,
        "passed": coverage_rate >= 0.8 and len(violations) == 0,
    }
"""
    print(code)


def demo_llm_as_judge():
    """LLM-as-Judge 概念"""
    print("=" * 60)
    print("Demo 27-1 (补充): LLM-as-Judge")
    print("=" * 60)
    print()

    print("LLM-as-Judge 原理：")
    print("-" * 60)
    print("""
用强模型评估另一个模型的输出：
├── 适合评估主观质量：回答质量、逻辑性
├── 成本：每次评估消耗 tokens
└── 准确度：约 80-90%（与人工评估接近）

评估维度：
├── 相关性：回答是否与问题相关
├── 准确性：回答内容是否正确
├── 完整性：回答是否完整
├── 清晰度：回答是否易于理解
└── 有帮助性：回答是否有帮助
"")

    print("LLM-as-Judge Prompt 示例：")
    code = """
评估 Prompt:

请评估以下回答的质量。

问题: {question}
回答: {answer}

请从以下维度评分（1-5分）：
- 相关性：回答是否与问题相关
- 准确性：回答内容是否正确
- 完整性：回答是否完整
- 清晰度：回答是否易于理解

输出格式（JSON）：
{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "overall": X, "reason": "简要说明"}
"""
    print(code)


async def main():
    demo_evaluation_types()
    await demo_token_evaluation()
    await demo_latency_evaluation()
    demo_keyword_check()
    demo_llm_as_judge()

    print("=" * 60)
    print("Demo 27-1 完成!")
    print()
    print("评估要点：")
    print("  - 自动指标：Token用量、延迟（快速筛选）")
    print("  - 规则评估：关键词覆盖、格式检查（确定性）")
    print("  - LLM-as-Judge：用强模型评估质量（自动化）")
    print()
    print("评估场景：")
    print("  | 场景 | 核心指标 |")
    print("  |------|---------|")
    print("  | 问答系统 | 准确性、完整性 |")
    print("  | RAG 系统 | 忠实度、相关性 |")
    print("  | Agent 系统 | 任务完成率 |")
    print()
    print("最佳实践：")
    print("  - 开发阶段：自动评估快速迭代")
    print("  - 测试阶段：规则+LLM全面覆盖")
    print("  - 上线阶段：人工抽样把关")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())