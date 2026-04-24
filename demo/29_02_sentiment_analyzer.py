"""
第29章 Demo 2：情绪分析器

演示客服系统中的情绪分析功能。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path

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


def demo_sentiment_analysis():
    """情绪分析演示"""
    print("=" * 60)
    print(f"Demo 29-2 (1/3): 情绪分析 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    sentiment_types = """
可选情绪类型：
- positive: 积极情绪（满意、感谢、开心等）
- neutral: 中性情绪（一般询问、无明显情绪等）
- negative: 负面情绪（不满、抱怨等）
- angry: 愤怒情绪（强烈不满、投诉等）
"""

    analyze_prompt = ChatPromptTemplate.from_template("""
分析用户消息的情绪：

用户消息：{message}

{sentiment_types}

请输出：
1. 情绪类型
2. 情绪分数（-1到1，-1为极度负面，1为极度正面）
3. 关键词列表（表达情绪的关键词）

格式：情绪: xxx, 分数: xxx, 关键词: [xxx, xxx]
""")

    test_messages = [
        "非常满意你们的服务！",
        "我想问一下这个产品",
        "服务太差了，等了很久",
        "这什么垃圾产品！我要投诉！",
        "感谢客服小姐姐的帮助，问题解决了",
    ]

    print("情绪分析测试：")
    print("-" * 60)

    for msg in test_messages:
        result = (analyze_prompt | llm | StrOutputParser()).invoke({
            "message": msg,
            "sentiment_types": sentiment_types,
        })

        print(f"用户消息: {msg}")
        print(f"分析结果: {result}")
        print()


def demo_response_strategy():
    """基于情绪的响应策略"""
    print("=" * 60)
    print(f"Demo 29-2 (2/3): 响应策略选择 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    strategies = {
        "positive": "友好热情回应，鼓励用户",
        "neutral": "专业高效回应，简洁明了",
        "negative": "耐心安抚，提供解决方案",
        "angry": "优先安抚，快速处理，必要时转人工",
    }

    print("情绪 → 响应策略：")
    print("-" * 60)
    for sentiment, strategy in strategies.items():
        print(f"  {sentiment:<10}: {strategy}")
    print()

    # 演示策略应用
    angry_message = "这什么垃圾产品！买了不到一周就坏了！"

    response_prompt = ChatPromptTemplate.from_template("""
用户情绪：愤怒（分数 -0.8）
用户消息：{message}

响应策略：优先安抚，快速处理，必要时转人工

请生成符合策略的客服回复：
1. 先表达歉意和安抚
2. 承认问题严重性
3. 提供快速处理方案
4. 询问是否需要人工介入
""")

    response = (response_prompt | llm | StrOutputParser()).invoke({"message": angry_message})

    print("愤怒情绪处理示例：")
    print("-" * 60)
    print(f"用户消息: {angry_message}")
    print(f"客服回复:")
    print(response)
    print()


def demo_transfer_decision():
    """转人工决策"""
    print("=" * 60)
    print(f"Demo 29-2 (3/3): 转人工决策 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    print("转人工条件：")
    print("-" * 60)
    print("""
判断是否需要转人工的标准：

1. 情绪分数低于 -0.5（强烈负面情绪）
2. 用户明确要求人工服务
3. 问题超出 AI 处理能力
4. 涉及敏感信息或高风险操作
5. 多次尝试仍未解决问题
""")
    print()

    test_cases = [
        {"message": "我要退货", "score": -0.3, "intent": "return_refund"},
        {"message": "这简直是诈骗！我要报警！", "score": -0.9, "intent": "complaint"},
        {"message": "请给我转人工客服", "score": 0.0, "intent": "general_chat"},
        {"message": "我的账户被盗了，怎么办", "score": -0.6, "intent": "tech_support"},
    ]

    print("转人工判断示例：")
    print("-" * 60)

    for case in test_cases:
        should_transfer = case["score"] < -0.5 or "人工" in case["message"]
        print(f"消息: {case['message']}")
        print(f"情绪分数: {case['score']}")
        print(f"是否转人工: {should_transfer}")
        print()


if __name__ == "__main__":
    demo_sentiment_analysis()
    demo_response_strategy()
    demo_transfer_decision()

    print("=" * 60)
    print("Demo 29-2 完成!")
    print()
    print("情绪分析总结：")
    print("  - 4种情绪类型：positive、neutral、negative、angry")
    print("  - 情绪分数范围：-1（极度负面）到 1（极度正面）")
    print("  - 不同情绪采用不同响应策略")
    print()
    print("转人工标准：")
    print("  - 情绪分数 < -0.5")
    print("  - 用户明确要求")
    print("  - 问题超出 AI 能力")
    print("=" * 60)