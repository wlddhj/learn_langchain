"""
第29章 Demo 1：意图识别器

演示客服系统中的意图分类功能。
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


def demo_intent_classification():
    """意图分类演示"""
    print("=" * 60)
    print(f"Demo 29-1 (1/3): 意图分类 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 意图类型定义
    intent_types = """
可选意图类型：
- product_query: 产品咨询（询问产品信息、功能、价格等）
- order_status: 订单查询（查询订单状态、物流信息等）
- complaint: 投诉建议（投诉服务问题、提出建议等）
- tech_support: 技术支持（技术问题、使用问题等）
- return_refund: 退换货（退货、换货、退款请求等）
- general_chat: 一般闲聊（问候、闲聊等）
- unknown: 未识别
"""

    classify_prompt = ChatPromptTemplate.from_template("""
分析用户消息的意图，选择最合适的分类。

用户消息：{message}

{intent_types}

请输出：
1. 意图类型（从上面选择一个）
2. 置信度（0-1）

格式：意图: xxx, 置信度: xxx
""")

    test_messages = [
        "我想查询我的订单状态",
        "这个产品多少钱？",
        "我要退货",
        "怎么使用这个功能？",
        "你好",
        "服务太差了，我要投诉",
    ]

    print("意图分类测试：")
    print("-" * 60)

    for msg in test_messages:
        result = (classify_prompt | llm | StrOutputParser()).invoke({
            "message": msg,
            "intent_types": intent_types,
        })

        print(f"用户消息: {msg}")
        print(f"分类结果: {result}")
        print()


def demo_intent_handler_routing():
    """意图路由到处理器"""
    print("=" * 60)
    print(f"Demo 29-1 (2/3): 意图路由 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 处理器映射
    handler_mapping = {
        "product_query": "product_agent",
        "order_status": "order_agent",
        "complaint": "ticket_agent",
        "tech_support": "support_agent",
        "return_refund": "ticket_agent",
        "general_chat": "chat_agent",
        "unknown": "default_agent",
    }

    print("意图 → 处理器映射：")
    print("-" * 60)
    for intent, handler in handler_mapping.items():
        print(f"  {intent:<20} → {handler}")
    print()

    print("路由逻辑示例：")
    print("-" * 60)
    print("""
if intent == "order_status":
    # 调用订单查询 Agent
    result = order_agent.query(message)

elif intent == "product_query":
    # 先查知识库，再调用产品 Agent
    result = knowledge_retriever.query(message)
    if not result:
        result = product_agent.consult(message)

elif intent in ["complaint", "return_refund"]:
    # 创建工单
    result = ticket_agent.handle_issue(message)
""")


def demo_sub_intent_extraction():
    """子意图提取"""
    print("=" * 60)
    print(f"Demo 29-1 (3/3): 子意图提取 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    message = "我买的手机屏幕坏了，想换一个新的，或者退款也可以"

    sub_intent_prompt = ChatPromptTemplate.from_template("""
分析用户消息的子意图：

用户消息：{message}

请提取：
1. 主意图
2. 所有子意图（具体需求）
3. 关键信息（如产品类型、问题描述等）

输出格式：
主意图: xxx
子意图: [xxx, xxx, xxx]
关键信息: xxx
""")

    result = (sub_intent_prompt | llm | StrOutputParser()).invoke({"message": message})

    print("用户消息:", message)
    print()
    print("意图分析:")
    print(result)
    print()

    print("应用场景:")
    print("  - 复杂问题可能包含多个子意图")
    print("  - 提取关键信息帮助后续处理")
    print("  - 为工单系统提供详细信息")


if __name__ == "__main__":
    demo_intent_classification()
    demo_intent_handler_routing()
    demo_sub_intent_extraction()

    print("=" * 60)
    print("Demo 29-1 完成!")
    print()
    print("意图识别总结：")
    print("  - 6种主要意图：产品咨询、订单查询、投诉、技术支持、退换货、闲聊")
    print("  - 每种意图对应不同处理器/Agent")
    print("  - 复杂消息需要提取子意图和关键信息")
    print()
    print("实现要点：")
    print("  - 使用低温度保证分类准确性")
    print("  - 建立清晰的意图-处理器映射")
    print("  - 支持未知意图的兜底处理")
    print("=" * 60)