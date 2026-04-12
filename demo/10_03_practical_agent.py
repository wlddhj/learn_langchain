"""
第10章 Demo 3：实战 —— 智能客服 Agent

综合运用 Agent + 多工具，构建电商智能客服。
可独立运行。
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
from langchain_core.tools import tool
from langchain.agents import create_agent

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 电商客服工具集
# ============================================================

@tool
def search_product(keyword: str) -> str:
    """搜索商品信息，包括名称、价格、库存、评分。
    当用户想查找商品、了解商品详情时使用。
    keyword: 商品名称或关键词"""
    products = {
        "MacBook": "MacBook Pro 14寸 M3 | 价格: ¥14999 | 库存: 23台 | 评分: 4.8/5 | 颜色: 深空灰/银色",
        "iPhone": "iPhone 15 Pro Max | 价格: ¥9999 | 库存: 56台 | 评分: 4.7/5 | 颜色: 原色/蓝色/白色",
        "AirPods": "AirPods Pro 2 | 价格: ¥1899 | 库存: 120个 | 评分: 4.6/5 | 颜色: 白色",
        "iPad": "iPad Air M2 | 价格: ¥4799 | 库存: 45台 | 评分: 4.5/5 | 颜色: 星光色/深空灰",
        "Watch": "Apple Watch Ultra 2 | 价格: ¥5999 | 库存: 18块 | 评分: 4.7/5 | 颜色: 钛金属",
    }
    for key, value in products.items():
        if keyword.lower() in key.lower():
            return value
    return f"未找到 '{keyword}' 相关商品，请尝试其他关键词。"


@tool
def query_order(order_id: str) -> str:
    """查询订单详情，包括商品、价格、配送状态。
    当用户询问订单状态、物流信息时使用。
    order_id: 订单编号，格式如 ORD-001"""
    orders = {
        "ORD-001": ("订单: ORD-001 | 状态: 已发货 | 商品: MacBook Pro 14寸 | "
                     "价格: ¥14999 | 快递: 顺丰 SF1234567890 | 预计明天送达"),
        "ORD-002": ("订单: ORD-002 | 状态: 仓库备货中 | 商品: iPhone 15 Pro Max | "
                     "价格: ¥9999 | 预计2天后发货"),
        "ORD-003": ("订单: ORD-003 | 状态: 已签收 | 商品: AirPods Pro 2 | "
                     "价格: ¥1899 | 签收时间: 2024-01-15"),
        "ORD-004": ("订单: ORD-004 | 状态: 待付款 | 商品: iPad Air M2 | "
                     "价格: ¥4799 | 请在24小时内完成支付"),
    }
    return orders.get(order_id, f"未找到订单 {order_id}，请检查订单号。")


@tool
def check_return_policy(product_type: str) -> str:
    """查询退货政策和保修信息。
    当用户询问退换货、保修、售后问题时使用。
    product_type: 商品类型，如 '电子产品'、'配件'"""
    policies = {
        "电子产品": ("电子产品退货政策: \n"
                     "  - 7天无理由退货（未激活）\n"
                     "  - 15天换货（质量问题）\n"
                     "  - 1年官方保修\n"
                     "  - 延保服务: +¥299/年"),
        "配件": ("配件退货政策: \n"
                 "  - 7天无理由退货（未拆封）\n"
                 "  - 15天换货（质量问题）\n"
                 "  - 6个月保修"),
    }
    for key, value in policies.items():
        if key in product_type or product_type in key:
            return value
    return "通用退货政策: 7天无理由退货，15天质量问题换货。"


@tool
def calculate_price(price: float, discount_percent: float = 0, quantity: int = 1) -> str:
    """计算商品最终价格。
    当用户询问折扣价、总价、多件价格时使用。
    price: 商品单价
    discount_percent: 折扣百分比，如 85 表示85折
    quantity: 购买数量"""
    if discount_percent > 0:
        unit_price = price * discount_percent / 100
    else:
        unit_price = price
    total = unit_price * quantity
    result = f"单价: ¥{price}"
    if discount_percent > 0:
        result += f" → 折后: ¥{unit_price:.2f} ({discount_percent}折)"
    if quantity > 1:
        result += f" × {quantity}件"
    result += f" = 总价: ¥{total:.2f}"
    return result


# ============================================================
# 创建客服 Agent
# ============================================================

def create_customer_service_agent():
    """创建智能客服 Agent"""
    return create_agent(
        model=llm,
        tools=[search_product, query_order, check_return_policy, calculate_price],
        system_prompt="""你是「智选商城」的智能客服助手。请遵循以下规则：

1. 热情友好，使用中文回答
2. 查找商品信息时使用 search_product
3. 查询订单状态时使用 query_order
4. 退换货问题使用 check_return_policy
5. 价格计算使用 calculate_price
6. 回答要准确、简洁，不超过200字
7. 如果需要多个工具，请依次调用
8. 无法解决的问题，建议联系人工客服（400-123-4567）""",
    )


# ============================================================
# Demo 1: 单一问题处理
# ============================================================

def demo_single_questions():
    print("=" * 50)
    print(f"智能客服 Agent (1/3): 单一问题 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_customer_service_agent()

    questions = [
        "你好，我想了解一下你们有什么手机卖？",
        "帮我查一下订单 ORD-001 到哪了？",
        "笔记本电脑的退货政策是什么？",
    ]

    for q in questions:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print(f"用户: {q}")
        print(f"客服: {result['messages'][-1].content[:200]}")
        print()


# ============================================================
# Demo 2: 复合问题处理
# ============================================================

def demo_complex_questions():
    print("=" * 50)
    print(f"智能客服 Agent (2/3): 复合问题 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_customer_service_agent()

    questions = [
        "我想买2台 MacBook，如果有85折的话一共多少钱？",
        "我的订单 ORD-002 什么时候发货？另外耳机有什么推荐？",
        "查一下 ORD-003 的订单，这个产品保修多久？",
    ]

    for q in questions:
        print(f"用户: {q}")
        print("-" * 40)

        # 用 stream 追踪执行过程
        final_answer = ""
        for event in agent.stream(
            {"messages": [("user", q)]},
            config={"recursion_limit": 10},
        ):
            for node_name, node_output in event.items():
                msg = node_output["messages"][-1]
                if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [调用] {tc['name']}({tc['args']})")
                elif node_name == "tools":
                    print(f"  [结果] {msg.content[:80]}")
                elif node_name == "agent" and msg.content:
                    final_answer = msg.content

        print(f"客服: {final_answer[:200]}")
        print()


# ============================================================
# Demo 3: 多轮对话
# ============================================================

def demo_conversation():
    print("=" * 50)
    print(f"智能客服 Agent (3/3): 多轮对话 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_customer_service_agent()

    messages = []
    conversations = [
        "你好，我想买一台平板电脑",
        "iPad 多少钱？",
        "如果买2台的话，打9折是多少钱？",
        "好的，退货政策是什么样的？",
        "谢谢，我再考虑一下",
    ]

    for user_msg in conversations:
        messages.append(("user", user_msg))
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 10},
        )
        messages = result["messages"]
        ai_response = messages[-1].content

        print(f"用户: {user_msg}")
        print(f"客服: {ai_response[:150]}")
        print()

    print("Agent 在多轮对话中能记住之前的上下文，提供连贯的服务体验")


if __name__ == "__main__":
    demo_single_questions()
    demo_complex_questions()
    demo_conversation()

    print("=" * 50)
    print("Demo 10-3 完成!")
    print()
    print("智能客服 Agent 要点:")
    print("  1. 工具设计: 每个工具单一职责，描述清晰")
    print("  2. System Prompt: 定义角色、行为规则、回答风格")
    print("  3. 多轮对话: 追加 messages 维持上下文")
    print("  4. 复合问题: Agent 自动拆分并依次调用工具")
    print("  5. stream(): 实时追踪工具调用过程")
    print("=" * 50)
