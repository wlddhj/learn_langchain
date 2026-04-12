"""
第9章 Demo 3：实战 —— 多工具协同的智能助手

演示多工具绑定、手动工具调用循环、工具结果注入。
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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 工具定义
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """搜索公司内部知识库，返回与关键词匹配的文档片段。
    当用户询问公司政策、产品说明、技术文档等内部信息时使用此工具。
    query: 搜索关键词，尽量使用具体词汇"""
    kb = {
        "年假": "公司年假政策：入职满1年享有5天年假，满3年10天，满5年15天。年假可跨年累积，最多累积2年。",
        "报销": "报销流程：1. 填写报销单 2. 附上发票 3. 主管审批 4. 财务审核 5. 3个工作日内打款。",
        "考勤": "考勤制度：上班时间9:00-18:00，弹性30分钟。每月允许2次迟到（不超过30分钟），超过扣薪。",
        "远程办公": "远程办公政策：每周可申请2天远程办公，需提前1天在系统中申请，主管审批后生效。",
        "培训": "培训制度：公司每季度组织技术培训，员工可申请外部培训费用报销（上限5000元/年）。",
    }
    results = []
    for key, value in kb.items():
        if key in query or any(c in query for c in key):
            results.append(f"[{key}] {value}")
    if results:
        return "\n".join(results)
    return f"未找到与 '{query}' 相关的信息，请尝试其他关键词。"


@tool
def query_order(order_id: str) -> str:
    """查询订单详情，包括商品、价格、配送状态。
    当用户询问订单状态、物流信息时使用此工具。
    order_id: 订单编号，格式如 ORD-001"""
    orders = {
        "ORD-001": "订单状态: 已发货 | 商品: MacBook Pro 14寸 | 价格: ¥14999 | 配送: 顺丰 SF1234567890，预计明天送达",
        "ORD-002": "订单状态: 仓库备货中 | 商品: iPhone 15 Pro | 价格: ¥8999 | 预计2天后发货",
        "ORD-003": "订单状态: 已签收 | 商品: AirPods Pro | 价格: ¥1899 | 签收时间: 2024-01-15",
        "ORD-004": "订单状态: 待付款 | 商品: iPad Air | 价格: ¥4799 | 请在24小时内完成支付",
    }
    return orders.get(order_id, f"未找到订单 {order_id}，请检查订单号是否正确。")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和括号。
    当用户需要进行数学计算、价格计算、折扣计算时使用此工具。
    expression: 数学表达式，如 '14999 * 0.85' 或 '8999 + 1899'"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"表达式包含不允许的字符: {expression}"
        result = eval(expression)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "错误: 除数不能为零"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 手动工具调用循环
# ============================================================

def run_tool_loop(question: str, max_steps: int = 5) -> str:
    """手动模拟 Agent 的工具调用循环"""
    tools_map = {
        "search_knowledge_base": search_knowledge_base,
        "query_order": query_order,
        "calculate": calculate,
    }

    llm_with_tools = llm.bind_tools(list(tools_map.values()))
    messages = [HumanMessage(content=question)]

    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)

        # 没有 tool_calls，说明 LLM 认为可以直接回答
        if not response.tool_calls:
            return response.content

        # 有 tool_calls，执行工具
        messages.append(response)

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"  Step {step + 1}: 调用 {tool_name}({tool_args})")

            tool_result = tools_map[tool_name].invoke(tool_args)
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tc["id"],
            ))

    return "达到最大步数限制，请简化问题后重试。"


# ============================================================
# Demo 1: 单工具调用
# ============================================================

def demo_single_tool():
    print("=" * 50)
    print(f"Demo 9-3 (1/3): 单工具调用 [{QWEN_MODEL}]")
    print("=" * 50)

    questions = [
        "公司的年假政策是什么？",
        "帮我查一下订单 ORD-002 的状态",
        "你好，你是谁？",
    ]

    for q in questions:
        print(f"用户: {q}")
        answer = run_tool_loop(q)
        print(f"AI:   {answer[:150]}")
        print()


# ============================================================
# Demo 2: 多工具协同
# ============================================================

def demo_multi_tool():
    print("=" * 50)
    print(f"Demo 9-3 (2/3): 多工具协同调用 [{QWEN_MODEL}]")
    print("=" * 50)

    questions = [
        "查一下订单 ORD-001 的状态，然后帮我算一下如果打85折是多少钱",
        "公司培训政策是什么？培训费用报销上限是多少？乘以2是多少？",
        "ORD-003 的商品价格是多少？和 ORD-004 的价格加起来一共多少？",
    ]

    for q in questions:
        print(f"用户: {q}")
        answer = run_tool_loop(q)
        print(f"AI:   {answer[:200]}")
        print()


# ============================================================
# Demo 3: 完整工具调用追踪
# ============================================================

def demo_full_trace():
    print("=" * 50)
    print(f"Demo 9-3 (3/3): 完整工具调用追踪 [{QWEN_MODEL}]")
    print("=" * 50)

    tools_map = {
        "search_knowledge_base": search_knowledge_base,
        "query_order": query_order,
        "calculate": calculate,
    }

    llm_with_tools = llm.bind_tools(list(tools_map.values()))

    question = "我想了解报销流程，然后帮我算一下 1899 + 4799，看看这两个订单加起来多少钱"
    messages = [HumanMessage(content=question)]
    print(f"用户: {question}")
    print()

    for step in range(6):
        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            print(f"最终回答: {response.content[:200]}")
            break

        messages.append(response)

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_result = tools_map[tool_name].invoke(tool_args)

            print(f"  [Step {step + 1}]")
            print(f"    工具: {tool_name}")
            print(f"    参数: {tool_args}")
            print(f"    结果: {str(tool_result)[:100]}")
            print()

            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tc["id"],
            ))

    print(f"总消息数: {len(messages)}")
    print()


if __name__ == "__main__":
    demo_single_tool()
    demo_multi_tool()
    demo_full_trace()

    print("=" * 50)
    print("Demo 9-3 完成!")
    print()
    print("多工具协同要点:")
    print("  1. bind_tools()    - 将多个工具绑定到 LLM")
    print("  2. tool_calls      - LLM 返回的工具调用列表")
    print("  3. ToolMessage     - 将工具结果反馈给 LLM")
    print("  4. 循环执行        - 直到 LLM 不再调用工具为止")
    print("  5. max_steps       - 防止无限循环")
    print("=" * 50)
