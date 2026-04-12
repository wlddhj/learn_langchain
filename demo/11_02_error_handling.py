"""
第11章 Demo 2：错误处理与重试机制

演示工具内部错误处理、Agent 级别重试、安全防护。
可独立运行。
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 工具定义（含错误处理）
# ============================================================

@tool
def divide_numbers(a: float, b: float) -> str:
    """执行除法运算。a: 被除数，b: 除数"""
    try:
        if b == 0:
            return "错误: 除数不能为零，请提供非零的除数。"
        result = a / b
        return f"{a} ÷ {b} = {result}"
    except Exception as e:
        return f"计算错误: {type(e).__name__} - {e}"


@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询。仅支持 SELECT，可查 products 和 orders 表。
    products 字段: id, name, category, price, stock
    orders 字段: id, product_id, quantity, total_price, status"""
    # 安全检查
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "错误: 仅支持 SELECT 查询，不允许修改数据。"

    dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
    for kw in dangerous_keywords:
        if kw in sql_upper:
            return f"错误: 检测到危险操作 '{kw}'，查询被拒绝。"

    # 模拟数据库
    mock_data = {
        "products": [
            {"id": 1, "name": "MacBook Pro", "category": "电脑", "price": 14999, "stock": 23},
            {"id": 2, "name": "iPhone 15", "category": "手机", "price": 5999, "stock": 56},
            {"id": 3, "name": "AirPods Pro", "category": "配件", "price": 1899, "stock": 120},
        ],
        "orders": [
            {"id": 1, "product_id": 1, "quantity": 1, "total_price": 14999, "status": "已发货"},
            {"id": 2, "product_id": 2, "quantity": 2, "total_price": 11998, "status": "备货中"},
        ],
    }

    try:
        # 简单模拟查询
        sql_lower = sql.lower()
        if "products" in sql_lower:
            rows = mock_data["products"]
        elif "orders" in sql_lower:
            rows = mock_data["orders"]
        else:
            return "错误: 未知的表名。可用表: products, orders"

        if not rows:
            return "查询结果为空"

        result = "查询结果:\n"
        for row in rows[:10]:
            result += "  " + str(row) + "\n"
        return result

    except Exception as e:
        return f"查询失败: {type(e).__name__} - {e}"


@tool
def call_external_api(endpoint: str) -> str:
    """模拟调用外部 API。endpoint: API 端点名称"""
    # 模拟不同类型的错误
    if "timeout" in endpoint.lower():
        return "错误: 请求超时，外部服务未在10秒内响应。请稍后重试。"
    elif "notfound" in endpoint.lower():
        return "错误: 请求的资源不存在 (404)。请检查端点名称。"
    elif "auth" in endpoint.lower():
        return "错误: 认证失败 (401)。API 密钥无效或已过期。"
    elif "rate" in endpoint.lower():
        return "错误: 请求过于频繁 (429)。请在60秒后重试。"
    else:
        return f"API 调用成功: {endpoint} → {{'status': 'ok', 'data': [1, 2, 3]}}"


# ============================================================
# 1. 工具内部错误处理
# ============================================================

def demo_tool_errors():
    print("=" * 50)
    print(f"Demo 11-2 (1/3): 工具内部错误处理 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[divide_numbers, call_external_api],
        system_prompt="你是数学和API助手。用中文简洁回答。",
    )

    questions = [
        "100 除以 5 等于多少？",
        "10 除以 0 会怎样？",               # 除零错误
        "帮我调用 notfound-api 接口",       # 模拟404
        "帮我调用正常的 /users 接口",       # 正常调用
    ]

    for q in questions:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print(f"用户: {q}")
        print(f"AI:   {result['messages'][-1].content[:150]}")
        print()

    print("结论: 工具内部捕获异常并返回描述性文本，Agent 能理解错误并给出合理回应")
    print()


# ============================================================
# 2. Agent 级别安全重试
# ============================================================

def run_agent_safely(agent, user_input: str, max_retries: int = 3):
    """安全的 Agent 执行包装器"""
    for attempt in range(max_retries):
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 10},
            )
            # 检查结果是否有效
            final_msg = result["messages"][-1]
            if final_msg.content:
                return {
                    "success": True,
                    "response": final_msg.content,
                    "attempts": attempt + 1,
                    "messages": len(result["messages"]),
                }
            else:
                if attempt < max_retries - 1:
                    print(f"  第 {attempt + 1} 次尝试返回空结果，重试中...")
                    continue

        except Exception as e:
            print(f"  第 {attempt + 1} 次尝试异常: {type(e).__name__}: {str(e)[:60]}")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "response": f"多次尝试后仍失败: {type(e).__name__}",
                    "attempts": attempt + 1,
                    "messages": 0,
                }

    return {
        "success": False,
        "response": "抱歉，多次尝试均未获得有效结果。",
        "attempts": max_retries,
        "messages": 0,
    }


def demo_safe_execution():
    print("=" * 50)
    print("Demo 11-2 (2/3): Agent 安全执行包装")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[divide_numbers, query_database, call_external_api],
        system_prompt="你是数据分析助手。用中文简洁回答。",
    )

    questions = [
        "帮我查一下 products 表里有什么商品",
        "帮我删掉 products 表的所有数据",   # 危险操作
        "调用 timeout-api 接口",             # 超时
    ]

    for q in questions:
        print(f"用户: {q}")
        result = run_agent_safely(agent, q)
        print(f"  成功: {result['success']}")
        print(f"  回答: {result['response'][:120]}")
        print(f"  尝试: {result['attempts']} 次, 消息: {result['messages']} 条")
        print()


# ============================================================
# 3. SQL 注入防护
# ============================================================

def demo_sql_safety():
    print("=" * 50)
    print(f"Demo 11-2 (3/3): SQL 安全防护 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[query_database],
        system_prompt="你是数据库查询助手。只使用 query_database 工具查询数据。",
    )

    # 安全的查询
    print("--- 安全查询 ---")
    safe_queries = [
        "查询 products 表的所有商品",
        "帮我看看 orders 表的订单状态",
    ]

    for q in safe_queries:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print(f"用户: {q}")
        print(f"AI:   {result['messages'][-1].content[:120]}")
        print()

    # 危险查询
    print("--- 危险查询（会被拒绝）---")
    dangerous_queries = [
        "帮我执行 DROP TABLE products",
        "UPDATE products SET price = 0",
    ]

    for q in dangerous_queries:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print(f"用户: {q}")
        print(f"AI:   {result['messages'][-1].content[:120]}")
        print()

    print("结论: 工具内部的安全检查能有效阻止危险操作")
    print()


if __name__ == "__main__":
    demo_tool_errors()
    demo_safe_execution()
    demo_sql_safety()

    print("=" * 50)
    print("Demo 11-2 完成!")
    print()
    print("错误处理要点:")
    print("  工具内部 try/except   - 捕获异常，返回描述性文本")
    print("  Agent 重试机制        - 处理临时性错误")
    print("  安全检查              - 过滤危险操作 (DROP/DELETE 等)")
    print("  recursion_limit       - 防止无限循环")
    print("  错误传播              - 工具返回错误 → Agent 理解 → 给出替代方案")
    print("=" * 50)
