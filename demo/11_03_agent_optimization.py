"""
第11章 Demo 3：Agent 性能优化与可观测性

演示 Token 用量分析、工具描述优化、执行效率对比。
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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 工具定义 - 好描述 vs 差描述
# ============================================================

# 差描述的工具
@tool
def search_v1(query: str) -> str:
    """搜索"""
    data = {"北京天气": "晴天 25°C", "上海天气": "多云 22°C", "深圳天气": "小雨 28°C"}
    for k, v in data.items():
        if query in k or k in query:
            return v
    return f"未找到 '{query}'"


# 好描述的工具
@tool
def search_v2(query: str) -> str:
    """搜索天气信息数据库，返回指定城市的实时天气数据。
    当用户询问某个城市的天气状况、温度、是否下雨等问题时使用此工具。
    query: 城市名称 + 天气，如 '北京天气'、'上海天气'"""
    data = {"北京天气": "晴天 25°C，湿度 45%，适合户外活动", "上海天气": "多云 22°C，湿度 65%", "深圳天气": "小雨 28°C，湿度 80%，建议带伞"}
    for k, v in data.items():
        if query in k or k in query:
            return v
    return f"未找到 '{query}' 的天气信息，支持的城市: 北京、上海、深圳"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"不安全的表达式"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 分析函数
# ============================================================

def analyze_agent_run(result, elapsed_time):
    """分析 Agent 运行的资源消耗"""
    messages = result["messages"]
    tool_calls = 0
    total_content = 0

    for msg in messages:
        total_content += len(msg.content) if msg.content else 0
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            tool_calls += len(msg.tool_calls)

    return {
        "total_messages": len(messages),
        "tool_calls": tool_calls,
        "total_chars": total_content,
        "elapsed": round(elapsed_time, 2),
    }


# ============================================================
# 1. 工具描述质量对比
# ============================================================

def demo_description_quality():
    print("=" * 50)
    print(f"Demo 11-3 (1/3): 工具描述质量对比 [{QWEN_MODEL}]")
    print("=" * 50)

    # 差描述 Agent
    agent_bad = create_agent(
        model=llm,
        tools=[search_v1, calculator],
        system_prompt="你是助手，用中文回答。",
    )

    # 好描述 Agent
    agent_good = create_agent(
        model=llm,
        tools=[search_v2, calculator],
        system_prompt="你是助手，用中文回答。",
    )

    questions = [
        "北京天气怎么样？",
        "帮我算 123 * 456",
        "你好",  # 不需要工具
    ]

    for q in questions:
        print(f"问题: {q}")

        # 差描述
        start = time.time()
        result_bad = agent_bad.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        elapsed_bad = time.time() - start
        stats_bad = analyze_agent_run(result_bad, elapsed_bad)

        # 好描述
        start = time.time()
        result_good = agent_good.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        elapsed_good = time.time() - start
        stats_good = analyze_agent_run(result_good, elapsed_good)

        print(f"  差描述: {stats_bad['tool_calls']} 次工具调用, "
              f"{stats_bad['total_messages']} 条消息, {stats_bad['elapsed']}s")
        print(f"  好描述: {stats_good['tool_calls']} 次工具调用, "
              f"{stats_good['total_messages']} 条消息, {stats_good['elapsed']}s")
        print()


# ============================================================
# 2. Token 用量估算
# ============================================================

def demo_token_analysis():
    print("=" * 50)
    print("Demo 11-3 (2/3): Token 用量估算")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_v2, calculator],
        system_prompt="你是助手，用中文简洁回答。",
    )

    questions = [
        ("简单问题", "你好"),
        ("单工具", "北京天气怎么样？"),
        ("多工具", "北京天气如何？25+35等于多少？"),
        ("复杂问题", "比较北京和上海的天气，然后计算温差"),
    ]

    print(f"{'问题类型':<10} {'消息数':>6} {'工具调用':>8} {'总字符':>8} {'估算Token':>10} {'耗时':>6}")
    print("-" * 60)

    for label, q in questions:
        start = time.time()
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 10},
        )
        elapsed = time.time() - start
        stats = analyze_agent_run(result, elapsed)

        # 粗略估算: 中文约 1.5 字符/token
        est_tokens = int(stats["total_chars"] / 1.5)

        print(f"{label:<10} {stats['total_messages']:>6} {stats['tool_calls']:>8} "
              f"{stats['total_chars']:>8} {est_tokens:>10} {stats['elapsed']:>5.2f}s")

    print()
    print("注意: Token 估算为粗略值（中文约1.5字符/token），实际以 API 返回为准")
    print("建议: 使用 tiktoken 库进行精确计算")
    print()


# ============================================================
# 3. 执行效率优化对比
# ============================================================

def demo_efficiency():
    print("=" * 50)
    print(f"Demo 11-3 (3/3): 执行效率优化 [{QWEN_MODEL}]")
    print("=" * 50)

    # 信息丰富的工具（返回更多上下文，减少二次查询）
    @tool
    def get_weather_rich(city: str) -> str:
        """查询天气信息，返回完整的天气数据。
        当用户询问天气时使用。一次返回所有相关信息，无需重复查询。"""
        data = {
            "北京": "城市: 北京\n天气: 晴天\n温度: 25°C\n湿度: 45%\n风力: 3级\n空气质量: 良\n建议: 适合户外活动",
            "上海": "城市: 上海\n天气: 多云\n温度: 22°C\n湿度: 65%\n风力: 2级\n空气质量: 良\n建议: 适合出行",
        }
        return data.get(city, f"未找到 {city}")

    # 信息简单的工具（可能需要二次查询）
    @tool
    def get_weather_simple(city: str) -> str:
        """查询天气"""
        data = {"北京": "晴天 25°C", "上海": "多云 22°C"}
        return data.get(city, f"未找到 {city}")

    question = "北京天气怎么样？温度是多少？适合出行吗？"

    # 信息丰富的工具
    agent_rich = create_agent(
        model=llm,
        tools=[get_weather_rich],
        system_prompt="你是天气助手，用中文简洁回答。",
    )

    start = time.time()
    result_rich = agent_rich.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 8},
    )
    elapsed_rich = time.time() - start

    # 信息简单的工具
    agent_simple = create_agent(
        model=llm,
        tools=[get_weather_simple],
        system_prompt="你是天气助手，用中文简洁回答。",
    )

    start = time.time()
    result_simple = agent_simple.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 8},
    )
    elapsed_simple = time.time() - start

    print(f"问题: {question}")
    print()
    print(f"信息丰富工具: {len(result_rich['messages'])} 条消息, {elapsed_rich:.2f}s")
    print(f"信息简单工具: {len(result_simple['messages'])} 条消息, {elapsed_simple:.2f}s")
    print()
    print("结论: 工具返回信息越丰富，Agent 越少需要重复查询，效率越高")
    print()


if __name__ == "__main__":
    demo_description_quality()
    demo_token_analysis()
    demo_efficiency()

    print("=" * 50)
    print("Demo 11-3 完成!")
    print()
    print("Agent 优化要点:")
    print("  1. 工具描述要详细   - 帮助 LLM 准确选择工具")
    print("  2. 工具返回要丰富   - 减少二次查询，提高效率")
    print("  3. 监控 Token 用量  - 控制成本")
    print("  4. 记录执行时间     - 发现性能瓶颈")
    print("  5. 对比不同方案     - 选择最优配置")
    print("=" * 50)
