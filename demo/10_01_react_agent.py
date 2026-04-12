"""
第10章 Demo 1：ReAct Agent 基础

演示 create_react_agent 创建 Agent、工具调用流程、执行过程查看。
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
from langchain.agents import create_agent

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 工具定义
# ============================================================

@tool
def search_weather(city: str) -> str:
    """查询城市天气信息。
    当用户询问某个城市的天气、温度、是否下雨时使用。
    city: 城市名称，如 '北京'、'上海'"""
    weather_data = {
        "北京": "晴天，气温 25°C，湿度 45%，空气质量 良，适合户外活动",
        "上海": "多云转阴，气温 22°C，湿度 65%，傍晚可能有小雨",
        "深圳": "小雨，气温 28°C，湿度 80%，建议带伞",
        "成都": "阴天，气温 20°C，湿度 70%，空气质量 良",
        "杭州": "晴天，气温 24°C，湿度 50%，适合出行",
        "广州": "雷阵雨，气温 30°C，湿度 85%，注意防雷",
    }
    return weather_data.get(city, f"未找到 {city} 的天气信息")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持加减乘除和括号。
    当用户需要做数学运算、价格计算时使用。
    expression: 数学表达式，如 '25 * 4' 或 '(100 + 50) / 3'"""
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
# 1. 基础 Agent 创建与运行
# ============================================================

def demo_basic_agent():
    print("=" * 50)
    print(f"Demo 10-1 (1/3): 基础 ReAct Agent [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_weather, calculator],
    )
    result = agent.invoke({
        "messages": [("user", "北京天气怎么样？")]
    })

    print("问题: 北京天气怎么样？")
    print(f"回答: {result['messages'][-1].content[:150]}")
    print()

    # 需要多步推理的问题
    result = agent.invoke({
        "messages": [("user", "北京和上海哪个城市温度更高？温差是多少？")]
    })

    print("问题: 北京和上海哪个城市温度更高？温差是多少？")
    print(f"回答: {result['messages'][-1].content[:200]}")
    print()


# ============================================================
# 2. 查看完整执行过程
# ============================================================

def demo_execution_trace():
    print("=" * 50)
    print("Demo 10-1 (2/3): 执行过程追踪")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_weather, calculator],
    )

    result = agent.invoke({
        "messages": [("user", "深圳天气如何？如果温度是28度，那28*3是多少？")]
    })

    # 打印完整执行过程
    print("问题: 深圳天气如何？如果温度是28度，那28*3是多少？")
    print("-" * 40)

    step = 0
    for msg in result["messages"]:
        msg_type = msg.__class__.__name__

        if isinstance(msg, HumanMessage):
            print(f"[用户] {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step += 1
                    print(f"[Step {step}] Agent 决定调用: {tc['name']}({tc['args']})")
            elif msg.content:
                print(f"[Agent 最终回答] {msg.content[:200]}")

        elif isinstance(msg, ToolMessage):
            print(f"  ↳ 工具结果: {msg.content[:100]}")

    print()
    print(f"总步数: {step}")
    print(f"总消息数: {len(result['messages'])}")
    print()


# ============================================================
# 3. 流式输出
# ============================================================

def demo_stream():
    print("=" * 50)
    print(f"Demo 10-1 (3/3): 流式输出 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_weather, calculator],
    )

    print("问题: 成都和杭州天气怎么样？两个城市的温差是多少？")
    print("-" * 40)

    for event in agent.stream({
        "messages": [("user", "成都和杭州天气怎么样？两个城市的温差是多少？")]
    }):
        for node_name, node_output in event.items():
            if node_name == "agent":
                msg = node_output["messages"][-1]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [调用] {tc['name']}({tc['args']})")
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"  [思考] {msg.content[:100]}")
            elif node_name == "tools":
                msg = node_output["messages"][-1]
                print(f"  [结果] {msg.content[:100]}")

    print()
    print("流式输出让我们能实时看到 Agent 的每一步执行过程")
    print()


if __name__ == "__main__":
    demo_basic_agent()
    demo_execution_trace()
    demo_stream()

    print("=" * 50)
    print("Demo 10-1 完成!")
    print()
    print("ReAct Agent 要点:")
    print("  create_react_agent  - 用 LLM + 工具创建 Agent")
    print("  Thought → Action → Observation - ReAct 循环")
    print("  messages 列表       - 完整记录执行过程")
    print("  stream()            - 流式查看执行步骤")
    print("  recursion_limit     - 限制最大步数（防止无限循环）")
    print("=" * 50)
