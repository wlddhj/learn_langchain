"""
第13章 Demo 2：条件路由与工具调用 Graph

演示 conditional_edges、ToolNode、循环执行。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 工具定义
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询城市天气信息"""
    weather = {
        "北京": "晴天，25°C，湿度45%，适合户外活动",
        "上海": "多云，22°C，湿度65%，傍晚可能有小雨",
        "深圳": "小雨，28°C，湿度80%，建议带伞",
        "成都": "阴天，20°C，湿度70%",
    }
    return weather.get(city, f"未找到 {city} 的天气信息")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "不安全的表达式"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


tools = [get_weather, calculate]
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 1. 带工具调用的 Graph
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: ChatState):
    """LLM 决策节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: ChatState) -> Literal["tools", "__end__"]:
    """判断是否需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def demo_tool_graph():
    print("=" * 50)
    print(f"Demo 13-2 (1/3): 带工具调用的 Graph [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue)
    builder.add_edge("tools", "chatbot")  # 工具执行后回到 chatbot

    graph = builder.compile()

    # 需要工具的问题
    result = graph.invoke({"messages": [("user", "北京天气怎么样？")]})
    print("问题: 北京天气怎么样？")
    print(f"回答: {result['messages'][-1].content[:150]}")
    print()

    # 需要多步推理
    result = graph.invoke({"messages": [("user", "北京和上海哪个温度高？温差多少？")]})
    print("问题: 北京和上海哪个温度高？温差多少？")
    print(f"回答: {result['messages'][-1].content[:200]}")
    print()

    # 不需要工具
    result = graph.invoke({"messages": [("user", "你好，介绍一下你自己")]})
    print("问题: 你好，介绍一下你自己")
    print(f"回答: {result['messages'][-1].content[:100]}")
    print()


# ============================================================
# 2. 条件路由到不同专家
# ============================================================

def route_by_topic(state: ChatState) -> Literal["math_expert", "weather_expert", "chat"]:
    """根据问题内容路由到不同专家"""
    last_msg = state["messages"][-1]
    content = last_msg.content.lower() if isinstance(last_msg, HumanMessage) else ""

    if any(w in content for w in ["计算", "数学", "等于", "加", "减", "乘", "除"]):
        return "math_expert"
    elif any(w in content for w in ["天气", "温度", "下雨"]):
        return "weather_expert"
    else:
        return "chat"


def math_expert(state: ChatState):
    """数学专家节点"""
    response = llm.invoke([
        ("system", "你是数学专家。直接回答数学问题，必要时使用计算工具。简洁回答。"),
        *state["messages"],
    ])
    return {"messages": [response]}


def weather_expert(state: ChatState):
    """天气专家节点"""
    response = llm_with_tools.invoke([
        ("system", "你是天气专家。查询天气并给出专业建议。"),
        *state["messages"],
    ])
    return {"messages": [response]}


def chat_node(state: ChatState):
    """通用聊天节点"""
    response = llm.invoke([
        ("system", "你是友好的助手。简洁回答。"),
        *state["messages"],
    ])
    return {"messages": [response]}


def expert_should_continue(state: ChatState) -> Literal["tools", "__end__"]:
    """天气专家可能需要调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def demo_routing():
    print("=" * 50)
    print(f"Demo 13-2 (2/3): 条件路由 [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(ChatState)
    builder.add_node("math_expert", math_expert)
    builder.add_node("weather_expert", weather_expert)
    builder.add_node("chat", chat_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_conditional_edges(START, route_by_topic)
    builder.add_edge("math_expert", END)
    builder.add_conditional_edges("weather_expert", expert_should_continue)
    builder.add_edge("chat", END)
    builder.add_edge("tools", "weather_expert")

    graph = builder.compile()

    questions = [
        ("25 * 36 等于多少？", "数学"),
        ("深圳今天天气怎么样？", "天气"),
        ("你好，推荐一本好书", "通用"),
    ]

    for q, expected in questions:
        result = graph.invoke({"messages": [("user", q)]})
        answer = result["messages"][-1].content
        print(f"问题: {q}  (预期路由: {expected})")
        print(f"回答: {answer[:120]}")
        print()


# ============================================================
# 3. 执行过程追踪
# ============================================================

def demo_stream_trace():
    print("=" * 50)
    print(f"Demo 13-2 (3/3): stream 执行追踪 [{QWEN_MODEL}]")
    print("=" * 50)

    builder = StateGraph(ChatState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue)
    builder.add_edge("tools", "chatbot")
    graph = builder.compile()

    question = "北京天气如何？温度乘以2是多少？"
    print(f"问题: {question}")
    print("-" * 40)

    for event in graph.stream({"messages": [("user", question)]}):
        for node_name, node_output in event.items():
            if node_name == "chatbot":
                msg = node_output["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  [chatbot] 调用工具: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"  [chatbot] 回答: {msg.content[:120]}")
            elif node_name == "tools":
                msg = node_output["messages"][-1]
                print(f"  [tools]   结果: {msg.content[:80]}")

    print()
    print("流程: START → chatbot → tools → chatbot → tools → chatbot → END")
    print()


if __name__ == "__main__":
    demo_tool_graph()
    demo_routing()
    demo_stream_trace()

    print("=" * 50)
    print("Demo 13-2 完成!")
    print()
    print("条件路由要点:")
    print("  add_conditional_edges - 条件边，根据返回值路由")
    print("  ToolNode              - 内置工具执行节点")
    print("  should_continue       - 判断是否继续调用工具")
    print("  工具循环              - tools → chatbot → tools → ...")
    print("  stream()              - 逐步追踪图执行过程")
    print("=" * 50)
