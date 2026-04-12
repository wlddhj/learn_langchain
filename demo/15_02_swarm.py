"""
第15章 Demo 2：Swarm 交接模式

演示 Agent 间直接交接、无需中心调度器。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# State
# ============================================================

class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str


# ============================================================
# Agent 交接路由
# ============================================================

def route_entry(state: SwarmState):
    """入口路由：根据问题类型分配给合适的 Agent"""
    last_msg = state["messages"][-1]
    content = last_msg.content.lower() if hasattr(last_msg, "content") else ""

    if any(w in content for w in ["订单", "物流", "发货", "签收"]):
        return "order_agent"
    elif any(w in content for w in ["技术", "故障", "无法", "错误", "bug"]):
        return "tech_agent"
    else:
        return "general_agent"


def route_from_agent(state: SwarmState):
    """从 Agent 路由：根据 Agent 的交接意图决定"""
    agent = state.get("current_agent", "")
    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage):
        content = last_msg.content.lower()
        if "转交" in content or "交接" in content:
            if "订单" in content or "order" in content:
                return "order_agent"
            elif "技术" in content or "tech" in content:
                return "tech_agent"
            elif "通用" in content or "general" in content:
                return "general_agent"

    return END


# ============================================================
# Agent 节点
# ============================================================

def order_agent(state: SwarmState):
    """订单客服 Agent"""
    response = llm.invoke([
        SystemMessage(content="""你是订单客服「小单」。处理订单查询、物流追踪、退换货。
用中文简洁回答（不超过100字）。
如果问题不属于订单范围，回复: '这个问题我转交给通用客服处理。[转交通用]'。"""),
        *state["messages"][-4:],
    ])
    return {"messages": [response], "current_agent": "order_agent"}


def tech_agent(state: SwarmState):
    """技术支持 Agent"""
    response = llm.invoke([
        SystemMessage(content="""你是技术支持「小技」。处理产品故障、技术问题、使用指导。
用中文简洁回答（不超过100字）。
如果问题不属于技术范围，回复: '这个问题我转交给通用客服处理。[转交通用]'。"""),
        *state["messages"][-4:],
    ])
    return {"messages": [response], "current_agent": "tech_agent"}


def general_agent(state: SwarmState):
    """通用客服 Agent"""
    response = llm.invoke([
        SystemMessage(content="""你是通用客服「小通」。处理一般咨询、产品推荐、活动信息。
用中文简洁回答（不超过100字）。
如果涉及订单问题，回复: '我帮您转接到订单专员。[转交订单]'。
如果涉及技术问题，回复: '我帮您转接到技术支持。[转交技术]'。"""),
        *state["messages"][-4:],
    ])
    return {"messages": [response], "current_agent": "general_agent"}


# ============================================================
# 构建 Swarm Graph
# ============================================================

def build_swarm_graph():
    builder = StateGraph(SwarmState)
    builder.add_node("order_agent", order_agent)
    builder.add_node("tech_agent", tech_agent)
    builder.add_node("general_agent", general_agent)

    # 入口路由
    builder.add_conditional_edges(START, route_entry)

    # 每个 Agent 执行后检查是否需要交接
    builder.add_conditional_edges("order_agent", route_from_agent)
    builder.add_conditional_edges("tech_agent", route_from_agent)
    builder.add_conditional_edges("general_agent", route_from_agent)

    return builder.compile()


# ============================================================
# Demo 1: 直接匹配路由
# ============================================================

def demo_direct_routing():
    print("=" * 50)
    print(f"Demo 15-2 (1/3): 直接路由 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_swarm_graph()

    questions = [
        ("我的订单 ORD-001 到哪了？", "订单客服"),
        ("我的手机连不上 WiFi", "技术支持"),
        ("你们有什么优惠活动？", "通用客服"),
    ]

    for q, expected in questions:
        result = graph.invoke({
            "messages": [HumanMessage(content=q)],
            "current_agent": "",
        })

        agent = result.get("current_agent", "?")
        answer = result["messages"][-1].content
        print(f"问题: {q}  (预期: {expected})")
        print(f"路由: {agent}")
        print(f"回答: {answer[:100]}")
        print()


# ============================================================
# Demo 2: Agent 交接
# ============================================================

def demo_handoff():
    print("=" * 50)
    print(f"Demo 15-2 (2/3): Agent 交接 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_swarm_graph()

    # 先问一个通用问题，然后引导到专业问题
    questions = [
        "你好，我想了解一下你们的产品",
        "我买了一个产品，想查一下订单状态",
    ]

    state_data = {"messages": [], "current_agent": ""}

    for q in questions:
        state_data["messages"].append(HumanMessage(content=q))
        result = graph.invoke(state_data)

        agent = result.get("current_agent", "?")
        answer = result["messages"][-1].content

        print(f"用户: {q}")
        print(f"当前 Agent: {agent}")
        print(f"回答: {answer[:120]}")
        print()

        # 更新状态用于下一轮
        state_data = result


# ============================================================
# Demo 3: 追踪交接过程
# ============================================================

def demo_trace():
    print("=" * 50)
    print(f"Demo 15-2 (3/3): 追踪交接过程 [{QWEN_MODEL}]")
    print("=" * 50)

    graph = build_swarm_graph()

    question = "我遇到一个技术问题，手机无法开机"
    print(f"用户: {question}")
    print("-" * 40)

    for event in graph.stream({
        "messages": [HumanMessage(content=question)],
        "current_agent": "",
    }):
        for node, output in event.items():
            agent = output.get("current_agent", node)
            if "messages" in output:
                msg = output["messages"][-1]
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"  [{agent}] {msg.content[:100]}")

    print()


if __name__ == "__main__":
    demo_direct_routing()
    demo_handoff()
    demo_trace()

    print("=" * 50)
    print("Demo 15-2 完成!")
    print()
    print("Swarm 交接要点:")
    print("  无中心调度      - Agent 间直接交接")
    print("  route_entry     - 入口路由，分配初始 Agent")
    print("  route_from_agent - 检查是否需要交接")
    print("  关键词匹配      - '转交' + 目标 Agent")
    print("  灵活协作        - Agent 自主决定是否交接")
    print("=" * 50)
