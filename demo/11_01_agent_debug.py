"""
第11章 Demo 1：Agent 调试与执行追踪

演示 stream 事件追踪、自定义回调、执行过程可视化。
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
from langchain_core.callbacks import BaseCallbackHandler
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
    """查询城市天气信息"""
    weather = {"北京": "晴天 25°C", "上海": "多云 22°C", "深圳": "小雨 28°C"}
    return weather.get(city, f"未找到 {city} 的天气")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"不安全的表达式: {expression}"
        return f"{expression} = {eval(expression)}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def translate(text: str, target_lang: str = "en") -> str:
    """翻译文本。text: 要翻译的文本，target_lang: 目标语言(en/ja/ko)"""
    translations = {
        "你好": {"en": "Hello", "ja": "こんにちは", "ko": "안녕하세요"},
        "谢谢": {"en": "Thank you", "ja": "ありがとう", "ko": "감사합니다"},
        "再见": {"en": "Goodbye", "ja": "さようなら", "ko": "안녕히 가세요"},
    }
    if text in translations and target_lang in translations[text]:
        return translations[text][target_lang]
    return f"[模拟翻译] '{text}' → {target_lang}"


# ============================================================
# 1. 消息级别追踪
# ============================================================

def demo_message_trace():
    print("=" * 50)
    print(f"Demo 11-1 (1/3): 消息级别追踪 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_weather, calculate, translate],
    )

    result = agent.invoke(
        {"messages": [("user", "北京天气怎么样？把'你好'翻译成日语")]},
        config={"recursion_limit": 10},
    )

    # 逐条分析消息
    print("问题: 北京天气怎么样？把'你好'翻译成日语")
    print()

    tool_call_count = 0
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__

        if isinstance(msg, HumanMessage):
            print(f"  [{i}] {msg_type}: {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_count += 1
                    print(f"  [{i}] {msg_type} (tool_call #{tool_call_count}): "
                          f"{tc['name']}({tc['args']})")
            else:
                content_preview = msg.content[:120].replace("\n", " ")
                print(f"  [{i}] {msg_type} (最终回答): {content_preview}")

        elif isinstance(msg, ToolMessage):
            print(f"  [{i}] {msg_type}: {msg.content[:80]}")

    print()
    print(f"统计: {len(result['messages'])} 条消息, {tool_call_count} 次工具调用")
    print()


# ============================================================
# 2. 自定义回调追踪
# ============================================================

class DebugCallbackHandler(BaseCallbackHandler):
    """自定义回调，详细记录 Agent 行为"""

    def __init__(self):
        self.llm_calls = 0
        self.tool_calls = 0
        self.log = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
        self.log.append(f"[LLM #{self.llm_calls}] 开始思考...")

    def on_llm_end(self, response, **kwargs):
        self.log.append(f"[LLM #{self.llm_calls}] 思考完成")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_calls += 1
        tool_name = serialized.get("name", "unknown")
        self.log.append(f"[Tool #{self.tool_calls}] {tool_name} 输入: {str(input_str)[:60]}")

    def on_tool_end(self, output, **kwargs):
        self.log.append(f"[Tool #{self.tool_calls}] 结果: {str(output)[:80]}")


def demo_callback_trace():
    print("=" * 50)
    print("Demo 11-1 (2/3): 自定义回调追踪")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_weather, calculate, translate],
    )

    callback = DebugCallbackHandler()

    result = agent.invoke(
        {"messages": [("user", "上海天气如何？25*4是多少？")]},
        config={
            "recursion_limit": 10,
            "callbacks": [callback],
        },
    )

    print("问题: 上海天气如何？25*4是多少？")
    print("-" * 40)
    for entry in callback.log:
        print(f"  {entry}")

    print()
    print(f"最终回答: {result['messages'][-1].content[:150]}")
    print(f"统计: LLM 调用 {callback.llm_calls} 次, 工具调用 {callback.tool_calls} 次")
    print()


# ============================================================
# 3. stream 事件级别追踪
# ============================================================

def demo_stream_events():
    print("=" * 50)
    print(f"Demo 11-1 (3/3): stream 事件追踪 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_weather, calculate],
    )

    print("问题: 北京和深圳的温差是多少？")
    print("-" * 40)

    step = 0
    for event in agent.stream(
        {"messages": [("user", "北京和深圳的温差是多少？")]},
        config={"recursion_limit": 10},
    ):
        for node_name, node_output in event.items():
            msg = node_output["messages"][-1]

            if node_name == "agent":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        step += 1
                        print(f"  [Step {step}] Agent → 调用 {tc['name']}({tc['args']})")
                elif msg.content:
                    step += 1
                    print(f"  [Step {step}] Agent → 最终回答: {msg.content[:120]}")

            elif node_name == "tools":
                print(f"         工具返回: {msg.content[:80]}")

    print()
    print(f"总步骤: {step}")
    print()


if __name__ == "__main__":
    demo_message_trace()
    demo_callback_trace()
    demo_stream_events()

    print("=" * 50)
    print("Demo 11-1 完成!")
    print()
    print("Agent 调试要点:")
    print("  messages 列表      - 逐条查看完整执行过程")
    print("  自定义 Callback     - 记录 LLM 和工具调用次数")
    print("  stream()           - 实时追踪每个节点的输入输出")
    print("  工具调用链          - AIMessage.tool_calls → ToolMessage → AIMessage")
    print("=" * 50)
