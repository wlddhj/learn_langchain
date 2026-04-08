"""
第8章 Demo 2：记忆策略 —— 窗口截断与摘要压缩

演示滑动窗口记忆、摘要记忆、手动管理历史消息。
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 1. 滑动窗口记忆
# ============================================================

class SlidingWindowMemory:
    """只保留最近 N 轮对话"""

    def __init__(self, max_rounds=3):
        self.max_rounds = max_rounds
        self.messages = []

    def add(self, role: str, content: str):
        """添加消息"""
        if role == "user":
            self.messages.append(HumanMessage(content=content))
        else:
            self.messages.append(AIMessage(content=content))
        self._trim()

    def _trim(self):
        """保留最近 N 轮（2N 条消息）"""
        max_msgs = self.max_rounds * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def get_messages(self):
        return list(self.messages)

    def clear(self):
        self.messages = []


def demo_sliding_window():
    print("=" * 50)
    print("Demo 8-2 (1/3): 滑动窗口记忆")
    print("=" * 50)

    memory = SlidingWindowMemory(max_rounds=2)  # 只保留最近2轮

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
    ])

    conversations = [
        ("user", "第1轮：我叫小明"),
        ("user", "第2轮：我喜欢编程"),
        ("user", "第3轮：我在学 LangChain"),
        ("user", "请告诉我，我叫什么名字？我在学什么？"),
    ]

    for role, msg in conversations:
        memory.add("user", msg)

        # 构建带历史的消息
        messages = [("system", "你是一个有帮助的助手。")]
        for m in memory.get_messages():
            if isinstance(m, HumanMessage):
                messages.append(("human", m.content))
            else:
                messages.append(("ai", m.content))

        chain = ChatPromptTemplate.from_messages(messages) | llm | StrOutputParser()
        response = chain.invoke({})
        memory.add("ai", response)

        print(f"用户: {msg}")
        print(f"AI:   {response[:100]}")
        print(f"  当前窗口: {len(memory.get_messages())} 条消息")
        print()

    print(f"最终结果: AI 只能看到最近 2 轮（4条消息），早期信息会被丢弃")
    print()


# ============================================================
# 2. 摘要记忆
# ============================================================

class SummaryMemory:
    """超过阈值时，用 LLM 总结早期对话"""

    def __init__(self, max_messages=6):
        self.max_messages = max_messages
        self.messages = []
        self.summary = ""

    def add(self, role: str, content: str):
        if role == "user":
            self.messages.append(HumanMessage(content=content))
        else:
            self.messages.append(AIMessage(content=content))

        if len(self.messages) > self.max_messages:
            self._summarize()

    def _summarize(self):
        """总结早期消息"""
        to_summarize = self.messages[:-4]  # 保留最近4条
        self.messages = self.messages[-4:]

        # 构建总结 prompt
        history_text = ""
        for m in to_summarize:
            role = "用户" if isinstance(m, HumanMessage) else "AI"
            # 截取前100个字符，避免过长
            content = m.content[:100].replace("{", "{{").replace("}", "}}")  # 转义花括号
            history_text += f"{role}: {content}\n"

        summary_prompt = f"请用1-2句话总结以下对话的关键信息：\n\n{history_text}\n总结："

        existing = f"之前的摘要：{self.summary}\n\n" if self.summary else ""

        # 调用 LLM 并获取响应
        response = llm.invoke(existing + summary_prompt)

        # 确保获取的是字符串内容
        if hasattr(response, 'content'):
            new_summary = str(response.content)
        else:
            new_summary = str(response)

        # 清理可能的花括号
        new_summary = new_summary.replace("{", "{{").replace("}", "}}")

        self.summary = new_summary

    def get_context(self):
        """获取完整上下文：摘要 + 最近消息"""
        context = []
        if self.summary:
            context.append(SystemMessage(content=f"之前的对话摘要：{self.summary}"))
        context.extend(self.messages)
        return context


def demo_summary_memory():
    print("=" * 50)
    print(f"Demo 8-2 (2/3): 摘要记忆 [{QWEN_MODEL}]")
    print("=" * 50)

    memory = SummaryMemory(max_messages=6)

    conversations = [
        "我叫小红，我是一名前端开发者。",
        "我擅长 React 和 TypeScript。",
        "我有5年的开发经验。",
        "最近在学习 Next.js。",
        "我想转型做全栈开发。",
        "你能推荐一些后端技术吗？",  # 超过阈值，触发摘要
        "基于我的背景，你觉得我应该先学什么？",
    ]

    for msg in conversations:
        memory.add("user", msg)

        # 构建消息列表 - 不使用 ChatPromptTemplate
        messages = [SystemMessage(content="你是一个有帮助的助手。")]
        context = memory.get_context()
        for m in context:
            if isinstance(m, SystemMessage):
                messages.append(SystemMessage(content=m.content))
            elif isinstance(m, HumanMessage):
                messages.append(HumanMessage(content=m.content))
            else:
                messages.append(AIMessage(content=m.content))

        # 添加当前用户消息
        messages.append(HumanMessage(content=msg))

        # 直接调用 LLM
        response = llm.invoke(messages)
        if hasattr(response, 'content'):
            response_text = str(response.content)
        else:
            response_text = str(response)

        memory.add("ai", response_text)

        print(f"用户: {msg}")
        print(f"  消息数: {len(memory.messages)}, 摘要: {memory.summary[:60] if memory.summary else '无'}")
        print()

    print(f"最终摘要: {memory.summary}")
    print()


# ============================================================
# 3. Token 预算控制
# ============================================================

def demo_token_budget():
    print("=" * 50)
    print("Demo 8-2 (3/3): Token 预算控制")
    print("=" * 50)

    # 模拟一组历史消息
    history = [
        HumanMessage(content="请介绍一下 Python"),
        AIMessage(content="Python 是一种解释型高级编程语言..."),
        HumanMessage(content="Python 有哪些应用领域？"),
        AIMessage(content="Python 广泛应用于 Web 开发、数据分析、AI 等领域..."),
        HumanMessage(content="推荐一些学习资源"),
        AIMessage(content="推荐以下资源：1. 官方文档 2. Real Python 3. LeetCode..."),
        HumanMessage(content="Python 和 Go 哪个更适合后端？"),
        AIMessage(content="取决于场景。Python 适合快速开发，Go 适合高并发..."),
    ]

    def trim_by_char_budget(messages, budget=200):
        """按字符预算截断（粗略估算：4字符≈1 token）"""
        trimmed = []
        total = 0
        for msg in reversed(messages):
            msg_len = len(msg.content)
            if total + msg_len > budget:
                break
            trimmed.insert(0, msg)
            total += msg_len
        return trimmed, total

    for budget in [100, 200, 500]:
        trimmed, total = trim_by_char_budget(history, budget)
        print(f"预算={budget} 字符: 保留 {len(trimmed)}/{len(history)} 条消息 (实际 {total} 字符)")
        if trimmed:
            first = trimmed[0]
            role = "用户" if isinstance(first, HumanMessage) else "AI"
            print(f"  最早保留: [{role}] {first.content[:40]}...")
        print()

    print("建议: 生产环境中使用 tiktoken 精确计算 token 数")
    print()


if __name__ == "__main__":
    demo_sliding_window()
    demo_summary_memory()
    demo_token_budget()

    print("=" * 50)
    print("Demo 8-2 完成!")
    print()
    print("记忆策略对比:")
    print("  完整历史   - 信息不丢失，但 token 消耗大")
    print("  滑动窗口   - 简单高效，但丢失早期信息")
    print("  摘要记忆   - 节省 token，但摘要可能丢失细节")
    print("  推荐       - 最近5轮完整 + 早期摘要（混合策略）")
    print("=" * 50)
