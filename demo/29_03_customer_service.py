"""
第29章 Demo 3：完整客服系统模拟

演示意图识别 + 情绪分析 + 响应生成的完整流程。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

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


class SimpleCustomerService:
    """简化版客服系统"""

    def __init__(self):
        self.sessions = {}
        self.intent_types = """
- product_query: 产品咨询
- order_status: 订单查询
- complaint: 投诉建议
- tech_support: 技术支持
- return_refund: 退换货
- general_chat: 闲聊
"""
        self.sentiment_types = """
- positive: 积极
- neutral: 中性
- negative: 负面
- angry: 愤怒
"""

    def create_session(self, user_id: str) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            "user_id": user_id,
            "history": [],
            "created_at": datetime.now(),
        }
        return session_id

    def classify_intent(self, message: str) -> dict:
        """意图分类"""
        prompt = ChatPromptTemplate.from_template("""
分类用户意图：
用户消息：{message}
可选意图：{intent_types}
输出格式：意图: xxx
""")
        result = (prompt | llm | StrOutputParser()).invoke({
            "message": message,
            "intent_types": self.intent_types,
        })

        # 解析结果
        intent = "unknown"
        for line in result.split("\n"):
            if "意图:" in line:
                intent = line.split("意图:")[-1].strip().split()[0]
                break

        return {"intent": intent, "raw_result": result}

    def analyze_sentiment(self, message: str) -> dict:
        """情绪分析"""
        prompt = ChatPromptTemplate.from_template("""
分析用户情绪：
用户消息：{message}
可选情绪：{sentiment_types}
输出格式：情绪: xxx, 分数: xxx
""")
        result = (prompt | llm | StrOutputParser()).invoke({
            "message": message,
            "sentiment_types": self.sentiment_types,
        })

        # 解析结果
        sentiment = "neutral"
        score = 0.0
        for line in result.split("\n"):
            if "情绪:" in line:
                sentiment = line.split("情绪:")[-1].strip().split()[0]
            if "分数:" in line:
                try:
                    score = float(line.split("分数:")[-1].strip())
                except:
                    pass

        return {"sentiment": sentiment, "score": score}

    def generate_response(self, message: str, intent: str, sentiment: str) -> str:
        """生成响应"""
        strategies = {
            "positive": "友好热情回应",
            "neutral": "专业高效回应",
            "negative": "耐心安抚，提供解决方案",
            "angry": "优先道歉安抚",
        }

        strategy = strategies.get(sentiment, "专业回应")

        prompt = ChatPromptTemplate.from_template("""
作为客服助手，回复用户消息。

用户消息：{message}
用户意图：{intent}
用户情绪：{sentiment}
响应策略：{strategy}

请生成简短的客服回复（不超过50字）。
""")
        return (prompt | llm | StrOutputParser()).invoke({
            "message": message,
            "intent": intent,
            "sentiment": sentiment,
            "strategy": strategy,
        })

    def chat(self, session_id: str, message: str) -> dict:
        """完整聊天流程"""
        # 1. 意图识别
        intent_result = self.classify_intent(message)

        # 2. 情绪分析
        sentiment_result = self.analyze_sentiment(message)

        # 3. 生成响应
        response = self.generate_response(
            message,
            intent_result["intent"],
            sentiment_result["sentiment"],
        )

        # 4. 判断是否转人工
        transfer_needed = sentiment_result["score"] < -0.5

        # 5. 记录历史
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                "user": message,
                "assistant": response,
                "intent": intent_result["intent"],
                "sentiment": sentiment_result["sentiment"],
            })

        return {
            "response": response,
            "intent": intent_result["intent"],
            "sentiment": sentiment_result["sentiment"],
            "score": sentiment_result["score"],
            "transfer_needed": transfer_needed,
        }


def demo_full_flow():
    """完整流程演示"""
    print("=" * 60)
    print(f"Demo 29-3: 完整客服系统模拟 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    service = SimpleCustomerService()

    # 创建会话
    session_id = service.create_session("user_001")
    print(f"创建会话: {session_id}")
    print()

    # 多轮对话测试
    conversations = [
        "你好，我想咨询一下你们的产品",
        "这个产品的价格是多少？",
        "我买的这个产品坏了，怎么办？",
        "这太糟糕了！我等了三天都没人处理！",
    ]

    print("多轮对话测试：")
    print("-" * 60)

    for msg in conversations:
        result = service.chat(session_id, msg)

        print(f"用户: {msg}")
        print(f"客服: {result['response']}")
        print(f"意图: {result['intent']}, 情绪: {result['sentiment']} ({result['score']})")

        if result["transfer_needed"]:
            print("⚠️  情绪分数过低，建议转人工!")

        print()


def demo_session_stats():
    """会话统计"""
    print("=" * 60)
    print("会话统计")
    print("=" * 60)
    print()

    print("会话历史分析：")
    print("-" * 60)
    print("""
会话统计指标：
- 对话轮数
- 意图分布
- 情绪变化趋势
- 平均响应时间
- 转人工次数
- 问题解决率
""")
    print()

    print("使用场景：")
    print("  - 客服绩效评估")
    print("  - 用户满意度分析")
    print("  - 产品问题发现")
    print("  - 服务优化方向")


if __name__ == "__main__":
    demo_full_flow()
    demo_session_stats()

    print("=" * 60)
    print("Demo 29-3 完成!")
    print()
    print("客服系统核心流程：")
    print("  1. 意图识别 → 确定处理路径")
    print("  2. 情绪分析 → 选择响应策略")
    print("  3. 响应生成 → 生成回复内容")
    print("  4. 转人工判断 → 处理复杂情况")
    print()
    print("扩展建议：")
    print("  - 集成知识库 RAG")
    print("  - 添加订单查询 Agent")
    print("  - 实现工单系统")
    print("  - 添加对话记忆")
    print("=" * 60)