"""
第33章 Demo 3：反馈系统与问答集成

演示完整的反馈系统集成。
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
from collections import Counter

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


class QAWithFeedback:
    """带反馈的问答系统"""

    def __init__(self):
        self.qa_records = {}
        self.feedbacks = {}

    def ask(self, question: str) -> dict:
        """问答"""
        # 生成答案
        answer = llm.invoke(question).content

        # 记录
        qa_id = str(uuid.uuid4())[:8]
        self.qa_records[qa_id] = {
            "id": qa_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
        }

        return {
            "qa_id": qa_id,
            "question": question,
            "answer": answer,
        }

    def collect_feedback(self, qa_id: str, rating: int, comment: str = None) -> str:
        """收集反馈"""
        feedback_id = str(uuid.uuid4())[:8]
        self.feedbacks[feedback_id] = {
            "id": feedback_id,
            "qa_id": qa_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now(),
        }
        return feedback_id

    def get_stats(self) -> dict:
        """统计"""
        if not self.feedbacks:
            return {"total": 0}

        ratings = [f["rating"] for f in self.feedbacks.values()]
        avg_rating = sum(ratings) / len(ratings)

        return {
            "total_qa": len(self.qa_records),
            "total_feedback": len(self.feedbacks),
            "avg_rating": avg_rating,
            "distribution": Counter(ratings),
        }


def demo_full_system():
    """完整系统演示"""
    print("=" * 60)
    print(f"Demo 33-3: 反馈系统与问答集成 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    system = QAWithFeedback()

    # 问答并收集反馈
    questions = [
        "Python 是什么？",
        "什么是 RAG？",
        "LangChain 有什么用？",
    ]

    print("问答测试：")
    print("-" * 60)

    for q in questions:
        result = system.ask(q)

        print(f"问题: {q}")
        print(f"答案: {result['answer'][:50]}...")
        print(f"qa_id: {result['qa_id']}")

        # 模拟用户反馈
        rating = 4 if "Python" in q else 5
        system.collect_feedback(result["qa_id"], rating, "满意")

        print(f"已收集反馈: rating={rating}")
        print()

    # 统计
    stats = system.get_stats()
    print("系统统计：")
    print("-" * 60)
    print(f"问答总数: {stats['total_qa']}")
    print(f"反馈总数: {stats['total_feedback']}")
    print(f"平均评分: {stats['avg_rating']:.2f}")
    print(f"评分分布: {dict(stats['distribution'])}")


if __name__ == "__main__":
    demo_full_system()

    print("=" * 60)
    print("Demo 33-3 完成!")
    print()
    print("反馈系统集成要点：")
    print("  - 问答时记录 qa_id")
    print("  - 反馈关联到 qa_id")
    print("  - 实时统计反馈数据")
    print()
    print("扩展建议：")
    print("  - 添加持久化存储")
    print("  - 实现 A/B 测试")
    print("  - 自动生成改进建议")
    print("  - 反馈驱动 Prompt 优化")
    print("=" * 60)