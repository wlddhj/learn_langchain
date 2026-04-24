"""
第33章 Demo 1：用户反馈收集系统

演示反馈收集的基本实现。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class FeedbackCollector:
    """反馈收集器"""

    def __init__(self):
        self.feedbacks = {}
        self.qa_records = {}

    def record_qa(self, question: str, answer: str) -> str:
        """记录问答"""
        qa_id = str(uuid.uuid4())[:8]
        self.qa_records[qa_id] = {
            "id": qa_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
        }
        return qa_id

    def collect_rating(self, qa_id: str, user_id: str, rating: int) -> str:
        """收集评分"""
        feedback_id = str(uuid.uuid4())[:8]
        self.feedbacks[feedback_id] = {
            "id": feedback_id,
            "qa_id": qa_id,
            "type": "rating",
            "user_id": user_id,
            "rating": rating,
            "timestamp": datetime.now(),
        }
        return feedback_id

    def collect_correctness(self, qa_id: str, user_id: str, is_correct: bool) -> str:
        """收集正确性反馈"""
        feedback_id = str(uuid.uuid4())[:8]
        self.feedbacks[feedback_id] = {
            "id": feedback_id,
            "qa_id": qa_id,
            "type": "correctness",
            "user_id": user_id,
            "is_correct": is_correct,
            "timestamp": datetime.now(),
        }
        return feedback_id

    def collect_text(self, qa_id: str, user_id: str, comment: str) -> str:
        """收集文字反馈"""
        feedback_id = str(uuid.uuid4())[:8]
        self.feedbacks[feedback_id] = {
            "id": feedback_id,
            "qa_id": qa_id,
            "type": "text",
            "user_id": user_id,
            "comment": comment,
            "timestamp": datetime.now(),
        }
        return feedback_id


def demo_feedback_types():
    """反馈类型介绍"""
    print("=" * 60)
    print("Demo 33-1: 用户反馈收集系统")
    print("=" * 60)
    print()

    print("反馈类型：")
    print("-" * 60)
    print("""
| 类型 | 说明 | 用途 |
|------|------|------|
| 评分反馈 | 用户对答案评分（1-5星） | 整体质量评估 |
| 正确性反馈 | 答案是否正确 | 准确性评估 |
| 有用性反馈 | 答案是否有帮助 | 实用性评估 |
| 文字反馈 | 用户详细评论 | 问题分析 |
| 对比反馈 | 多个答案对比选择 | A/B 测试 |
| 行为反馈 | 用户行为记录 | 隐性反馈 |
""")
    print()


def demo_feedback_collection():
    """反馈收集演示"""
    print("=" * 60)
    print("Demo 33-1 (2/3): 反馈收集流程")
    print("=" * 60)
    print()

    collector = FeedbackCollector()

    # 模拟问答
    qa_id = collector.record_qa(
        "产品价格是多少？",
        "产品价格 99 元/月。"
    )
    print(f"问答记录: qa_id={qa_id}")
    print()

    # 收集各类反馈
    rating_id = collector.collect_rating(qa_id, "user_001", 4)
    print(f"评分反馈: rating=4, feedback_id={rating_id}")

    correctness_id = collector.collect_correctness(qa_id, "user_001", True)
    print(f"正确性反馈: is_correct=True, feedback_id={correctness_id}")

    text_id = collector.collect_text(qa_id, "user_001", "回答简洁准确")
    print(f"文字反馈: comment='回答简洁准确', feedback_id={text_id}")
    print()

    print(f"收集反馈总数: {len(collector.feedbacks)}")


def demo_api_design():
    """API 设计示例"""
    print("=" * 60)
    print("Demo 33-1 (3/3): API 设计")
    print("=" * 60)
    print()

    print("反馈 API 设计：")
    print("-" * 60)
    print("""
POST /feedback/rating
{
    "qa_id": "abc123",
    "rating": 4
}

POST /feedback/correctness
{
    "qa_id": "abc123",
    "is_correct": true,
    "correct_answer": null
}

POST /feedback/text
{
    "qa_id": "abc123",
    "comment": "回答准确",
    "category": "好评"
}

GET /feedback/{qa_id}
返回该问答的所有反馈
""")


if __name__ == "__main__":
    demo_feedback_types()
    demo_feedback_collection()
    demo_api_design()

    print("=" * 60)
    print("Demo 33-1 完成!")
    print()
    print("反馈收集要点：")
    print("  - 多渠道收集（评分、正确性、文字）")
    print("  - 关联问答记录（qa_id）")
    print("  - 记录时间戳（用于分析）")
    print("=" * 60)