"""
第33章 Demo 2：反馈分析系统

演示反馈统计分析和改进建议生成。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import uuid

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class FeedbackAnalyzer:
    """反馈分析器"""

    def __init__(self, feedbacks: dict):
        self.feedbacks = feedbacks

    def analyze_ratings(self) -> dict:
        """评分分析"""
        ratings = [
            f["rating"] for f in self.feedbacks.values()
            if f["type"] == "rating"
        ]

        if not ratings:
            return {}

        return {
            "total": len(ratings),
            "average": sum(ratings) / len(ratings),
            "distribution": Counter(ratings),
            "positive_rate": len([r for r in ratings if r >= 4]) / len(ratings),
        }

    def analyze_correctness(self) -> dict:
        """正确性分析"""
        correctness = [
            f["is_correct"] for f in self.feedbacks.values()
            if f["type"] == "correctness"
        ]

        if not correctness:
            return {}

        return {
            "total": len(correctness),
            "accuracy": sum(correctness) / len(correctness),
        }

    def identify_problems(self) -> list:
        """识别问题问答"""
        problems = []

        for f in self.feedbacks.values():
            # 低评分
            if f["type"] == "rating" and f["rating"] <= 2:
                problems.append(f["qa_id"])

            # 标记错误
            if f["type"] == "correctness" and not f["is_correct"]:
                problems.append(f["qa_id"])

        return list(set(problems))

    def get_summary(self) -> dict:
        """整体摘要"""
        return {
            "total_feedbacks": len(self.feedbacks),
            "rating_analysis": self.analyze_ratings(),
            "correctness_analysis": self.analyze_correctness(),
            "problem_qas": self.identify_problems(),
        }


def demo_feedback_analysis():
    """反馈分析演示"""
    print("=" * 60)
    print("Demo 33-2: 反馈分析系统")
    print("=" * 60)
    print()

    # 模拟反馈数据
    feedbacks = {}
    for i in range(20):
        feedback_id = str(uuid.uuid4())[:8]
        qa_id = f"qa_{i % 5}"

        # 模拟不同类型反馈
        if i % 3 == 0:
            feedbacks[feedback_id] = {
                "id": feedback_id,
                "qa_id": qa_id,
                "type": "rating",
                "rating": 4 if i % 5 != 0 else 2,  # qa_0 有低评分
                "timestamp": datetime.now(),
            }
        elif i % 3 == 1:
            feedbacks[feedback_id] = {
                "id": feedback_id,
                "qa_id": qa_id,
                "type": "correctness",
                "is_correct": i % 5 != 0,  # qa_0 标记错误
                "timestamp": datetime.now(),
            }
        else:
            feedbacks[feedback_id] = {
                "id": feedback_id,
                "qa_id": qa_id,
                "type": "text",
                "comment": "好评" if i % 5 != 0 else "答案错误",
                "timestamp": datetime.now(),
            }

    analyzer = FeedbackAnalyzer(feedbacks)

    # 评分分析
    rating_analysis = analyzer.analyze_ratings()
    print("评分分析：")
    print("-" * 60)
    print(f"总评分数: {rating_analysis['total']}")
    print(f"平均评分: {rating_analysis['average']:.2f}")
    print(f"评分分布: {dict(rating_analysis['distribution'])}")
    print(f"好评率: {rating_analysis['positive_rate']:.2%}")
    print()

    # 正确性分析
    correctness_analysis = analyzer.analyze_correctness()
    print("正确性分析：")
    print("-" * 60)
    print(f"总反馈数: {correctness_analysis['total']}")
    print(f"准确率: {correctness_analysis['accuracy']:.2%}")
    print()

    # 问题识别
    problems = analyzer.identify_problems()
    print("问题问答：")
    print("-" * 60)
    print(f"问题问答ID: {problems}")
    print()


def demo_improvement_cycle():
    """改进循环示意"""
    print("=" * 60)
    print("Demo 33-2 (2/2): 改进循环")
    print("=" * 60)
    print()

    print("反馈驱动改进流程：")
    print("-" * 60)
    print("""
1. 收集反馈 → 评分、正确性、评论
    ↓
2. 分析反馈 → 统计、识别问题
    ↓
3. 定位问题 → 问题问答、低评分原因
    ↓
4. 改进措施 → Prompt优化、知识库更新
    ↓
5. 验证效果 → A/B测试、再评估
    ↓
6. 循环迭代 → 持续优化
""")
    print()

    print("改进措施示例：")
    print("-" * 60)
    print("""
发现: qa_0 评分低、标记错误
分析: 答案内容不准确
改进:
  1. 更新知识库内容
  2. 调整 Prompt 约束
  3. 增加来源引用要求
验证: A/B测试新版本
""")


if __name__ == "__main__":
    demo_feedback_analysis()
    demo_improvement_cycle()

    print("=" * 60)
    print("Demo 33-2 完成!")
    print()
    print("反馈分析要点：")
    print("  - 评分统计：平均、分布、好评率")
    print("  - 正确性分析：准确率")
    print("  - 问题识别：低评分、错误标记")
    print()
    print("改进循环：")
    print("  收集 → 分析 → 定位 → 改进 → 验证 → 迭代")
    print("=" * 60)