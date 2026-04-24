# 第33章：用户反馈系统

## 33.1 为什么需要用户反馈系统

用户反馈是改进 AI 应用质量的关键数据来源，能够帮助：
- 发现模型缺陷和不足
- 收集真实用户需求
- 优化 Prompt 和策略
- 训练更好的模型
- 建立质量评估体系

### 反馈类型

| 类型 | 说明 | 用途 |
|------|------|------|
| **评分反馈** | 用户对答案评分（1-5星） | 整体质量评估 |
| **正确性反馈** | 答案是否正确 | 准确性评估 |
| **有用性反馈** | 答案是否有帮助 | 实用性评估 |
| **文本反馈** | 用户详细评论 | 问题分析 |
| **对比反馈** | 多个答案对比选择 | A/B 测试 |
| **行为反馈** | 用户行为记录 | 隐性反馈 |

## 33.2 反馈数据模型

### 基础数据结构

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class FeedbackType(str, Enum):
    RATING = "rating"          # 评分
    CORRECTNESS = "correctness"  # 正确性
    USEFULNESS = "usefulness"    # 有用性
    COMMENT = "comment"         # 文字评论
    COMPARISON = "comparison"    # 对比选择

class RatingFeedback(BaseModel):
    """评分反馈"""
    id: str
    qa_id: str               # 问答 ID
    user_id: str
    rating: int = Field(ge=1, le=5)  # 1-5 星
    created_at: datetime
    metadata: dict = {}

class CorrectnessFeedback(BaseModel):
    """正确性反馈"""
    id: str
    qa_id: str
    user_id: str
    is_correct: bool
    correct_answer: Optional[str] = None  # 用户提供的正确答案
    created_at: datetime
    metadata: dict = {}

class UsefulnessFeedback(BaseModel):
    """有用性反馈"""
    id: str
    qa_id: str
    user_id: str
    is_useful: bool
    reason: Optional[str] = None
    created_at: datetime
    metadata: dict = {}

class TextFeedback(BaseModel):
    """文字反馈"""
    id: str
    qa_id: str
    user_id: str
    comment: str
    category: Optional[str] = None  # 分类：问题/建议/投诉等
    created_at: datetime
    metadata: dict = {}

class ComparisonFeedback(BaseModel):
    """对比反馈"""
    id: str
    qa_id: str
    user_id: str
    selected_answer: str     # 用户选择的答案 ID
    options: List[str]       # 对比选项列表
    reason: Optional[str] = None
    created_at: datetime
    metadata: dict = {}

class QARecord(BaseModel):
    """问答记录"""
    id: str
    question: str
    answer: str
    sources: List[dict] = []
    confidence: float
    model: str
    prompt_template: str
    created_at: datetime
    user_id: str
    session_id: str
    metadata: dict = {}
```

## 33.3 反馈收集系统

### 多渠道反馈收集

```python
from datetime import datetime
import uuid

class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self):
        self.feedback_store = {}
        self.qa_records = {}
    
    def record_qa(
        self,
        question: str,
        answer: str,
        user_id: str,
        session_id: str,
        **kwargs,
    ) -> str:
        """记录问答"""
        qa_id = str(uuid.uuid4())
        
        self.qa_records[qa_id] = QARecord(
            id=qa_id,
            question=question,
            answer=answer,
            sources=kwargs.get("sources", []),
            confidence=kwargs.get("confidence", 0),
            model=kwargs.get("model", ""),
            prompt_template=kwargs.get("prompt_template", ""),
            created_at=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            metadata=kwargs.get("metadata", {}),
        )
        
        return qa_id
    
    def collect_rating(
        self,
        qa_id: str,
        user_id: str,
        rating: int,
    ) -> str:
        """收集评分"""
        feedback_id = str(uuid.uuid4())
        
        self.feedback_store[feedback_id] = RatingFeedback(
            id=feedback_id,
            qa_id=qa_id,
            user_id=user_id,
            rating=rating,
            created_at=datetime.now(),
        )
        
        return feedback_id
    
    def collect_correctness(
        self,
        qa_id: str,
        user_id: str,
        is_correct: bool,
        correct_answer: str = None,
    ) -> str:
        """收集正确性反馈"""
        feedback_id = str(uuid.uuid4())
        
        self.feedback_store[feedback_id] = CorrectnessFeedback(
            id=feedback_id,
            qa_id=qa_id,
            user_id=user_id,
            is_correct=is_correct,
            correct_answer=correct_answer,
            created_at=datetime.now(),
        )
        
        return feedback_id
    
    def collect_text(
        self,
        qa_id: str,
        user_id: str,
        comment: str,
        category: str = None,
    ) -> str:
        """收集文字反馈"""
        feedback_id = str(uuid.uuid4())
        
        self.feedback_store[feedback_id] = TextFeedback(
            id=feedback_id,
            qa_id=qa_id,
            user_id=user_id,
            comment=comment,
            category=category,
            created_at=datetime.now(),
        )
        
        return feedback_id
    
    def collect_comparison(
        self,
        qa_id: str,
        user_id: str,
        selected_answer: str,
        options: List[str],
        reason: str = None,
    ) -> str:
        """收集对比反馈"""
        feedback_id = str(uuid.uuid4())
        
        self.feedback_store[feedback_id] = ComparisonFeedback(
            id=feedback_id,
            qa_id=qa_id,
            user_id=user_id,
            selected_answer=selected_answer,
            options=options,
            reason=reason,
            created_at=datetime.now(),
        )
        
        return feedback_id
    
    def get_feedback_for_qa(self, qa_id: str) -> List:
        """获取某问答的所有反馈"""
        return [
            f for f in self.feedback_store.values()
            if f.qa_id == qa_id
        ]
    
    def get_qa_record(self, qa_id: str) -> QARecord:
        """获取问答记录"""
        return self.qa_records.get(qa_id)


# 使用示例
collector = FeedbackCollector()

# 记录问答
qa_id = collector.record_qa(
    question="产品价格是多少？",
    answer="产品价格 99 元。",
    user_id="user_123",
    session_id="session_abc",
)

# 收集反馈
collector.collect_rating(qa_id, "user_123", 4)
collector.collect_correctness(qa_id, "user_123", True)
collector.collect_text(qa_id, "user_123", "回答简洁明了", category="好评")

# 获取反馈
feedbacks = collector.get_feedback_for_qa(qa_id)
print(f"收到 {len(feedbacks)} 条反馈")
```

### Web 界面反馈组件

```python
# API 接口设计
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="反馈系统 API")

collector = FeedbackCollector()

class RatingRequest(BaseModel):
    qa_id: str
    rating: int

class CorrectnessRequest(BaseModel):
    qa_id: str
    is_correct: bool
    correct_answer: Optional[str] = None

class TextFeedbackRequest(BaseModel):
    qa_id: str
    comment: str
    category: Optional[str] = None

@app.post("/feedback/rating")
async def submit_rating(request: RatingRequest):
    """提交评分"""
    feedback_id = collector.collect_rating(
        request.qa_id,
        "user_123",
        request.rating,
    )
    return {"feedback_id": feedback_id, "status": "success"}

@app.post("/feedback/correctness")
async def submit_correctness(request: CorrectnessRequest):
    """提交正确性反馈"""
    feedback_id = collector.collect_correctness(
        request.qa_id,
        "user_123",
        request.is_correct,
        request.correct_answer,
    )
    return {"feedback_id": feedback_id, "status": "success"}

@app.post("/feedback/text")
async def submit_text_feedback(request: TextFeedbackRequest):
    """提交文字反馈"""
    feedback_id = collector.collect_text(
        request.qa_id,
        "user_123",
        request.comment,
        request.category,
    )
    return {"feedback_id": feedback_id, "status": "success"}

@app.get("/feedback/{qa_id}")
async def get_feedbacks(qa_id: str):
    """获取问答反馈"""
    feedbacks = collector.get_feedback_for_qa(qa_id)
    return {"feedbacks": [f.dict() for f in feedbacks]}
```

## 33.4 反馈分析系统

### 统计分析

```python
from typing import List
from collections import Counter

class FeedbackAnalyzer:
    """反馈分析器"""
    
    def __init__(self, feedback_store: dict):
        self.feedback_store = feedback_store
    
    def analyze_ratings(self) -> dict:
        """评分分析"""
        ratings = [
            f.rating for f in self.feedback_store.values()
            if hasattr(f, 'rating')
        ]
        
        if not ratings:
            return {}
        
        return {
            "total_count": len(ratings),
            "average_rating": sum(ratings) / len(ratings),
            "rating_distribution": Counter(ratings),
            "positive_rate": len([r for r in ratings if r >= 4]) / len(ratings),
            "negative_rate": len([r for r in ratings if r <= 2]) / len(ratings),
        }
    
    def analyze_correctness(self) -> dict:
        """正确性分析"""
        correctness = [
            f.is_correct for f in self.feedback_store.values()
            if hasattr(f, 'is_correct')
        ]
        
        if not correctness:
            return {}
        
        return {
            "total_count": len(correctness),
            "accuracy_rate": sum(correctness) / len(correctness),
            "correct_count": sum(correctness),
            "incorrect_count": len(correctness) - sum(correctness),
        }
    
    def analyze_text_feedbacks(self) -> dict:
        """文字反馈分析"""
        comments = [
            f.comment for f in self.feedback_store.values()
            if hasattr(f, 'comment') and f.comment
        ]
        
        if not comments:
            return {}
        
        # 分类统计
        categories = [
            f.category for f in self.feedback_store.values()
            if hasattr(f, 'category') and f.category
        ]
        
        return {
            "total_count": len(comments),
            "category_distribution": Counter(categories),
            "average_length": sum(len(c) for c in comments) / len(comments),
            "sample_comments": comments[:5],
        }
    
    def get_overall_summary(self) -> dict:
        """整体摘要"""
        return {
            "total_feedbacks": len(self.feedback_store),
            "rating_analysis": self.analyze_ratings(),
            "correctness_analysis": self.analyze_correctness(),
            "text_analysis": self.analyze_text_feedbacks(),
        }
    
    def identify_problem_qas(self) -> List[str]:
        """识别问题问答"""
        # 低评分或标记错误的问答
        problem_qa_ids = set()
        
        for f in self.feedback_store.values():
            if hasattr(f, 'rating') and f.rating <= 2:
                problem_qa_ids.add(f.qa_id)
            
            if hasattr(f, 'is_correct') and not f.is_correct:
                problem_qa_ids.add(f.qa_id)
        
        return list(problem_qa_ids)
    
    def get_trend(self, days: int = 7) -> dict:
        """趋势分析"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        start_date = now - timedelta(days=days)
        
        recent_feedbacks = [
            f for f in self.feedback_store.values()
            if f.created_at >= start_date
        ]
        
        # 每日统计
        daily_stats = {}
        
        for f in recent_feedbacks:
            date_key = f.created_at.strftime("%Y-%m-%d")
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {"count": 0, "ratings": []}
            
            daily_stats[date_key]["count"] += 1
            
            if hasattr(f, 'rating'):
                daily_stats[date_key]["ratings"].append(f.rating)
        
        # 计算每日平均
        for date, stats in daily_stats.items():
            if stats["ratings"]:
                stats["avg_rating"] = sum(stats["ratings"]) / len(stats["ratings"])
        
        return {
            "period_days": days,
            "daily_stats": daily_stats,
            "total_recent": len(recent_feedbacks),
        }


# 使用示例
analyzer = FeedbackAnalyzer(collector.feedback_store)

print("评分分析:", analyzer.analyze_ratings())
print("正确性分析:", analyzer.analyze_correctness())
print("问题问答:", analyzer.identify_problem_qas())
print("趋势:", analyzer.get_trend(7))
```

### LLM 辅助分析

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class LLMAssistedFeedbackAnalyzer:
    """LLM 辅助反馈分析"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def categorize_feedback(self, comment: str) -> str:
        """分类反馈"""
        prompt = ChatPromptTemplate.from_template("""
        将以下用户反馈分类：
        
        反馈内容：{comment}
        
        可选分类：
        - accuracy: 答案准确性问题
        - completeness: 答案完整性问题
        - tone: 回答语气问题
        - speed: 响应速度问题
        - format: 格式问题
        - other: 其他
        
        只输出分类名称。
        """)
        
        return (prompt | self.llm).invoke({"comment": comment}).content
    
    def extract_issues(self, comments: List[str]) -> List[dict]:
        """提取问题"""
        prompt = ChatPromptTemplate.from_template("""
        从以下用户反馈中提取主要问题：
        
        {comments}
        
        请输出：
        1. 主要问题列表（每个问题一句话）
        2. 问题严重程度（高/中/低）
        3. 建议改进方向
        
        输出 JSON 格式。
        """)
        
        comments_text = "\n".join(comments)
        
        result = (prompt | self.llm).invoke({"comments": comments_text}).content
        
        return result
    
    def generate_improvement_suggestions(self, analysis: dict) -> str:
        """生成改进建议"""
        prompt = ChatPromptTemplate.from_template("""
        基于以下反馈分析，生成改进建议：
        
        分析结果：{analysis}
        
        请生成：
        1. 紧急改进项（需立即处理）
        2. 短期改进项（本周处理）
        3. 长期改进项（持续优化）
        """)
        
        return (prompt | self.llm).invoke({"analysis": str(analysis)}).content
    
    def analyze_feedback_sentiment(self, comment: str) -> dict:
        """分析反馈情绪"""
        prompt = ChatPromptTemplate.from_template("""
        分析以下反馈的情绪和意图：
        
        反馈：{comment}
        
        请输出：
        - 情绪：正面/负面/中性
        - 紧急程度：高/中/低
        - 主要诉求
        """)
        
        result = (prompt | self.llm).invoke({"comment": comment}).content
        
        return result


# 使用示例
llm_analyzer = LLMAssistedFeedbackAnalyzer()

# 分类反馈
category = llm_analyzer.categorize_feedback("答案不准确，缺少关键信息")
print("分类:", category)

# 提取问题
comments = [
    "答案不准确",
    "缺少关键信息",
    "回答太慢",
    "格式不清晰",
]

issues = llm_analyzer.extract_issues(comments)
print("问题:", issues)

# 生成建议
analysis = analyzer.get_overall_summary()
suggestions = llm_analyzer.generate_improvement_suggestions(analysis)
print("建议:", suggestions)
```

## 33.5 反馈驱动优化

### 自动优化循环

```python
class FeedbackDrivenOptimizer:
    """反馈驱动优化器"""
    
    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        feedback_analyzer: FeedbackAnalyzer,
    ):
        self.collector = feedback_collector
        self.analyzer = feedback_analyzer
        self.llm_analyzer = LLMAssistedFeedbackAnalyzer()
    
    def optimize_prompt(self, problematic_qas: List[str]) -> dict:
        """基于问题反馈优化 Prompt"""
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 收集问题问答和反馈
        problems = []
        
        for qa_id in problematic_qas:
            qa_record = self.collector.get_qa_record(qa_id)
            feedbacks = self.collector.get_feedback_for_qa(qa_id)
            
            if qa_record:
                problems.append({
                    "question": qa_record.question,
                    "answer": qa_record.answer,
                    "feedbacks": [f.dict() for f in feedbacks],
                })
        
        # 生成改进建议
        prompt = ChatPromptTemplate.from_template("""
        分析以下失败案例，生成 Prompt 改进建议：
        
        失败案例：
        {problems}
        
        请：
        1. 分析失败原因
        2. 提出 Prompt 改进建议
        3. 生成改进后的 Prompt 模板
        
        输出 JSON 格式。
        """)
        
        result = (prompt | llm).invoke({"problems": str(problems)}).content
        
        return {
            "problem_count": len(problematic_qas),
            "analysis": result,
        }
    
    def adjust_retrieval_params(self) -> dict:
        """调整检索参数"""
        # 分析正确性反馈
        correctness_analysis = self.analyzer.analyze_correctness()
        
        if correctness_analysis.get("accuracy_rate", 1) < 0.7:
            # 准确率低，增加检索数量
            suggestion = "增加 top_k 参数，提高检索覆盖率"
            new_top_k = 10
        else:
            suggestion = "保持当前参数"
            new_top_k = 5
        
        return {
            "current_accuracy": correctness_analysis.get("accuracy_rate"),
            "suggestion": suggestion,
            "recommended_top_k": new_top_k,
        }
    
    def prioritize_improvements(self) -> List[dict]:
        """优先级排序改进项"""
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 收集所有问题
        problem_qas = self.analyzer.identify_problem_qas()
        rating_analysis = self.analyzer.analyze_ratings()
        
        prompt = ChatPromptTemplate.from_template("""
        基于以下反馈数据，确定改进优先级：
        
        评分分析：{rating_analysis}
        问题问答数：{problem_count}
        
        请生成改进优先级列表：
        1. 最高优先级改进
        2. 高优先级改进
        3. 中优先级改进
        4. 低优先级改进
        
        输出 JSON 格式列表。
        """)
        
        result = (prompt | llm).invoke({
            "rating_analysis": str(rating_analysis),
            "problem_count": len(problem_qas),
        }).content
        
        return result
    
    def run_optimization_cycle(self) -> dict:
        """执行优化循环"""
        # 1. 分析反馈
        summary = self.analyzer.get_overall_summary()
        
        # 2. 识别问题
        problem_qas = self.analyzer.identify_problem_qas()
        
        # 3. 优化 Prompt
        prompt_improvements = self.optimize_prompt(problem_qas)
        
        # 4. 调整参数
        retrieval_adjustments = self.adjust_retrieval_params()
        
        # 5. 优先级排序
        priorities = self.prioritize_improvements()
        
        return {
            "feedback_summary": summary,
            "problem_qa_count": len(problem_qas),
            "prompt_improvements": prompt_improvements,
            "retrieval_adjustments": retrieval_adjustments,
            "priorities": priorities,
        }
```

### A/B 测试集成

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import random

class ABTestingWithFeedback:
    """带反馈收集的 A/B 测试"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.collector = FeedbackCollector()
        self.ab_results = {}
    
    def create_variant(
        self,
        base_prompt: str,
        variant_type: str,
    ) -> str:
        """创建变体 Prompt"""
        if variant_type == "concise":
            return base_prompt + "\n请简洁回答，不超过50字。"
        elif variant_type == "detailed":
            return base_prompt + "\n请详细回答，包含具体细节。"
        elif variant_type == "structured":
            return base_prompt + "\n请按以下格式回答：\n1. 概述\n2. 详情\n3. 建议"
        else:
            return base_prompt
    
    def run_test(
        self,
        question: str,
        variants: dict,  # {variant_name: prompt_template}
    ) -> dict:
        """运行 A/B 测试"""
        results = {}
        
        for variant_name, prompt_template in variants.items():
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            answer = (prompt | self.llm).invoke({"question": question}).content
            
            qa_id = self.collector.record_qa(
                question=question,
                answer=answer,
                user_id="ab_test",
                session_id="ab_test_session",
                metadata={"variant": variant_name},
            )
            
            results[variant_name] = {
                "qa_id": qa_id,
                "answer": answer,
            }
        
        return results
    
    def collect_comparison_feedback(
        self,
        user_id: str,
        question: str,
        results: dict,
        selected_variant: str,
        reason: str = None,
    ) -> str:
        """收集对比反馈"""
        # 找到选中答案的 qa_id
        selected_qa_id = results[selected_variant]["qa_id"]
        options = [r["qa_id"] for r in results.values()]
        
        feedback_id = self.collector.collect_comparison(
            qa_id=selected_qa_id,
            user_id=user_id,
            selected_answer=selected_qa_id,
            options=options,
            reason=reason,
        )
        
        return feedback_id
    
    def analyze_ab_results(self) -> dict:
        """分析 A/B 测试结果"""
        comparison_feedbacks = [
            f for f in self.collector.feedback_store.values()
            if hasattr(f, 'selected_answer')
        ]
        
        # 统计各变体胜出次数
        win_counts = {}
        
        for f in comparison_feedbacks:
            # 找到变体名称
            qa_record = self.collector.get_qa_record(f.selected_answer)
            if qa_record:
                variant = qa_record.metadata.get("variant", "unknown")
                win_counts[variant] = win_counts.get(variant, 0) + 1
        
        return {
            "total_comparisons": len(comparison_feedbacks),
            "win_counts": win_counts,
            "winner": max(win_counts, key=win_counts.get) if win_counts else None,
        }
    
    def recommend_best_variant(self) -> str:
        """推荐最佳变体"""
        analysis = self.analyze_ab_results()
        
        return analysis.get("winner", "base")


# 使用示例
ab_tester = ABTestingWithFeedback()

# 创建变体
base_prompt = "请回答问题：{question}"
variants = {
    "base": base_prompt,
    "concise": ab_tester.create_variant(base_prompt, "concise"),
    "detailed": ab_tester.create_variant(base_prompt, "detailed"),
}

# 运行测试
question = "产品价格是多少？"
results = ab_tester.run_test(question, variants)

# 用户选择
selected_variant = "concise"
feedback_id = ab_tester.collect_comparison_feedback(
    "user_123",
    question,
    results,
    selected_variant,
    "简洁明了",
)

# 分析结果
print(ab_tester.analyze_ab_results())
print("推荐变体:", ab_tester.recommend_best_variant())
```

## 33.6 反馈数据持久化

### 数据库存储

```python
import sqlite3
import json
from datetime import datetime

class FeedbackDatabase:
    """反馈数据库"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 问答记录表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_records (
            id TEXT PRIMARY KEY,
            question TEXT,
            answer TEXT,
            sources TEXT,
            confidence REAL,
            model TEXT,
            prompt_template TEXT,
            created_at TEXT,
            user_id TEXT,
            session_id TEXT,
            metadata TEXT
        )
        """)
        
        # 反馈表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id TEXT PRIMARY KEY,
            qa_id TEXT,
            feedback_type TEXT,
            user_id TEXT,
            content TEXT,
            created_at TEXT,
            metadata TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def save_qa_record(self, record: QARecord):
        """保存问答记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO qa_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.id,
            record.question,
            record.answer,
            json.dumps(record.sources),
            record.confidence,
            record.model,
            record.prompt_template,
            record.created_at.isoformat(),
            record.user_id,
            record.session_id,
            json.dumps(record.metadata),
        ))
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, feedback):
        """保存反馈"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        content = {}
        feedback_type = ""
        
        if hasattr(feedback, 'rating'):
            feedback_type = "rating"
            content["rating"] = feedback.rating
        elif hasattr(feedback, 'is_correct'):
            feedback_type = "correctness"
            content["is_correct"] = feedback.is_correct
            content["correct_answer"] = feedback.correct_answer
        elif hasattr(feedback, 'comment'):
            feedback_type = "text"
            content["comment"] = feedback.comment
            content["category"] = feedback.category
        elif hasattr(feedback, 'selected_answer'):
            feedback_type = "comparison"
            content["selected_answer"] = feedback.selected_answer
            content["options"] = feedback.options
            content["reason"] = feedback.reason
        
        cursor.execute("""
        INSERT INTO feedbacks VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.id,
            feedback.qa_id,
            feedback_type,
            feedback.user_id,
            json.dumps(content),
            feedback.created_at.isoformat(),
            json.dumps(feedback.metadata),
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedbacks_by_qa(self, qa_id: str) -> List:
        """获取问答的反馈"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM feedbacks WHERE qa_id = ?
        """, (qa_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows
    
    def get_all_feedbacks(self) -> List:
        """获取所有反馈"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM feedbacks")
        rows = cursor.fetchall()
        conn.close()
        
        return rows
    
    def get_qa_record(self, qa_id: str) -> dict:
        """获取问答记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM qa_records WHERE id = ?
        """, (qa_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "question": row[1],
                "answer": row[2],
                "sources": json.loads(row[3]),
                "confidence": row[4],
                "model": row[5],
                "prompt_template": row[6],
                "created_at": row[7],
                "user_id": row[8],
                "session_id": row[9],
                "metadata": json.loads(row[10]),
            }
        
        return None


# 使用示例
db = FeedbackDatabase()

# 保存问答记录
qa_id = collector.record_qa(
    question="产品价格？",
    answer="99元",
    user_id="user_123",
    session_id="session_1",
)
qa_record = collector.get_qa_record(qa_id)
db.save_qa_record(qa_record)

# 保存反馈
feedback_id = collector.collect_rating(qa_id, "user_123", 4)
feedback = collector.feedback_store[feedback_id]
db.save_feedback(feedback)

# 查询
feedbacks = db.get_feedbacks_by_qa(qa_id)
print(f"查询到 {len(feedbacks)} 条反馈")
```

### Redis 存储

```python
import redis
import json

class FeedbackRedisStore:
    """Redis 反馈存储"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
    
    def save_qa_record(self, record: QARecord):
        """保存问答记录"""
        key = f"qa:{record.id}"
        self.redis.set(key, json.dumps(record.dict()))
    
    def save_feedback(self, feedback):
        """保存反馈"""
        key = f"feedback:{feedback.id}"
        self.redis.set(key, json.dumps(feedback.dict()))
        
        # 同时添加到问答的反馈列表
        list_key = f"qa:{feedback.qa_id}:feedbacks"
        self.redis.rpush(list_key, feedback.id)
    
    def get_qa_record(self, qa_id: str) -> dict:
        """获取问答记录"""
        key = f"qa:{qa_id}"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)
        
        return None
    
    def get_feedbacks_for_qa(self, qa_id: str) -> List[dict]:
        """获取问答的反馈"""
        list_key = f"qa:{qa_id}:feedbacks"
        feedback_ids = self.redis.lrange(list_key, 0, -1)
        
        feedbacks = []
        for fid in feedback_ids:
            key = f"feedback:{fid.decode()}"
            data = self.redis.get(key)
            if data:
                feedbacks.append(json.loads(data))
        
        return feedbacks
    
    def get_recent_feedbacks(self, limit: int = 100) -> List[dict]:
        """获取最近反馈"""
        keys = self.redis.keys("feedback:*")
        
        feedbacks = []
        for key in keys[:limit]:
            data = self.redis.get(key)
            if data:
                feedbacks.append(json.loads(data))
        
        return feedbacks
```

## 33.7 行为反馈系统

### 隐性反馈收集

```python
from datetime import datetime
import uuid

class BehaviorFeedbackCollector:
    """行为反馈收集器"""
    
    def __init__(self):
        self.behaviors = {}
    
    def record_user_action(
        self,
        qa_id: str,
        user_id: str,
        action_type: str,
        details: dict = {},
    ) -> str:
        """记录用户行为"""
        behavior_id = str(uuid.uuid4())
        
        self.behaviors[behavior_id] = {
            "id": behavior_id,
            "qa_id": qa_id,
            "user_id": user_id,
            "action_type": action_type,
            "details": details,
            "timestamp": datetime.now(),
        }
        
        return behavior_id
    
    def analyze_behaviors(self) -> dict:
        """分析行为反馈"""
        # 常见行为类型
        action_counts = {}
        
        for b in self.behaviors.values():
            action = b["action_type"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 计算隐性评分
        implicit_scores = {}
        
        for b in self.behaviors.values():
            qa_id = b["qa_id"]
            action = b["action_type"]
            
            # 根据行为类型推断评分
            score_mapping = {
                "copy_answer": 4,       # 复制答案 -> 有用
                "regenerate": 2,        # 重新生成 -> 不满意
                "follow_up": 3,         #追问 -> 有兴趣
                "close_immediately": 1, # 立即关闭 -> 不满意
                "share": 5,             # 分享 -> 非常满意
                "bookmark": 4,          # 收藏 -> 有用
            }
            
            score = score_mapping.get(action, 3)
            
            if qa_id not in implicit_scores:
                implicit_scores[qa_id] = []
            
            implicit_scores[qa_id].append(score)
        
        # 计算平均隐性评分
        avg_implicit_scores = {
            qa_id: sum(scores) / len(scores)
            for qa_id, scores in implicit_scores.items()
        }
        
        return {
            "total_behaviors": len(self.behaviors),
            "action_distribution": action_counts,
            "implicit_scores": avg_implicit_scores,
        }


# 使用示例
behavior_collector = BehaviorFeedbackCollector()

qa_id = "qa_123"

# 记录各种行为
behavior_collector.record_user_action(qa_id, "user_123", "view", {})
behavior_collector.record_user_action(qa_id, "user_123", "copy_answer", {})
behavior_collector.record_user_action(qa_id, "user_123", "bookmark", {})

print(behavior_collector.analyze_behaviors())
```

### 时间停留分析

```python
class TimeSpentAnalyzer:
    """时间停留分析"""
    
    def __init__(self):
        self.time_records = {}
    
    def record_view_time(
        self,
        qa_id: str,
        user_id: str,
        view_duration: float,  # 秒
    ) -> str:
        """记录停留时间"""
        record_id = str(uuid.uuid4())
        
        self.time_records[record_id] = {
            "id": record_id,
            "qa_id": qa_id,
            "user_id": user_id,
            "view_duration": view_duration,
            "timestamp": datetime.now(),
        }
        
        return record_id
    
    def analyze(self) -> dict:
        """分析停留时间"""
        if not self.time_records:
            return {}
        
        durations = [r["view_duration"] for r in self.time_records.values()]
        
        # 按问答分组
        qa_durations = {}
        
        for r in self.time_records.values():
            qa_id = r["qa_id"]
            
            if qa_id not in qa_durations:
                qa_durations[qa_id] = []
            
            qa_durations[qa_id].append(r["view_duration"])
        
        # 推断满意度
        satisfaction_inference = {}
        
        for qa_id, durations in qa_durations.items():
            avg_duration = sum(durations) / len(durations)
            
            # 根据停留时间推断
            if avg_duration > 30:
                satisfaction = "high"  # 长时间停留 -> 详细阅读 -> 满意
            elif avg_duration > 10:
                satisfaction = "medium"
            else:
                satisfaction = "low"   # 短时间停留 -> 快速离开 -> 不满意
            
            satisfaction_inference[qa_id] = {
                "avg_duration": avg_duration,
                "satisfaction": satisfaction,
            }
        
        return {
            "total_records": len(self.time_records),
            "average_duration": sum(durations) / len(durations),
            "satisfaction_inference": satisfaction_inference,
        }
```

## 33.8 反馈报告生成

### 自动报告生成

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta

class FeedbackReportGenerator:
    """反馈报告生成器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
    
    def generate_daily_report(self, analysis: dict) -> str:
        """生成每日报告"""
        prompt = ChatPromptTemplate.from_template("""
        基于以下反馈数据，生成每日反馈报告：
        
        分析数据：{analysis}
        报告日期：{date}
        
        报告格式：
        ## 每日反馈报告
        
        ### 总体概况
        - 今日反馈总数
        - 平均评分
        - 准确率
        
        ### 主要发现
        - 高评分问答类型
        - 问题问答类型
        
        ### 用户声音
        - 代表性反馈摘录
        
        ### 改进建议
        - 今日需处理的问题
        
        ### 明日重点关注
        """)
        
        return (prompt | self.llm).invoke({
            "analysis": str(analysis),
            "date": datetime.now().strftime("%Y-%m-%d"),
        }).content
    
    def generate_weekly_report(self, analysis: dict, trend: dict) -> str:
        """生成每周报告"""
        prompt = ChatPromptTemplate.from_template("""
        基于以下数据，生成每周反馈报告：
        
        本周分析：{analysis}
        周趋势：{trend}
        
        报告格式：
        ## 每周反馈报告
        
        ### 本周概况
        - 反馈总数
        - 平均评分变化趋势
        - 准确率变化
        
        ### 突出亮点
        - 评分最高的问答
        
        ### 待解决问题
        - 重复出现的问题
        
        ### 优化进展
        - 本周改进措施
        
        ### 下周计划
        """)
        
        return (prompt | self.llm).invoke({
            "analysis": str(analysis),
            "trend": str(trend),
        }).content
    
    def generate_action_items(self, problems: List[str]) -> List[dict]:
        """生成行动项"""
        prompt = ChatPromptTemplate.from_template("""
        基于以下问题列表，生成具体行动项：
        
        问题：{problems}
        
        请生成：
        1. 紧急行动项（需立即处理）
        2. 本周行动项
        3. 持续优化项
        
        输出 JSON 格式列表。
        """)
        
        return (prompt | self.llm).invoke({
            "problems": str(problems),
        }).content


# 使用示例
report_generator = FeedbackReportGenerator()

# 生成每日报告
analysis = analyzer.get_overall_summary()
daily_report = report_generator.generate_daily_report(analysis)
print(daily_report)

# 生成行动项
problems = analyzer.identify_problem_qas()
action_items = report_generator.generate_action_items(problems)
print(action_items)
```

## 33.9 完整反馈系统

```python
class CompleteFeedbackSystem:
    """完整反馈系统"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.collector = FeedbackCollector()
        self.db = FeedbackDatabase(db_path)
        self.analyzer = FeedbackAnalyzer(self.collector.feedback_store)
        self.llm_analyzer = LLMAssistedFeedbackAnalyzer()
        self.optimizer = FeedbackDrivenOptimizer(self.collector, self.analyzer)
        self.report_generator = FeedbackReportGenerator()
        self.behavior_collector = BehaviorFeedbackCollector()
    
    def record_qa(
        self,
        question: str,
        answer: str,
        user_id: str,
        session_id: str,
        **kwargs,
    ) -> str:
        """记录问答"""
        qa_id = self.collector.record_qa(
            question=question,
            answer=answer,
            user_id=user_id,
            session_id=session_id,
            **kwargs,
        )
        
        # 同步到数据库
        qa_record = self.collector.get_qa_record(qa_id)
        self.db.save_qa_record(qa_record)
        
        return qa_id
    
    def collect_feedback(
        self,
        qa_id: str,
        user_id: str,
        feedback_type: str,
        **kwargs,
    ) -> str:
        """收集反馈"""
        feedback_id = None
        
        if feedback_type == "rating":
            feedback_id = self.collector.collect_rating(
                qa_id, user_id, kwargs.get("rating", 3)
            )
        elif feedback_type == "correctness":
            feedback_id = self.collector.collect_correctness(
                qa_id, user_id,
                kwargs.get("is_correct", True),
                kwargs.get("correct_answer"),
            )
        elif feedback_type == "text":
            feedback_id = self.collector.collect_text(
                qa_id, user_id,
                kwargs.get("comment"),
                kwargs.get("category"),
            )
        
        # 同步到数据库
        if feedback_id:
            feedback = self.collector.feedback_store[feedback_id]
            self.db.save_feedback(feedback)
        
        return feedback_id
    
    def record_behavior(
        self,
        qa_id: str,
        user_id: str,
        action_type: str,
    ) -> str:
        """记录行为反馈"""
        return self.behavior_collector.record_user_action(
            qa_id, user_id, action_type
        )
    
    def get_analysis(self) -> dict:
        """获取分析结果"""
        return {
            "explicit_feedback": self.analyzer.get_overall_summary(),
            "implicit_feedback": self.behavior_collector.analyze_behaviors(),
            "problems": self.analyzer.identify_problem_qas(),
            "trend": self.analyzer.get_trend(7),
        }
    
    def run_optimization(self) -> dict:
        """运行优化"""
        return self.optimizer.run_optimization_cycle()
    
    def generate_report(self, report_type: str = "daily") -> str:
        """生成报告"""
        analysis = self.get_analysis()
        
        if report_type == "daily":
            return self.report_generator.generate_daily_report(analysis)
        elif report_type == "weekly":
            return self.report_generator.generate_weekly_report(
                analysis["explicit_feedback"],
                analysis["trend"],
            )
        
        return ""
    
    def get_dashboard_data(self) -> dict:
        """获取仪表盘数据"""
        return {
            "total_feedbacks": len(self.collector.feedback_store),
            "analysis": self.get_analysis(),
            "recent_feedbacks": self.db.get_all_feedbacks()[-10:],
        }


# 使用示例
system = CompleteFeedbackSystem()

# 记录问答
qa_id = system.record_qa(
    question="产品价格？",
    answer="99元",
    user_id="user_123",
    session_id="session_1",
)

# 收集反馈
system.collect_feedback(qa_id, "user_123", "rating", rating=4)
system.collect_feedback(qa_id, "user_123", "text", comment="简洁明了")

# 记录行为
system.record_behavior(qa_id, "user_123", "copy_answer")

# 获取分析
print(system.get_analysis())

# 运行优化
print(system.run_optimization())

# 生成报告
print(system.generate_report("daily"))
```

## 33.10 本章小结

- 反馈类型：评分、正确性、有用性、文字、对比、行为
- 数据模型：Feedback 基类、QARecord 记录
- 收集系统：多渠道收集、Web API
- 分析系统：统计分析、LLM 辅助分析、趋势分析
- 优化驱动：Prompt 优化、参数调整、A/B 测试
- 数据持久化：SQLite、Redis 存储
- 行为反馈：隐性反馈、时间停留分析
- 报告生成：每日报告、每周报告、行动项
- 完整系统：集成收集、分析、优化、报告全流程
- 价值：持续改进、质量提升、用户满意