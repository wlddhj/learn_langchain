"""
第2章 Demo 3：结构化输出 (Structured Output)

演示如何让 LLM 返回 Pydantic 模型、JSON Schema 等结构化数据。
可独立运行，需要 GLM_API_KEY。
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GLM_API_KEY") or os.environ["GLM_API_KEY"].startswith("your-"):
    print("错误: 未设置 GLM_API_KEY")
    sys.exit(1)

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(
    model=GLM_MODEL,
    temperature=0,
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL,
)


# ============================================================
# 示例 1：简单的结构化输出
# ============================================================

class CityInfo(BaseModel):
    """城市信息"""
    name: str = Field(description="城市名称")
    country: str = Field(description="所属国家")
    population: str = Field(description="大致人口")
    famous_for: str = Field(description="以什么闻名")


def demo_simple_structured():
    """简单的结构化输出"""
    print("=" * 50)
    print(f"Demo 2-3 (1/3): 简单结构化输出 [{GLM_MODEL}]")
    print("=" * 50)

    structured_llm = llm.with_structured_output(CityInfo)

    result = structured_llm.invoke("介绍一下巴黎")
    print(f"类型: {type(result).__name__}")
    print(f"城市: {result.name}")
    print(f"国家: {result.country}")
    print(f"人口: {result.population}")
    print(f"闻名于: {result.famous_for}")
    print()

    # 可以直接转换为 dict / JSON
    print(f"转为字典: {result.model_dump()}")
    print()


# ============================================================
# 示例 2：嵌套结构 + 列表
# ============================================================

class BookRecommendation(BaseModel):
    """单本书的推荐"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    reason: str = Field(description="推荐理由（一句话）")


class ReadingList(BaseModel):
    """阅读书单"""
    topic: str = Field(description="书单主题")
    difficulty: str = Field(description="难度等级: 入门/中级/高级")
    books: list[BookRecommendation] = Field(description="推荐书籍列表")
    summary: str = Field(description="一句话总结这个书单")


def demo_nested_structured():
    """嵌套结构的结构化输出"""
    print("=" * 50)
    print(f"Demo 2-3 (2/3): 嵌套结构 + 列表 [{GLM_MODEL}]")
    print("=" * 50)

    structured_llm = llm.with_structured_output(ReadingList)

    result = structured_llm.invoke("推荐一份 Python 入门学习的书单，包含 3 本书")
    print(f"主题: {result.topic}")
    print(f"难度: {result.difficulty}")
    print(f"总结: {result.summary}")
    print()
    print("推荐书单:")
    for i, book in enumerate(result.books, 1):
        print(f"  {i}. 《{book.title}》 - {book.author}")
        print(f"     推荐理由: {book.reason}")
    print()


# ============================================================
# 示例 3：实用场景 —— 情感分析（使用 PydanticOutputParser）
# ============================================================

class SentimentAnalysis(BaseModel):
    """情感分析结果"""
    sentiment: str = Field(description="情感倾向: 积极/消极/中性")
    confidence: float = Field(description="置信度 0.0-1.0")
    keywords: list[str] = Field(description="关键词列表")
    reason: str = Field(description="判断理由")


def demo_sentiment_analysis():
    """实用场景：批量情感分析"""
    print("=" * 50)
    print(f"Demo 2-3 (3/3): 实用场景 - 情感分析 [{GLM_MODEL}]")
    print("=" * 50)

    parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

    texts = [
        "这个产品真的太棒了！使用体验非常好，强烈推荐。",
        "质量很差，用了一周就坏了，非常失望。",
        "今天收到了快递，包装还行，还没拆开用。",
    ]

    for text in texts:
        prompt = f"""分析以下文本的情感，按 JSON 格式返回。

文本: {text}

{parser.get_format_instructions()}"""

        response = llm.invoke(prompt)

        try:
            result = parser.parse(response.content)
            emoji = {"积极": "😊", "消极": "😞", "中性": "😐"}.get(result.sentiment, "❓")

            print(f"文本: {text}")
            print(f"情感: {emoji} {result.sentiment} (置信度: {result.confidence:.0%})")
            print(f"关键词: {', '.join(result.keywords)}")
            print(f"理由: {result.reason}")
        except Exception as e:
            print(f"文本: {text}")
            print(f"解析失败: {e}")
            print(f"原始输出: {response.content[:200]}")
        print()


if __name__ == "__main__":
    demo_simple_structured()
    demo_nested_structured()
    demo_sentiment_analysis()

    print("=" * 50)
    print("Demo 2-3 完成!")
    print()
    print("结构化输出要点:")
    print("  1. 用 Pydantic BaseModel 定义输出 schema")
    print("  2. Field(description=...) 帮助 LLM 理解每个字段")
    print("  3. with_structured_output() 包装 LLM")
    print("  4. 支持嵌套模型、列表等复杂结构")
    print("=" * 50)
