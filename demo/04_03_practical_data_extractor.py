"""
第4章 Demo 3：实战 —— 多格式信息提取器

综合运用多种输出解析方式，从非结构化文本中提取结构化数据。
可独立运行。
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 数据模型
# ============================================================

class PersonInfo(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: Optional[int] = Field(default=None, description="年龄")
    occupation: Optional[str] = Field(default=None, description="职业")
    location: Optional[str] = Field(default=None, description="所在地")


class EventInfo(BaseModel):
    """事件信息"""
    title: str = Field(description="事件标题")
    date: Optional[str] = Field(default=None, description="发生日期")
    location: Optional[str] = Field(default=None, description="地点")
    description: str = Field(description="事件描述")


class ArticleAnalysis(BaseModel):
    """文章分析结果"""
    topic: str = Field(description="主题")
    summary: str = Field(description="一句话摘要")
    persons: list[PersonInfo] = Field(description="提到的人物")
    events: list[EventInfo] = Field(description="提到的事件")
    keywords: list[str] = Field(description="关键词")


# ============================================================
# Demo 函数
# ============================================================

def demo_extract_persons():
    """提取人物信息"""
    print("=" * 50)
    print("Demo 4-3 (1/3): 人物信息提取")
    print("=" * 50)

    structured_llm = llm.with_structured_output(PersonInfo)
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "从用户给出的文本中提取人物信息。\n{format_instructions}"),
    #     ("human", "{text}"),
    # ])
    #
    # parser = JsonOutputParser(pydantic_object=PersonInfo)
    # chain = prompt | llm | parser

    texts = [
        "张三今年28岁，是一名软件工程师，目前在北京工作。",
        "李华是清华大学的教授，专注于人工智能研究。",
        "王五刚从上海搬到深圳，成为了一名自由职业者。",
    ]

    for text in texts:
        result = structured_llm.invoke(f"从以下文本中提取人物信息：\n{text}")
        # result = chain.invoke({
        #     "text": text,
        #     "format_instructions": parser.get_format_instructions(),
        # })
        print(result)
        print(f"原文: {text}")
        print(f"  姓名: {result.name}")
        print(f"  年龄: {result.age}")
        print(f"  职业: {result.occupation}")
        print(f"  所在地: {result.location}")
        print()


def demo_extract_events():
    """提取事件信息"""
    print("=" * 50)
    print("Demo 4-3 (2/3): 事件信息提取")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "从用户给出的文本中提取事件信息。\n{format_instructions}"),
        ("human", "{text}"),
    ])

    parser = JsonOutputParser(pydantic_object=EventInfo)
    chain = prompt | llm | parser

    texts = [
        "2024年7月，嫦娥六号成功从月球背面带回土壤样品，这是人类历史上的首次。",
        "昨天下午，杭州亚运会开幕式在奥体中心隆重举行，数万名观众到场观看。",
    ]

    for text in texts:
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions(),
        })
        print(f"原文: {text}")
        print(f"  事件: {result['title']}")
        print(f"  日期: {result.get('date', '未知')}")
        print(f"  地点: {result.get('location', '未知')}")
        print(f"  描述: {result['description'][:100]}")
        print()


def demo_article_analysis():
    """综合文章分析：提取主题、人物、事件、关键词"""
    print("=" * 50)
    print("Demo 4-3 (3/3): 综合文章分析")
    print("=" * 50)

    structured_llm = llm.with_structured_output(ArticleAnalysis)

    article = """\
2024年3月，OpenAI 发布了 GPT-4 Turbo，引发了 AI 行业的广泛关注。
CEO Sam Altman 在发布会上表示，这是目前最强大的语言模型。
与此同时，Google 的 DeepMind 团队也在伦敦发布了 Gemini 1.5 Pro，
由负责人 Demis Hassabis 亲自演示了其百万级上下文窗口能力。
这两场发布会几乎同时进行，被视为 AI 领域的一次正面对决。
"""

    result = structured_llm.invoke(f"分析以下文章：\n\n{article}")

    print(f"主题: {result.topic}")
    print(f"摘要: {result.summary}")
    print(f"关键词: {', '.join(result.keywords)}")
    print()

    print("提到的人物:")
    for p in result.persons:
        print(f"  - {p.name}" + (f" ({p.occupation})" if p.occupation else ""))
    print()

    print("提到的事件:")
    for e in result.events:
        print(f"  - {e.title}")
        if e.date:
            print(f"    日期: {e.date}")
    print()

    # 输出完整 JSON
    print("完整结构化数据:")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    demo_extract_persons()
    demo_extract_events()
    demo_article_analysis()

    print("=" * 50)
    print("Demo 4-3 完成!")
    print()
    print("信息提取要点:")
    print("  1. 用 Pydantic 定义清晰的数据结构")
    print("  2. with_structured_output 或 JsonOutputParser 获取结构化数据")
    print("  3. Optional 字段处理可能缺失的信息")
    print("  4. 嵌套模型处理复杂结构（人物、事件、关键词）")
    print("=" * 50)
