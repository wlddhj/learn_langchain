"""
第4章 Demo 2：PydanticOutputParser、自定义解析器、with_structured_output 对比

演示 PydanticOutputParser 与 with_structured_output 的区别，以及自定义解析器。
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

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 共享的 Pydantic 模型
# ============================================================

class BookReview(BaseModel):
    """书评结构"""
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    rating: float = Field(description="评分 1.0-5.0")
    summary: str = Field(description="一句话评价")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")


def demo_pydantic_parser():
    """PydanticOutputParser：返回带类型的 Pydantic 对象"""
    print("=" * 50)
    print("Demo 4-2 (1/3): PydanticOutputParser")
    print("=" * 50)

    parser = PydanticOutputParser(pydantic_object=BookReview)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个书评家。\n{format_instructions}"),
        ("human", "评价一下《{book}》这本书"),
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "book": "三体",
        "format_instructions": parser.get_format_instructions(),
    })

    print(f"类型: {type(result).__name__}")
    print(f"书名: {result.title}")
    print(f"作者: {result.author}")
    print(f"评分: {result.rating}")
    print(f"评价: {result.summary}")
    print(f"优点: {', '.join(result.pros)}")
    print(f"缺点: {', '.join(result.cons)}")
    print()
    print("优势: 返回 Pydantic 对象，可点号访问属性，有类型校验")
    print()


def demo_structured_output():
    """with_structured_output：推荐的结构化输出方式"""
    print("=" * 50)
    print("Demo 4-2 (2/3): with_structured_output (推荐)")
    print("=" * 50)

    structured_llm = llm.with_structured_output(BookReview)

    result = structured_llm.invoke("评价一下《活着》这本书")

    print(f"类型: {type(result).__name__}")
    print(f"书名: {result.title}")
    print(f"评分: {result.rating}")
    print(f"评价: {result.summary}")
    print(f"优点: {', '.join(result.pros)}")
    print(f"缺点: {', '.join(result.cons)}")
    print()
    print("优势: 无需 format_instructions，更简洁可靠")
    print()


def demo_custom_parser():
    """自定义解析器：将输出解析为布尔值"""
    print("=" * 50)
    print("Demo 4-2 (3/3): 自定义 BooleanOutputParser")
    print("=" * 50)

    from langchain_core.output_parsers import BaseOutputParser
    from typing import ClassVar

    class BooleanOutputParser(BaseOutputParser[bool]):
        """将 LLM 输出解析为布尔值"""
        true_values: ClassVar[set] = {"是", "yes", "true", "对", "正确", "1", "有的", "有"}
        false_values: ClassVar[set] = {"否", "no", "false", "错", "错误", "0", "没有", "没", "不是"}

        def parse(self, text: str) -> bool:
            cleaned = text.strip().lower()
            # 只取第一行/第一个词
            first_word = cleaned.split("\n")[0].split("，")[0].split(",")[0].strip()
            if first_word in self.true_values:
                return True
            if first_word in self.false_values:
                return False
            # 模糊匹配
            for tv in self.true_values:
                if tv in cleaned:
                    return True
            for fv in self.false_values:
                if fv in cleaned:
                    return False
            raise ValueError(f"无法将 '{text[:50]}' 解析为布尔值")

        @property
        def _type(self) -> str:
            return "boolean_output_parser"

    bool_parser = BooleanOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "请只用'是'或'否'回答以下问题，不要解释。"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    questions = [
        "Python 是一种编译型语言吗？",
        "地球是太阳系中最大的行星吗？",
        "HTTP 是无状态协议吗？",
    ]

    for q in questions:
        raw = chain.invoke({"question": q})
        try:
            parsed = bool_parser.parse(raw)
            print(f"问题: {q}")
            print(f"  原始回复: {raw.strip()[:50]}")
            print(f"  解析结果: {parsed}")
        except ValueError as e:
            print(f"问题: {q}")
            print(f"  解析失败: {e}")
        print()


if __name__ == "__main__":
    demo_pydantic_parser()
    demo_structured_output()
    demo_custom_parser()

    print("=" * 50)
    print("Demo 4-2 完成!")
    print()
    print("结构化输出方式对比:")
    print("  PydanticOutputParser - 需 format_instructions，返回 Pydantic 对象")
    print("  with_structured_output - 推荐！更简洁，利用 LLM 原生能力")
    print("  自定义 BaseOutputParser - 满足特殊解析需求")
    print("=" * 50)
