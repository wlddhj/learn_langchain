"""
第4章 Demo 1：StrOutputParser、CommaSeparatedListOutputParser、JsonOutputParser

演示三种内置输出解析器的使用方式和适用场景。
可独立运行。
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def demo_str_output_parser():
    """StrOutputParser：提取纯字符串"""
    print("=" * 50)
    print("Demo 4-1 (1/3): StrOutputParser")
    print("=" * 50)

    # 不用 parser
    prompt = ChatPromptTemplate.from_template("用一句话解释{concept}")
    chain_no_parser = prompt | llm

    result_raw = chain_no_parser.invoke({"concept": "递归"})
    print(f"不用 parser → 类型: {type(result_raw).__name__}")
    print(f"  内容: {result_raw.content[:100]}")
    print()

    # 用 StrOutputParser
    chain_with_parser = prompt | llm | StrOutputParser()

    result_str = chain_with_parser.invoke({"concept": "递归"})
    print(f"用 StrOutputParser → 类型: {type(result_str).__name__}")
    print(f"  内容: {result_str[:100]}")
    print()
    print("结论: StrOutputParser 把 AIMessage 转为纯字符串，方便后续处理")
    print()


def demo_list_output_parser():
    """CommaSeparatedListOutputParser：解析逗号分隔列表"""
    print("=" * 50)
    print("Demo 4-1 (2/3): CommaSeparatedListOutputParser")
    print("=" * 50)

    parser = CommaSeparatedListOutputParser()
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个命名专家。\n{format_instructions}"),
        ("human", "给我5个{language}项目的名字"),
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "language": "Python",
        "format_instructions": format_instructions,
    })

    print(f"类型: {type(result).__name__}")
    print(f"结果: {result}")
    print(f"数量: {len(result)}")
    print()

    # 另一个例子
    result2 = chain.invoke({
        "language": "Rust",
        "format_instructions": format_instructions,
    })
    print(f"Rust 项目名: {result2}")
    print()


def demo_json_output_parser():
    """JsonOutputParser：解析 JSON 输出"""
    print("=" * 50)
    print("Demo 4-1 (3/3): JsonOutputParser")
    print("=" * 50)

    class Recipe(BaseModel):
        """菜谱"""
        name: str = Field(description="菜名")
        ingredients: list[str] = Field(description="食材列表")
        steps: list[str] = Field(description="烹饪步骤，不超过5步")
        cook_time: str = Field(description="烹饪时间")

    parser = JsonOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个厨艺专家。\n{format_instructions}"),
        ("human", "教我做{dish}"),
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "dish": "番茄炒蛋",
        "format_instructions": parser.get_format_instructions(),
    })

    print(f"类型: {type(result).__name__}")
    print(f"菜名: {result['name']}")
    print(f"食材: {', '.join(result['ingredients'])}")
    print(f"耗时: {result['cook_time']}")
    print(f"步骤:")
    for i, step in enumerate(result["steps"], 1):
        print(f"  {i}. {step}")
    print()

    # 格式化 JSON 输出
    print(f"完整 JSON:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()


if __name__ == "__main__":
    demo_str_output_parser()
    demo_list_output_parser()
    demo_json_output_parser()

    print("=" * 50)
    print("Demo 4-1 完成!")
    print()
    print("解析器对比:")
    print("  StrOutputParser              - AIMessage → str")
    print("  CommaSeparatedListOutputParser - 文本 → list[str]")
    print("  JsonOutputParser              - 文本 → dict (基于 Pydantic schema)")
    print("=" * 50)
