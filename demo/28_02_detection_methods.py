"""
第28章 Demo 2：幻觉检测方法

演示自我检测法、事实核对法、RAG源引用检测等方法。
需要 QWEN_API_KEY。
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

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def demo_self_detection():
    """自我检测法：让模型检查生成内容的准确性"""
    print("=" * 60)
    print(f"Demo 28-2 (1/4): 自我检测法 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 生成回答
    question = "李白出生于哪一年？"

    generate_prompt = ChatPromptTemplate.from_template("""
请回答以下问题，提供具体年份：
{question}
""")

    answer = (generate_prompt | llm | StrOutputParser()).invoke({"question": question})
    print(f"问题: {question}")
    print(f"回答: {answer}")
    print()

    # 自我检测
    verify_prompt = ChatPromptTemplate.from_template("""
请检查以下回答中是否存在幻觉（不正确或虚构的信息）：

问题：{question}
回答：{answer}

请检测：
1. 回答中的具体陈述是否正确？
2. 是否存在编造的信息？
3. 给出整体可信度评分（高/中/低）

如果有错误，请指出正确信息。
""")

    verification = (verify_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "answer": answer,
    })

    print("检测结果:")
    print(verification)
    print()


def demo_fact_extraction():
    """从回答中提取关键事实"""
    print("=" * 60)
    print(f"Demo 28-2 (2/4): 关键事实提取 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    answer = """
李白（701年－762年），字太白，号青莲居士，是中国唐代伟大的浪漫主义诗人。
他出生于四川绵州（今江油市），被誉为"诗仙"。
代表作包括《静夜思》、《将进酒》、《望庐山瀑布》等。
"""

    extract_prompt = ChatPromptTemplate.from_template("""
从以下回答中提取需要验证的关键事实陈述（如日期、地点、人物关系等）：

回答：{answer}

请列出所有需要验证的事实，每行一个，包括：
- 日期年份
- 地点名称
- 人物称号
- 作品名称
- 其他具体陈述

只输出事实列表，不要其他内容。
""")

    facts = (extract_prompt | llm | StrOutputParser()).invoke({"answer": answer})

    print("原始回答:")
    print(answer)
    print()
    print("提取的关键事实:")
    print(facts)
    print()


def demo_rag_source_check():
    """RAG 源引用检测模拟"""
    print("=" * 60)
    print(f"Demo 28-2 (3/4): RAG 源引用检测 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 模拟检索内容
    retrieved_context = """
[来源 1] 李白（701年－762年），唐代诗人，被誉为"诗仙"。
[来源 2] 李白出生于绵州昌隆县（今四川省江油市）。
[来源 3] 李白代表作有《静夜思》、《将进酒》等。
"""

    # 模拟回答
    answer = """
李白出生于公元 701 年，是唐代伟大的诗人。
他出生于四川成都，被誉为"诗仙"。
他的代表作包括《静夜思》、《将进酒》和《红楼梦》。
"""

    check_prompt = ChatPromptTemplate.from_template("""
检查以下回答是否正确使用了检索内容：

检索内容：
{context}

回答：{answer}

请检查：
1. 回答中的信息是否都来自检索内容？
2. 是否有添加检索内容之外的"新"信息？
3. 是否有与检索内容矛盾的信息？

请指出：
- 正确引用的内容
- 额外添加的内容（可能幻觉）
- 与检索矛盾的内容

输出检测结果。
""")

    result = (check_prompt | llm | StrOutputParser()).invoke({
        "context": retrieved_context,
        "answer": answer,
    })

    print("检索内容:")
    print(retrieved_context)
    print()
    print("回答:")
    print(answer)
    print()
    print("源引用检测结果:")
    print(result)
    print()


def demo_confidence_assessment():
    """置信度评估"""
    print("=" * 60)
    print(f"Demo 28-2 (4/4): 答案置信度评估 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    question = "李白的具体出生日期是哪一天？"

    # 生成回答
    answer = "李白出生于公元 701 年 2 月 28 日，在四川绵州。"

    assess_prompt = ChatPromptTemplate.from_template("""
评估以下答案的可信度：

问题：{question}
答案：{answer}

评估维度：
1. 信息来源可靠性（历史记载是否支持如此精确的日期？）
2. 过度精确风险（是否给出了无法验证的精确细节？）
3. 整体可信度评分（高/中/低）

请给出评估结果和评分。
""")

    assessment = (assess_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "answer": answer,
    })

    print(f"问题: {question}")
    print(f"答案: {answer}")
    print()
    print("置信度评估:")
    print(assessment)
    print()

    print("分析提示:")
    print("  - 李白的具体出生日期在历史上并无确切记载")
    print("  - '2月28日'这样的精确日期可能是幻觉")
    print("  - 高风险：过度精确的细节")


if __name__ == "__main__":
    demo_self_detection()
    demo_fact_extraction()
    demo_rag_source_check()
    demo_confidence_assessment()

    print("=" * 60)
    print("Demo 28-2 完成!")
    print()
    print("幻觉检测方法总结：")
    print("  1. 自我检测法：让模型检查自己的回答")
    print("  2. 事实提取法：从回答中提取关键事实进行验证")
    print("  3. RAG源引用检测：检查是否基于检索内容回答")
    print("  4. 置信度评估：评估回答的可信度")
    print()
    print("最佳实践：")
    print("  - 组合使用多种检测方法")
    print("  - 对高风险内容进行人工复核")
    print("  - 建立持续监控机制")
    print("=" * 60)