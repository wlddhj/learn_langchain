"""
第5章 Demo 2：RunnablePassthrough、RunnableLambda、RunnableParallel

演示数据透传、自定义函数包装、并行执行多个 chain。
可独立运行。
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GLM_API_KEY") or os.environ["GLM_API_KEY"].startswith("your-"):
    print("错误: 未设置 GLM_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def demo_passthrough():
    """RunnablePassthrough：透传数据"""
    print("=" * 50)
    print("Demo 5-2 (1/3): RunnablePassthrough")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_template("用一句话回答：{question}")
    answer_chain = prompt | llm | StrOutputParser()

    # 同时保留原始问题和 LLM 回答
    chain = RunnableParallel(
        question=RunnablePassthrough(),     # 原样透传输入
        answer=answer_chain,                 # LLM 生成回答
    )

    result = chain.invoke("什么是 Python？")
    print(f"原始问题: {result['question']}")
    print(f"LLM 回答: {result['answer'][:150]}")
    print()

    # RunnablePassthrough.assign：追加计算字段
    chain_with_extra = RunnablePassthrough.assign(
        word_count=lambda x: len(x["answer"].split()),
        char_count=lambda x: len(x["answer"]),
    )

    enriched = chain_with_extra.invoke(result)
    print(f"追加统计: 词数={enriched['word_count']}, 字符数={enriched['char_count']}")
    print()


def demo_lambda():
    """RunnableLambda：包装自定义函数"""
    print("=" * 50)
    print("Demo 5-2 (2/3): RunnableLambda")
    print("=" * 50)

    def parse_upper(text: str) -> str:
        """将结果转大写"""
        return text.upper()

    def add_prefix(text: str) -> str:
        """添加前缀"""
        return f"[AI回答] {text}"

    prompt = ChatPromptTemplate.from_template("用一句话解释{concept}")
    chain = (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(add_prefix)
        | RunnableLambda(parse_upper)
    )

    result = chain.invoke({"concept": "API"})
    print(f"结果 (大写+前缀): {result[:150]}")
    print()

    # 实用场景：统计输出
    count_chain = (
        ChatPromptTemplate.from_template("列举5个{category}的名字")
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda text: {
            "text": text,
            "line_count": len(text.strip().split("\n")),
            "char_count": len(text),
        })
    )

    result = count_chain.invoke({"category": "编程语言"})
    print(f"统计结果:")
    print(f"  行数: {result['line_count']}")
    print(f"  字符数: {result['char_count']}")
    print(f"  内容: {result['text'][:100]}...")
    print()


def demo_parallel():
    """RunnableParallel：并行执行多个 chain"""
    print("=" * 50)
    print("Demo 5-2 (3/3): RunnableParallel 并行执行")
    print("=" * 50)

    text = "人工智能正在深刻改变我们的生活和工作方式，从自动驾驶到智能医疗，AI 技术的应用越来越广泛。"

    # 定义三个并行的分析任务
    summary_prompt = ChatPromptTemplate.from_template("用一句话总结：{text}")
    keywords_prompt = ChatPromptTemplate.from_template("提取3个关键词，用逗号分隔：{text}")
    sentiment_prompt = ChatPromptTemplate.from_template("判断以下文本的情感（积极/消极/中性），只回复一个词：{text}")

    summary_chain = summary_prompt | llm | StrOutputParser()
    keywords_chain = keywords_prompt | llm | StrOutputParser()
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()

    # 串行执行（对比用）
    print("--- 串行执行（对比）---")
    start = time.time()
    s1 = summary_chain.invoke({"text": text})
    k1 = keywords_chain.invoke({"text": text})
    st1 = sentiment_chain.invoke({"text": text})
    serial_time = time.time() - start
    print(f"  耗时: {serial_time:.2f}s")
    print()

    # 并行执行
    print("--- 并行执行 (RunnableParallel) ---")
    parallel = RunnableParallel(
        summary=summary_chain,
        keywords=keywords_chain,
        sentiment=sentiment_chain,
    )

    start = time.time()
    result = parallel.invoke({"text": text})
    parallel_time = time.time() - start

    print(f"  摘要: {result['summary']}")
    print(f"  关键词: {result['keywords']}")
    print(f"  情感: {result['sentiment']}")
    print()
    print(f"  串行耗时: {serial_time:.2f}s | 并行耗时: {parallel_time:.2f}s")
    print(f"  加速比: {serial_time / parallel_time:.1f}x")
    print()

    # 字典语法是 RunnableParallel 的简写
    print("提示: 字典语法 {key: runnable} 是 RunnableParallel 的简写形式")
    print()


if __name__ == "__main__":
    demo_passthrough()
    demo_lambda()
    demo_parallel()

    print("=" * 50)
    print("Demo 5-2 完成!")
    print()
    print("Runnable 工具对比:")
    print("  RunnablePassthrough  - 透传数据，不修改")
    print("  RunnableLambda       - 包装自定义函数")
    print("  RunnableParallel     - 并行执行多个 chain")
    print("=" * 50)
