"""
第5章 Demo 3：条件路由、Fallback 与重试

演示 RunnableBranch 条件路由、with_fallbacks 失败回退、with_retry 自动重试。
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def demo_branch():
    """RunnableBranch：条件路由到不同 chain"""
    print("=" * 50)
    print("Demo 5-3 (1/3): 条件路由")
    print("=" * 50)

    # 不同领域的 prompt
    math_prompt = ChatPromptTemplate.from_template(
        "你是数学老师，用清晰的步骤解答：{question}"
    )
    code_prompt = ChatPromptTemplate.from_template(
        "你是编程专家，给出代码示例和解释：{question}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "你是一个有帮助的助手，简洁回答：{question}"
    )

    math_chain = math_prompt | llm | StrOutputParser()
    code_chain = code_prompt | llm | StrOutputParser()
    general_chain = general_prompt | llm | StrOutputParser()

    # 用 LLM 先分类，再路由
    classify_prompt = ChatPromptTemplate.from_template(
        "将以下问题分类为一个词（math/code/general）：{question}"
    )
    classify_chain = classify_prompt | llm | StrOutputParser()

    def route_question(question: str):
        category = classify_chain.invoke({"question": question}).strip().lower()
        print(f"  分类结果: {category}")

        if "math" in category:
            return math_chain.invoke({"question": question})
        elif "code" in category:
            return code_chain.invoke({"question": question})
        else:
            return general_chain.invoke({"question": question})

    # 测试不同类型的问题
    questions = [
        "什么是斐波那契数列？",
        "Python 如何读取文件？",
        "今天的日期是什么格式？",
    ]

    for q in questions:
        print(f"\n问题: {q}")
        answer = route_question(q)
        print(f"回答: {answer[:200]}")

    print()


def demo_fallback():
    """with_fallbacks：失败回退"""
    print("=" * 50)
    print("Demo 5-3 (2/3): Fallback 回退机制")
    print("=" * 50)

    # 主 chain：使用一个不存在的模型（会失败）
    bad_llm = ChatOpenAI(
        model="nonexistent-model",
        api_key=GLM_API_KEY,
        base_url=GLM_BASE_URL,
        max_retries=0,
    )
    primary_chain = (
        ChatPromptTemplate.from_template("回答：{question}")
        | bad_llm
        | StrOutputParser()
    )

    # 备选 chain：使用正常模型
    fallback_chain = (
        ChatPromptTemplate.from_template("回答：{question}")
        | llm
        | StrOutputParser()
    )

    # 主 chain 失败时自动使用备选
    chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

    result = chain_with_fallback.invoke({"question": "什么是 LCEL？"})
    print(f"主模型失败，自动回退到 {GLM_MODEL}")
    print(f"回答: {result[:200]}")
    print()


def demo_retry():
    """with_retry：自动重试"""
    print("=" * 50)
    print("Demo 5-3 (3/3): Retry 重试机制")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_template("用一句话回答：{question}")
    chain = prompt | llm | StrOutputParser()

    # 添加重试机制
    chain_with_retry = chain.with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(Exception,),
    )

    print("测试正常请求（带重试保护）:")
    start = time.time()
    result = chain_with_retry.invoke({"question": "什么是 LCEL？"})
    elapsed = time.time() - start
    print(f"  回答: {result[:150]}")
    print(f"  耗时: {elapsed:.2f}s (首次成功无需重试)")
    print()

    print("with_retry 配置:")
    print("  stop_after_attempt=3  — 最多重试3次")
    print("  retry_if_exception_type — 遇到指定异常时重试")
    print()


if __name__ == "__main__":
    demo_branch()
    demo_fallback()
    demo_retry()

    print("=" * 50)
    print("Demo 5-3 完成!")
    print()
    print("高级特性总结:")
    print("  条件路由  - 先分类，再路由到专门 chain")
    print("  Fallback  - 主 chain 失败时自动切换备选")
    print("  Retry     - 自动重试，提高可靠性")
    print("=" * 50)
