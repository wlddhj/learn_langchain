"""
第5章 Demo 1：LCEL 基础与 Runnable 接口

演示 LCEL 管道操作、invoke/batch/stream 统一接口、数据流。
可独立运行。
"""

import asyncio
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

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def demo_basic_chain():
    """LCEL 基础管道：prompt | llm | parser"""
    print("=" * 50)
    print("Demo 5-1 (1/3): LCEL 基础管道")
    print(f"模型: {GLM_MODEL}")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_template("用一句话解释{concept}")
    parser = StrOutputParser()

    # 用 | 管道连接组件
    chain = prompt | llm | parser

    print("数据流: dict → ChatPromptTemplate → ChatModel → StrOutputParser → str")
    print()

    # invoke
    result = chain.invoke({"concept": "机器学习"})
    print(f"invoke  → 类型: {type(result).__name__}")
    print(f"         结果: {result[:100]}")
    print()

    # batch
    results = chain.batch([
        {"concept": "深度学习"},
        {"concept": "强化学习"},
        {"concept": "迁移学习"},
    ])
    print(f"batch   → 类型: {type(results).__name__}, 数量: {len(results)}")
    for r in results:
        print(f"         - {r[:60]}")
    print()

    # stream
    print("stream  → ")
    print("         ", end="", flush=True)
    for chunk in chain.stream({"concept": "神经网络"}):
        print(chunk, end="", flush=True)
    print("\n")


def demo_runnable_interface():
    """Runnable 接口：统一的方法集"""
    print("=" * 50)
    print("Demo 5-1 (2/3): Runnable 接口")
    print("=" * 50)
    print("所有 LCEL 组件都实现 Runnable 接口，统一支持:")
    print("  invoke / ainvoke  — 单次调用")
    print("  batch / abatch    — 批量调用")
    print("  stream / astream  — 流式输出")
    print()

    prompt = ChatPromptTemplate.from_template(
        "将以下概念用比喻来解释，让10岁小孩也能理解：{concept}"
    )
    chain = prompt | llm | StrOutputParser()

    # invoke
    print("--- invoke (同步) ---")
    start = time.time()
    result = chain.invoke({"concept": "数据库索引"})
    print(f"  耗时: {time.time() - start:.2f}s")
    print(f"  结果: {result[:150]}")
    print()

    # batch
    print("--- batch (批量) ---")
    start = time.time()
    results = chain.batch([
        {"concept": "缓存"},
        {"concept": "负载均衡"},
    ])
    print(f"  耗时: {time.time() - start:.2f}s")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r[:100]}")
    print()

    # ainvoke (异步)
    async def demo_async():
        print("--- ainvoke (异步) ---")
        start = time.time()
        result = await chain.ainvoke({"concept": "消息队列"})
        print(f"  耗时: {time.time() - start:.2f}s")
        print(f"  结果: {result[:150]}")
        print()

    asyncio.run(demo_async())


def demo_chain_with_multiple_steps():
    """多步骤 Chain：逐步变换数据"""
    print("=" * 50)
    print("Demo 5-1 (3/3): 多步骤 Chain")
    print("=" * 50)

    from langchain_core.runnables import RunnableLambda

    # 步骤1：生成解释
    explain_prompt = ChatPromptTemplate.from_template(
        "用简洁的中文解释：{word}"
    )
    explain_chain = explain_prompt | llm | StrOutputParser()

    # 步骤2：生成例句（基于步骤1的结果）
    example_prompt = ChatPromptTemplate.from_template(
        "基于以下解释，给出2个例句：\n\n解释：{explanation}\n\n词语：{word}"
    )
    example_chain = example_prompt | llm | StrOutputParser()

    word = "人工智能"

    # 步骤1
    print(f"词语: {word}")
    explanation = explain_chain.invoke({"word": word})
    print(f"解释: {explanation[:150]}")
    print()

    # 步骤2
    examples = example_chain.invoke({"word": word, "explanation": explanation})
    print(f"例句:\n{examples}")
    print()

    # 用 RunnableLambda 串成一步
    def generate_examples(data):
        return example_chain.invoke({
            "word": data["word"],
            "explanation": data["explanation"],
        })

    full_chain = RunnableLambda(lambda word: {
        "word": word,
        "explanation": explain_chain.invoke({"word": word}),
    }) | RunnableLambda(generate_examples)

    print("合并为一步调用:")
    result = full_chain.invoke("深度学习")
    print(result)
    print()


if __name__ == "__main__":
    demo_basic_chain()
    demo_runnable_interface()
    demo_chain_with_multiple_steps()

    print("=" * 50)
    print("Demo 5-1 完成!")
    print()
    print("LCEL 核心要点:")
    print("  | 管道操作符   - 串联组件")
    print("  invoke/batch/stream - 统一接口")
    print("  RunnableLambda - 包装自定义函数")
    print("=" * 50)
