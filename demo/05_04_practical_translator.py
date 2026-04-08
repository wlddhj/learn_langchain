"""
第5章 Demo 4：实战 —— 多语言翻译与质量评价

综合运用 LCEL 管道、RunnableParallel、RunnablePassthrough、RunnableLambda。
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

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


# ============================================================
# 翻译 + 评价 pipeline
# ============================================================

# 翻译 chain
translate_prompt = ChatPromptTemplate.from_template(
    "将以下文本翻译成{language}，只输出翻译结果：\n\n{text}"
)
translate_chain = translate_prompt | llm | StrOutputParser()

# 质量评价 chain
review_prompt = ChatPromptTemplate.from_template(
    "评价以下翻译的质量（1-10分），一句话说明理由：\n\n原文: {text}\n译文: {translation}"
)
review_chain = review_prompt | llm | StrOutputParser()

# 反向翻译 chain
back_translate_prompt = ChatPromptTemplate.from_template(
    "将以下文本翻译回中文：\n\n{text}"
)
back_translate_chain = back_translate_prompt | llm | StrOutputParser()


def demo_single_translate():
    """单语言翻译 + 质量评价"""
    print("=" * 50)
    print("Demo 5-4 (1/3): 翻译 + 质量评价")
    print("=" * 50)

    text = "人工智能正在深刻改变我们生活和工作的方式"
    language = "English"

    # 完整 pipeline
    chain = (
        # 步骤1：翻译，同时保留原始输入
        {
            "text": lambda x: x["text"],
            "language": lambda x: x["language"],
            "translation": translate_chain,
        }
        # 步骤2：评价翻译质量
        | RunnablePassthrough.assign(
            review=lambda x: review_chain.invoke({
                "text": x["text"],
                "translation": x["translation"],
            })
        )
    )

    result = chain.invoke({"text": text, "language": language})

    print(f"原文: {result['text']}")
    print(f"目标语言: {result['language']}")
    print(f"译文: {result['translation']}")
    print(f"评价: {result['review']}")
    print()


def demo_multi_translate():
    """多语言并行翻译"""
    print("=" * 50)
    print("Demo 5-4 (2/3): 多语言并行翻译")
    print("=" * 50)

    text = "学习使人进步，实践出真知"

    # 并行翻译到多个语言
    parallel_translate = RunnableParallel(
        english=lambda _: translate_chain.invoke({"text": text, "language": "English"}),
        japanese=lambda _: translate_chain.invoke({"text": text, "language": "Japanese"}),
        korean=lambda _: translate_chain.invoke({"text": text, "language": "Korean"}),
    )

    print(f"原文: {text}")
    print()

    start = time.time()
    results = parallel_translate.invoke({})
    elapsed = time.time() - start

    print(f"English: {results['english']}")
    print(f"Japanese: {results['japanese']}")
    print(f"Korean: {results['korean']}")
    print(f"\n3种语言并行翻译耗时: {elapsed:.2f}s")
    print()


def demo_back_translation():
    """反向翻译验证（Round-trip Translation）"""
    print("=" * 50)
    print("Demo 5-3 (3/3): 反向翻译验证")
    print("=" * 50)
    print("原理: 原文 → 译文 → 反向翻译回原文，对比差异来评估翻译质量")
    print()

    text = "好好学习，天天向上"
    language = "English"

    # 完整 pipeline
    chain = (
        {
            "original": lambda x: x["text"],
            "language": lambda x: x["language"],
            "translation": translate_chain,
        }
        # 反向翻译
        | RunnablePassthrough.assign(
            back_translation=lambda x: back_translate_chain.invoke({
                "text": x["translation"],
            })
        )
        # 相似度评估
        | RunnablePassthrough.assign(
            similarity=lambda x: _simple_similarity(x["original"], x["back_translation"]),
        )
    )

    result = chain.invoke({"text": text, "language": language})

    print(f"原文:       {result['original']}")
    print(f"英文翻译:   {result['translation']}")
    print(f"反向翻译:   {result['back_translation']}")
    print(f"语义保持度: {result['similarity']}")
    print()


def _simple_similarity(text1: str, text2: str) -> str:
    """简单的文本相似度评估"""
    # 用 LLM 评估
    prompt = ChatPromptTemplate.from_template(
        "评估以下两段中文文本的语义相似度，回复 高/中/低：\n\n文本1: {t1}\n文本2: {t2}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"t1": text1, "t2": text2})


if __name__ == "__main__":
    demo_single_translate()
    demo_multi_translate()
    demo_back_translation()

    print("=" * 50)
    print("Demo 5-4 完成!")
    print()
    print("本 demo 综合运用了:")
    print("  LCEL 管道        - prompt | llm | parser")
    print("  RunnableParallel - 并行翻译多语言")
    print("  RunnablePassthrough.assign - 追加计算字段")
    print("  RunnableLambda   - 自定义函数包装")
    print("=" * 50)
