"""
第28章 Demo 3：幻觉预防策略实战

演示 Prompt 约束法、分层验证法等预防策略。
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


def demo_unsafe_vs_safe_prompt():
    """对比不安全 Prompt vs 安全 Prompt"""
    print("=" * 60)
    print(f"Demo 28-3 (1/4): 不安全 vs 安全 Prompt [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    question = "李白的具体出生日期是哪一天？"

    # 不安全 Prompt
    unsafe_prompt = ChatPromptTemplate.from_template("""
请回答问题：{question}
要求详细、具体，提供精确的信息。
""")

    # 安全 Prompt
    safe_prompt = ChatPromptTemplate.from_template("""
请回答以下问题：
{question}

回答要求：
1. 如果问题超出你的知识范围，请明确说明"我无法确定"
2. 对于不确定的信息，使用"可能"、"据记载"等词
3. 不要编造无法验证的具体数字、日期
4. 如果历史记载不明确，请如实说明

请分层回答：
- 确认部分：[你确信的信息]
- 不确定部分：[你不确定的信息]
""")

    unsafe_answer = (unsafe_prompt | llm | StrOutputParser()).invoke({"question": question})
    safe_answer = (safe_prompt | llm | StrOutputParser()).invoke({"question": question})

    print("问题:", question)
    print()
    print("【不安全 Prompt 回答】")
    print(unsafe_answer)
    print()
    print("【安全 Prompt 回答】")
    print(safe_answer)
    print()

    print("对比分析:")
    print("  不安全 Prompt 可能诱导模型给出过于精确但不可验证的信息")
    print("  安全 Prompt 要求模型标注不确定性，减少幻觉风险")


def demo_layered_validation():
    """分层验证法"""
    print("=" * 60)
    print(f"Demo 28-3 (2/4): 分层验证法 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    question = "iPhone 15 的具体发布日期是什么？"

    layered_prompt = ChatPromptTemplate.from_template("""
请分层回答以下问题，按置信度分类：

问题：{question}

请将回答分为三层：
1. **高置信度**：可通过权威来源验证的事实
2. **中置信度**：广泛认知但难以精确验证的内容
3. **低置信度**：推断、估计、可能不准确的内容

如果某层没有内容请填写"无"。
""")

    result = (layered_prompt | llm | StrOutputParser()).invoke({"question": question})

    print("问题:", question)
    print()
    print("分层回答:")
    print(result)
    print()

    print("分层验证优势:")
    print("  - 用户可以看到哪些信息是可靠的")
    print("  - 模型被迫区分确信和不确定的内容")
    print("  - 降低整体幻觉风险")


def demo_strict_rag_simulation():
    """强约束 RAG 模拟"""
    print("=" * 60)
    print(f"Demo 28-3 (3/4): 强约束 RAG 模拟 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 模拟检索内容
    context = """
[来源 1] iPhone 15 于 2023 年 9 月发布。
[来源 2] iPhone 15 Pro 和 iPhone 15 Pro Max 同期发布。
[来源 3] iPhone 15 系列支持 USB-C 接口。
"""

    question = "iPhone 15 的详细规格参数有哪些？"

    strict_prompt = ChatPromptTemplate.from_template("""
**严格基于以下参考资料回答问题**

参考资料：
{context}

问题：{question}

规则：
1. 只使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，必须回答："参考资料中未找到相关信息"
3. 不要添加任何参考资料之外的知识
4. 不要猜测或推断
""")

    strict_answer = (strict_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })

    print("参考资料:")
    print(context)
    print()
    print("问题:", question)
    print()
    print("强约束回答:")
    print(strict_answer)
    print()

    print("强约束 RAG 特点:")
    print("  - 严格限制只使用检索内容")
    print("  - 缺少信息时明确说明")
    print("  - 最大化减少幻觉")


def demo_temperature_comparison():
    """温度对比演示"""
    print("=" * 60)
    print(f"Demo 28-3 (4/4): 温度参数对比 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    question = "李白的一生有哪些重要的历史事件？"

    # 低温度
    llm_low_temp = ChatOpenAI(
        model=QWEN_MODEL, temperature=0.1,
        api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL
    )

    # 较高温度
    llm_high_temp = ChatOpenAI(
        model=QWEN_MODEL, temperature=0.7,
        api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL
    )

    prompt = ChatPromptTemplate.from_template("""
请简要介绍：{question}
提供几个重要事件和年份。
""")

    low_temp_answer = (prompt | llm_low_temp | StrOutputParser()).invoke({"question": question})
    high_temp_answer = (prompt | llm_high_temp | StrOutputParser()).invoke({"question": question})

    print("问题:", question)
    print()
    print("【温度 0.1 回答】（更保守）")
    print(low_temp_answer)
    print()
    print("【温度 0.7 回答】（更有创意）")
    print(high_temp_answer)
    print()

    print("温度影响:")
    print("  - 低温度：输出更保守、确定性更高，幻觉风险更低")
    print("  - 高温度：输出更有创意、多样性更高，但幻觉风险增加")
    print("  - 事实查询场景：建议使用低温度（0-0.1）")


if __name__ == "__main__":
    demo_unsafe_vs_safe_prompt()
    demo_layered_validation()
    demo_strict_rag_simulation()
    demo_temperature_comparison()

    print("=" * 60)
    print("Demo 28-3 完成!")
    print()
    print("幻觉预防策略总结：")
    print("  1. 安全 Prompt：明确知识边界，标注不确定性")
    print("  2. 分层验证：按置信度分层输出")
    print("  3. 强约束 RAG：只使用检索内容，不添加额外信息")
    print("  4. 低温度参数：事实查询场景使用低温度")
    print()
    print("最佳实践组合：")
    print("  - 事实查询：安全 Prompt + 低温度")
    print("  - RAG 问答：强约束 Prompt + 源引用检测")
    print("  - 高风险场景：分层验证 + 人工复核")
    print("=" * 60)