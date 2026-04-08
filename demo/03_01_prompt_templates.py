"""
第3章 Demo 1：PromptTemplate 与 ChatPromptTemplate

演示两种模板的创建、变量注入、自动识别变量、多变量模板。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GLM_API_KEY") or os.environ["GLM_API_KEY"].startswith("your-"):
    print("错误: 未设置 GLM_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def demo_prompt_template():
    """PromptTemplate：纯文本模板"""
    print("=" * 50)
    print("Demo 3-1 (1/3): PromptTemplate 纯文本模板")
    print("=" * 50)

    # 单变量模板
    template = PromptTemplate.from_template("给我讲一个关于{topic}的笑话")
    print(f"模板变量: {template.input_variables}")
    print(f"渲染结果: {template.invoke({'topic': '程序员'}).to_string()}")
    print()

    # 多变量模板
    translate_template = PromptTemplate.from_template(
        "请将以下{source_lang}文本翻译成{target_lang}：\n\n{text}"
    )
    print(f"模板变量: {translate_template.input_variables}")

    result = translate_template.invoke({
        "source_lang": "中文",
        "target_lang": "英文",
        "text": "今天天气真好",
    })
    print(f"渲染结果:\n{result.to_string()}")
    print()


def demo_chat_prompt_template():
    """ChatPromptTemplate：对话模板（推荐）"""
    print("=" * 50)
    print("Demo 3-1 (2/3): ChatPromptTemplate 对话模板")
    print("=" * 50)

    # 方式一：从消息元组构建
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，用{style}的风格回答问题。"),
        ("human", "{question}"),
    ])

    print(f"模板变量: {prompt.input_variables}")
    print()

    # 构建 chain 并调用
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "role": "资深 Python 开发者",
        "style": "简洁专业",
        "question": "什么是装饰器？",
    })
    print(f"角色=资深Python开发者, 风格=简洁专业")
    print(f"回答: {result[:200]}")
    print()

    # 换个角色试试
    result2 = chain.invoke({
        "role": "5岁的小朋友",
        "style": "天真可爱，用比喻来解释",
        "question": "什么是装饰器？",
    })
    print(f"角色=5岁小朋友, 风格=天真可爱")
    print(f"回答: {result2[:200]}")
    print()


def demo_message_types():
    """多轮对话消息模板"""
    print("=" * 50)
    print("Demo 3-1 (3/3): 多轮对话消息模板")
    print("=" * 50)

    # 包含 AI 历史消息的模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的 AI 助手。"),
        ("human", "什么是 Python？"),
        ("ai", "Python 是一种广泛使用的高级编程语言。"),
        ("human", "{question}"),  # 用户追问
    ])

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "question": "它和 Java 有什么区别？",
    })
    print(f"上下文: 用户先问了'什么是Python'，AI已回答")
    print(f"追问: 它和 Java 有什么区别？")
    print(f"回答: {result[:300]}")
    print()


if __name__ == "__main__":
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_message_types()

    print("=" * 50)
    print("Demo 3-1 完成!")
    print()
    print("要点:")
    print("  PromptTemplate     - 纯文本模板，适用于 LLM 接口")
    print("  ChatPromptTemplate - 对话模板，适用于 ChatModel（推荐）")
    print("  from_messages()    - 从消息元组列表构建模板")
    print("  input_variables    - 自动识别模板中的变量")
    print("=" * 50)
