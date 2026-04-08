"""
第3章 Demo 2：MessagesPlaceholder 与 Few-Shot 提示

演示历史消息占位符、Few-Shot 少样本示例引导、模板组合。
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
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


def demo_messages_placeholder():
    """MessagesPlaceholder：动态插入对话历史"""
    print("=" * 50)
    print("Demo 3-2 (1/3): MessagesPlaceholder 对话历史")
    print("=" * 50)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手，能记住之前的对话内容。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # 模拟带历史的对话
    chat_history = [
        HumanMessage(content="我叫小明，我是一名前端开发者。"),
        AIMessage(content="你好小明！很高兴认识你。有什么前端开发方面的问题我可以帮你吗？"),
    ]

    result = chain.invoke({
        "chat_history": chat_history,
        "question": "我叫什么名字？我做什么工作？",
    })
    print(f"AI 能记住历史上下文:")
    print(f"  {result[:200]}")
    print()

    # 另一个例子：空历史
    result2 = chain.invoke({
        "chat_history": [],
        "question": "你好，我叫什么名字？",
    })
    print(f"空历史时:")
    print(f"  {result2[:200]}")
    print()


def demo_few_shot():
    """Few-Shot：少样本示例引导输出格式"""
    print("=" * 50)
    print("Demo 3-2 (2/3): Few-Shot 少样本提示")
    print("=" * 50)

    # 定义示例（输入-输出对）
    examples = [
        {"input": "开心", "output": "😄 开心 (happy) - 形容心情愉悦，常用于表达快乐和满足感"},
        {"input": "难过", "output": "😢 难过 (sad) - 形容心情低落，常用于表达失落和悲伤"},
        {"input": "生气", "output": "😠 生气 (angry) - 形容心情愤怒，常用于表达不满和恼火"},
    ]

    # 示例模板：每个示例的格式
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # 构建 Few-Shot 模板
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 组合成完整 prompt
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感词典，按照示例格式解释情感词汇。"),
        few_shot_prompt,
        ("human", "{word}"),
    ])

    chain = final_prompt | llm | StrOutputParser()

    # 测试新词汇
    test_words = ["紧张", "兴奋", "感动"]
    for word in test_words:
        result = chain.invoke({"word": word})
        print(f"输入: {word}")
        print(f"输出: {result}")
        print()


def demo_template_composition():
    """模板组合：用 + 拼接模板"""
    print("=" * 50)
    print("Demo 3-2 (3/3): 模板组合")
    print("=" * 50)

    # 基础系统 prompt
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{domain}专家。请用中文回答。"),
    ])

    # 用 + 追加用户消息
    full_prompt = system_prompt + ("human", "{question}")

    print(f"组合后模板变量: {full_prompt.input_variables}")

    chain = full_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "domain": "Python 编程",
        "question": "列表推导式和生成器表达式有什么区别？",
    })
    print(f"domain=Python编程")
    print(f"回答: {result[:300]}")
    print()

    # 换个领域
    result2 = chain.invoke({
        "domain": "营养学",
        "question": "每天需要摄入多少蛋白质？",
    })
    print(f"domain=营养学")
    print(f"回答: {result2[:300]}")
    print()


if __name__ == "__main__":
    demo_messages_placeholder()
    demo_few_shot()
    demo_template_composition()

    print("=" * 50)
    print("Demo 3-2 完成!")
    print()
    print("要点:")
    print("  MessagesPlaceholder - 动态插入对话历史消息")
    print("  FewShotChatMessagePromptTemplate - 少样本示例引导格式")
    print("  模板组合 - 用 + 拼接多个模板")
    print("=" * 50)
