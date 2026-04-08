"""
第2章 Demo 1：ChatModel 基础使用与消息类型

演示 ChatModel 的创建、消息类型、调用方式、Token 用量追踪。
可独立运行，需要 GLM_API_KEY。
"""

import os
import sys
from pathlib import Path

# 加载项目根目录的 .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# 检查 API Key
if not os.environ.get("GLM_API_KEY") or os.environ["GLM_API_KEY"].startswith("your-"):
    print("错误: 未设置 GLM_API_KEY")
    print("请在项目根目录的 .env 文件中配置: GLM_API_KEY=xxx")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 从环境变量读取 GLM 配置
GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")


def create_llm(temperature=0, max_tokens=None):
    """创建 GLM ChatModel 实例"""
    kwargs = dict(
        model=GLM_MODEL,
        temperature=temperature,
        api_key=GLM_API_KEY,
        base_url=GLM_BASE_URL,
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def demo_basic_invoke():
    """演示最基本的 LLM 调用"""
    print("=" * 50)
    print("Demo 2-1 (1/4): 基本调用")
    print(f"模型: {GLM_MODEL}")
    print("=" * 50)

    llm = create_llm()

    # 最简调用：直接传字符串
    response = llm.invoke("用一句话解释什么是人工智能。")
    print(f"回复: {response.content}")
    print(f"类型: {type(response).__name__}")
    print()


def demo_message_types():
    """演示三种消息类型"""
    print("=" * 50)
    print("Demo 2-1 (2/4): 消息类型")
    print("=" * 50)

    llm = create_llm()

    # --- 方式一：使用消息对象 ---
    print("--- 方式一: 消息对象 ---")
    messages_obj = [
        SystemMessage(content="你是一个海盗风格的助手，用海盗的口吻回答问题。"),
        HumanMessage(content="今天天气怎么样？"),
    ]
    response = llm.invoke(messages_obj)
    print(f"AI 回复: {response.content}")
    print()

    # --- 方式二：使用元组语法（推荐）---
    print("--- 方式二: 元组语法（推荐）---")
    messages_tuple = [
        ("system", "你是一个诗人，用简洁优美的语言回答。"),
        ("human", "什么是递归？"),
    ]
    response = llm.invoke(messages_tuple)
    print(f"AI 回复: {response.content}")
    print()

    # --- 多轮对话 ---
    print("--- 多轮对话 ---")
    conversation = [
        ("system", "你是一个历史老师，用通俗易懂的方式讲解历史。"),
        ("human", "唐朝是哪一年建立的？"),
    ]
    r1 = llm.invoke(conversation)
    print(f"用户: 唐朝是哪一年建立的？")
    print(f"AI: {r1.content}")

    # 把 AI 回复加入对话历史，继续追问
    conversation.append(("ai", r1.content))
    conversation.append(("human", "它灭亡的原因是什么？"))

    r2 = llm.invoke(conversation)
    print(f"\n用户: 它灭亡的原因是什么？")
    print(f"AI: {r2.content}")
    print()


def demo_temperature():
    """演示 temperature 参数对输出的影响"""
    print("=" * 50)
    print("Demo 2-1 (3/4): Temperature 参数对比")
    print("=" * 50)

    prompt = "给我起一个咖啡店的名字，只需要一个名字，不要其他内容。"

    # temperature=0: 确定性输出，多次调用结果基本一致
    llm_deterministic = create_llm(temperature=0)

    # temperature=1: 随机性更强，每次结果可能不同
    llm_creative = create_llm(temperature=1.0)

    print("temperature=0 (确定性):")
    for i in range(3):
        r = llm_deterministic.invoke(prompt)
        print(f"  第{i+1}次: {r.content.strip()}")

    print()
    print("temperature=1.0 (创造性):")
    for i in range(3):
        r = llm_creative.invoke(prompt)
        print(f"  第{i+1}次: {r.content.strip()}")
    print()


def demo_token_usage():
    """演示 Token 用量追踪"""
    print("=" * 50)
    print("Demo 2-1 (4/4): Token 用量追踪")
    print("=" * 50)

    llm = create_llm()

    response = llm.invoke("请详细解释什么是机器学习，包括它的主要类型和应用场景。")

    print(f"回复内容 ({len(response.content)} 字符):")
    print(f"  {response.content[:200]}...")
    print()

    # 提取 token 用量
    metadata = response.response_metadata
    print(f"模型: {metadata.get('model_name', 'unknown')}")
    print(f"完成原因: {metadata.get('finish_reason', 'unknown')}")
    print()

    token_usage = metadata.get("token_usage", {})
    if token_usage:
        print("Token 用量:")
        print(f"  输入 tokens:  {token_usage.get('prompt_tokens')}")
        print(f"  输出 tokens:  {token_usage.get('completion_tokens')}")
        print(f"  总计 tokens:  {token_usage.get('total_tokens')}")
    else:
        print("Token 用量信息不可用")

    print()
    print("提示: 监控 token 用量有助于控制 API 调用成本。")
    print()


if __name__ == "__main__":
    demo_basic_invoke()
    demo_message_types()
    demo_temperature()
    demo_token_usage()

    print("=" * 50)
    print("Demo 2-1 完成!")
    print("=" * 50)
