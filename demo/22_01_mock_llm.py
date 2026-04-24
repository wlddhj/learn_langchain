"""
第22章 Demo 1：Mock LLM 测试

演示使用 Mock LLM 进行单元测试。
无需 API Key。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class MockLLM:
    """模拟 LLM 用于测试"""

    def __init__(self, responses: dict = None):
        self.responses = responses or {
            "你好": "你好！有什么可以帮助你的？",
            "Python": "Python 是一种高级编程语言。",
            "default": "这是一个模拟回答。",
        }

    def invoke(self, input_text):
        """模拟 invoke"""
        # 根据关键词返回预设回答
        for key, response in self.responses.items():
            if key in str(input_text):
                return MockResponse(response)

        return MockResponse(self.responses["default"])

    def batch(self, inputs):
        """模拟 batch"""
        return [self.invoke(inp) for inp in inputs]


class MockResponse:
    """模拟响应"""

    def __init__(self, content: str):
        self.content = content


def demo_mock_llm():
    """Mock LLM 基础"""
    print("=" * 60)
    print("Demo 22-1 (1/3): Mock LLM 基础")
    print("=" * 60)
    print()

    mock_llm = MockLLM()

    print("使用 Mock LLM 测试：")
    print("-" * 60)

    test_inputs = ["你好", "介绍一下 Python", "其他问题"]
    for inp in test_inputs:
        result = mock_llm.invoke(inp)
        print(f"输入: {inp}")
        print(f"输出: {result.content}")
        print()


def demo_chain_testing():
    """Chain 测试"""
    print("=" * 60)
    print("Demo 22-1 (2/3): Chain 单元测试")
    print("=" * 60)
    print()

    from langchain_core.prompts import ChatPromptTemplate

    # 创建 Prompt
    prompt = ChatPromptTemplate.from_template("翻译成英文: {text}")

    # 使用 Mock LLM
    mock_llm = MockLLM({
        "你好": "Hello",
        "世界": "World",
        "default": "Translation result",
    })

    # 测试 Chain
    print("测试翻译 Chain：")
    print("-" * 60)

    test_texts = ["你好", "世界", "测试"]
    for text in test_texts:
        formatted = prompt.format(text=text)
        result = mock_llm.invoke(formatted)
        print(f"输入: {text}")
        print(f"输出: {result.content}")
        print()


def demo_unit_test_example():
    """单元测试示例"""
    print("=" * 60)
    print("Demo 22-1 (3/3): 单元测试代码示例")
    print("=" * 60)
    print()

    code = """
import pytest
from unittest.mock import Mock

def test_chain_output():
    """测试 Chain 输出格式"""
    # 创建 Mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = MockResponse("测试回答")

    # 创建 Chain
    chain = prompt | mock_llm | StrOutputParser()

    # 测试
    result = chain.invoke({"text": "输入"})
    assert result == "测试回答"

def test_batch_calls():
    """测试批量调用"""
    mock_llm = MockLLM({"test": "测试结果"})

    results = mock_llm.batch(["test1", "test2"])
    assert len(results) == 2
    assert all(r.content for r in results)
"""

    print("pytest 单元测试示例：")
    print("-" * 60)
    print(code)


if __name__ == "__main__":
    demo_mock_llm()
    demo_chain_testing()
    demo_unit_test_example()

    print("=" * 60)
    print("Demo 22-1 完成!")
    print()
    print("测试要点：")
    print("  - Mock LLM: 无需真实 API 的测试")
    print("  - 预设响应: 控制测试输出")
    print("  - pytest: Python 测试框架")
    print()
    print("测试覆盖：")
    print("  - Chain 输出格式")
    print("  - 错误处理逻辑")
    print("  - Prompt 模板格式化")
    print("=" * 60)