"""
第23章 Demo 1：LangSmith 追踪基础

演示 LangSmith 的基本追踪功能。
需要 LANGCHAIN_API_KEY。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def demo_langsmith_setup():
    """LangSmith 配置"""
    print("=" * 60)
    print("Demo 23-1 (1/3): LangSmith 配置")
    print("=" * 60)
    print()

    print("LangSmith 配置步骤：")
    print("-" * 60)
    print("""
1. 注册 LangSmith 账号
   https://smith.langchain.com

2. 获取 API Key
   Settings → API Keys → Create

3. 配置环境变量
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-api-key
   LANGCHAIN_PROJECT=your-project-name

4. 代码中自动启用追踪
   所有 LLM 调用会自动记录到 LangSmith
""")


def demo_tracing_benefits():
    """追踪的好处"""
    print("=" * 60)
    print("Demo 23-1 (2/3): LangSmith 功能")
    print("=" * 60)
    print()

    print("LangSmith 主要功能：")
    print("-" * 60)
    print("""
| 功能 | 说明 | 用途 |
|------|------|------|
| 追踪 | 记录所有 LLM 调用 | 调试、审计 |
| 调试 | 查看 Prompt 和输出 | 问题定位 |
| 评估 | 自动评估答案质量 | 性能监控 |
| 监控 | Token 使用和成本 | 成本控制 |
| 比较 | 对比不同版本效果 | A/B测试 |
| 数据集 | 管理测试数据 | 批量评估 |
""")
    print()


def demo_tracing_example():
    """追踪代码示例"""
    print("=" * 60)
    print("Demo 23-1 (3/3): 追踪代码示例")
    print("=" * 60)
    print()

    code = """
import os

# 配置 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

from langchain_openai import ChatOpenAI

# 创建 LLM（自动追踪）
llm = ChatOpenAI(model="gpt-4o-mini")

# 调用会自动记录到 LangSmith
result = llm.invoke("你好")

# 查看追踪
# https://smith.langchain.com/o/{org}/projects/p/{project}
"""

    print("追踪代码：")
    print("-" * 60)
    print(code)


if __name__ == "__main__":
    demo_langsmith_setup()
    demo_tracing_benefits()
    demo_tracing_example()

    print("=" * 60)
    print("Demo 23-1 完成!")
    print()
    print("LangSmith 核心价值：")
    print("  - 自动追踪所有调用")
    print("  - 调试 Prompt 和输出")
    print("  - 评估答案质量")
    print("  - 监控成本和性能")
    print()
    print("使用建议：")
    print("  - 开发时启用追踪")
    print("  - 生产环境用于监控")
    print("  - 定期评估系统质量")
    print("=" * 60)