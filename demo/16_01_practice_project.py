"""
第16章 Demo：综合实战项目架构演示

展示智能研究助手的核心架构和组件组合。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
import asyncio

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def demo_architecture():
    """项目架构概览"""
    print("=" * 60)
    print("Demo 16: 综合实战项目架构")
    print("=" * 60)
    print()

    print("智能研究助手架构：")
    print("-" * 60)
    print("""
用户输入
    ↓
[问题分类] → 简单问题 → 直接回答
    ↓ 复杂问题
[工具调用] → 搜索信息
    ↓
[研究整理] → 整合信息
    ↓
[分析推理] → 深度分析
    ↓
[报告生成] → 结构化输出
    ↓
[质量审查] → 不通过? → 循环改进
    ↓ 通过
输出结果 + 记忆持久化
""")
    print()

    print("涉及的核心知识：")
    print("-" * 60)
    print("""
| 章节 | 在项目中的应用 |
|------|----------------|
| 第1-2章 | 环境搭建、LLM 调用 |
| 第3-4章 | Prompt 模板、结构化输出 |
| 第5章 | LCEL 管道操作 |
| 第6-7章 | 文档加载、向量检索（RAG） |
| 第8章 | 对话记忆管理 |
| 第9、12章 | 自定义工具 |
| 第10-11章 | ReAct Agent 模式 |
| 第13-14章 | LangGraph 工作流 |
| 第15章 | 多 Agent 协作 |
""")


def demo_custom_tools():
    """自定义工具示例"""
    print("=" * 60)
    print("Demo 16 (2/3): 自定义工具")
    print("=" * 60)
    print()

    @tool
    def mock_search(query: str) -> str:
        """模拟搜索工具。返回预设结果用于演示。"""
        mock_data = {
            "python": "Python 是一门高级编程语言，具有简洁易读的语法...",
            "rag": "RAG（检索增强生成）是一种结合检索和生成的技术...",
            "agent": "AI Agent 是能够自主决策和执行任务的智能系统...",
        }
        for key, value in mock_data.items():
            if key in query.lower():
                return value
        return "未找到相关结果"

    @tool
    def save_note(content: str) -> str:
        """保存笔记到文件。"""
        import tempfile
        path = tempfile.mktemp(suffix=".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"笔记已保存: {path}"

    tools = [mock_search, save_note]
    llm_with_tools = llm.bind_tools(tools)

    print("工具定义示例：")
    print("-" * 60)

    # 测试工具绑定
    response = llm_with_tools.invoke("搜索 Python 的相关信息")
    print(f"用户: 搜索 Python 的相关信息")
    print(f"模型响应: {response.content[:100]}...")

    if response.tool_calls:
        print(f"工具调用: {response.tool_calls}")

    print()
    print("工具要点：")
    print("  - @tool 装饰器定义工具")
    print("  - bind_tools 绑定到模型")
    print("  - 工具描述帮助 LLM 正确选择")


async def demo_lcel_pipeline():
    """LCEL 管道示例"""
    print("=" * 60)
    print(f"Demo 16 (3/3): LCEL 管道 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    # 研究节点管道
    research_prompt = ChatPromptTemplate.from_template("""
你是一个研究专家。分析以下问题并整理信息。

问题: {question}

请输出研究笔记，包含：
1. 关键发现
2. 信息评估
3. 需补充的内容
""")

    research_chain = research_prompt | llm | StrOutputParser()

    # 分析节点管道
    analysis_prompt = ChatPromptTemplate.from_template("""
你是一个分析专家。基于研究笔记进行深度分析。

研究笔记:
{notes}

请输出分析报告，包含：
1. 核心观点
2. 对比分析
3. 专业见解
""")

    analysis_chain = analysis_prompt | llm | StrOutputParser()

    # 报告节点管道
    report_prompt = ChatPromptTemplate.from_template("""
你是一个写作专家。基于分析结果撰写简洁报告。

分析结果:
{analysis}

请输出 Markdown 格式的报告。
""")

    report_chain = report_prompt | llm | StrOutputParser()

    print("LCEL 管道示例：")
    print("-" * 60)

    # 模拟完整流程
    question = "什么是 LangChain？"

    print(f"问题: {question}")
    print()

    # 步骤1：研究
    print("[研究节点]")
    notes = await research_chain.ainvoke({"question": question})
    print(f"研究笔记: {notes[:150]}...")
    print()

    # 步骤2：分析
    print("[分析节点]")
    analysis = await analysis_chain.ainvoke({"notes": notes})
    print(f"分析结果: {analysis[:150]}...")
    print()

    # 步骤3：报告
    print("[报告节点]")
    report = await report_chain.ainvoke({"analysis": analysis})
    print(f"报告输出: {report[:150]}...")
    print()

    print("管道要点：")
    print("  - prompt | llm | parser 链式组合")
    print("  - 每个节点独立定义")
    print("  - 数据在节点间传递")


async def main():
    demo_architecture()
    demo_custom_tools()
    await demo_lcel_pipeline()

    print("=" * 60)
    print("Demo 16 完成!")
    print()
    print("项目架构要点：")
    print("  - 问题分类：路由到不同处理路径")
    print("  - 自定义工具：搜索、保存、分析")
    print("  - LCEL 管道：节点间数据传递")
    print("  - LangGraph：编排完整工作流")
    print()
    print("扩展方向：")
    print("  - Web 界面 (Streamlit)")
    print("  - 数据库持久化")
    print("  - API 服务化")
    print("  - Docker 部署")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())