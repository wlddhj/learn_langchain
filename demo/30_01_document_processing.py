"""
第30章 Demo 1：文档处理流程

演示文档加载、切分、向量嵌入的完整流程。
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

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

QWEN_API_KEY = os.environ["QWEN_API_KEY"]


def demo_document_structure():
    """Document 对象结构"""
    print("=" * 60)
    print("Demo 30-1 (1/4): Document 对象结构")
    print("=" * 60)
    print()

    # 创建文档示例
    doc = Document(
        page_content="Python 是一种解释型高级编程语言，以简洁的语法著称。",
        metadata={
            "source": "python_intro.txt",
            "author": "技术团队",
            "category": "编程语言",
            "created_at": "2024-01-15",
        }
    )

    print("Document 对象示例：")
    print("-" * 60)
    print(f"page_content: {doc.page_content}")
    print(f"metadata: {doc.metadata}")
    print()

    print("Document 关键属性：")
    print("  - page_content: 文本内容")
    print("  - metadata: 元数据字典（来源、作者、类别等）")
    print()


def demo_text_splitting():
    """文本切分演示"""
    print("=" * 60)
    print("Demo 30-1 (2/4): 文本切分策略")
    print("=" * 60)
    print()

    # 示例长文本
    long_text = """
Python 是一种解释型高级编程语言，由 Guido van Rossum 于 1991 年首次发布。
Python 的设计哲学强调代码的可读性和简洁性，它的语法允许程序员用更少的代码行表达概念。

Python 支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
它拥有一个庞大且活跃的社区，提供了丰富的第三方库和框架。

Python 的主要应用领域包括：
1. Web 开发：Django、Flask 等框架
2. 数据科学：NumPy、Pandas、Matplotlib
3. 机器学习：TensorFlow、PyTorch、Scikit-learn
4. 自动化脚本：系统管理、数据处理
5. 科学计算：SciPy、SymPy

Python 的优势在于其简洁的语法、丰富的库生态系统和跨平台支持。
这使得 Python 成为初学者入门和专业开发者的首选语言之一。
"""

    # 不同切分策略
    strategies = [
        {"chunk_size": 100, "chunk_overlap": 20, "name": "小片段"},
        {"chunk_size": 300, "chunk_overlap": 50, "name": "中等片段"},
        {"chunk_size": 500, "chunk_overlap": 100, "name": "大片段"},
    ]

    print("切分策略对比：")
    print("-" * 60)

    for strategy in strategies:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["chunk_overlap"],
            separators=["\n\n", "\n", "。", " ", ""],
        )

        chunks = splitter.split_text(long_text)

        print(f"\n【{strategy['name']}】chunk_size={strategy['chunk_size']}, overlap={strategy['chunk_overlap']}")
        print(f"生成片段数: {len(chunks)}")
        print(f"首个片段预览: {chunks[0][:50]}...")

    print()


def demo_chunk_size_selection():
    """Chunk 大小选择指南"""
    print("=" * 60)
    print("Demo 30-1 (3/4): Chunk 大小选择指南")
    print("=" * 60)
    print()

    guidelines = """
Chunk 大小选择建议：

| 场景 | 推荐 chunk_size | 推荐 overlap | 原因 |
|------|----------------|--------------|------|
| 简单问答 | 200-300 | 20-50 | 问题简短，小片段足够 |
| 详细解释 | 500-800 | 100-150 | 需要完整上下文 |
| 技术文档 | 300-500 | 50-100 | 平衡细节和上下文 |
| 长文档摘要 | 1000-1500 | 200-300 | 需要大范围信息 |
| 对话场景 | 300-500 | 50-100 | 支持多轮追问 |

选择原则：
1. chunk_size: 根据问题复杂度调整
2. chunk_overlap: 避免关键信息被切断
3. 检索后: 可用重排序优化结果
"""

    print(guidelines)


def demo_splitter_with_documents():
    """使用 Document 对象切分"""
    print("=" * 60)
    print("Demo 30-1 (4/4): Document 切分实战")
    print("=" * 60)
    print()

    # 创建多个文档
    docs = [
        Document(
            page_content="LangChain 是一个用于开发 LLM 应用的框架。它提供了链式调用、Agent、RAG 等功能。",
            metadata={"source": "langchain_intro.txt", "id": 1}
        ),
        Document(
            page_content="RAG（检索增强生成）结合了检索和生成技术，可以基于外部知识回答问题。",
            metadata={"source": "rag_intro.txt", "id": 2}
        ),
        Document(
            page_content="Agent 是能够自主决策和执行任务的 AI 系统，可以使用工具完成复杂任务。",
            metadata={"source": "agent_intro.txt", "id": 3}
        ),
    ]

    print("原始文档：")
    print("-" * 60)
    for doc in docs:
        print(f"  [{doc.metadata['id']}] {doc.page_content[:30]}...")
    print()

    # 切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=["。", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    print(f"切分后片段数: {len(chunks)}")
    print("-" * 60)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk.page_content[:40]}... (来源: {chunk.metadata['source']})")

    print()
    print("切分后 metadata 保留说明:")
    print("  - 原 metadata 自动继承到每个 chunk")
    print("  - 可添加 chunk_index 等额外信息")


if __name__ == "__main__":
    demo_document_structure()
    demo_text_splitting()
    demo_chunk_size_selection()
    demo_splitter_with_documents()

    print("=" * 60)
    print("Demo 30-1 完成!")
    print()
    print("文档处理流程总结：")
    print("  1. 加载文档 → Document 对象")
    print("  2. 选择切分策略 → chunk_size + overlap")
    print("  3. 执行切分 → 多个 Document chunks")
    print("  4. 保留 metadata → 来源追溯")
    print()
    print("关键参数：")
    print("  - chunk_size: 200-1000（根据场景）")
    print("  - chunk_overlap: 50-200（避免切断）")
    print("  - separators: 分隔符优先级")
    print("=" * 60)