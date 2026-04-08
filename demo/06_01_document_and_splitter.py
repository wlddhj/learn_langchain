"""
第6章 Demo 1：文档加载与文本分割

演示 Document 对象、TextLoader、RecursiveCharacterTextSplitter、FAISS 向量存储基础。
可独立运行，自动创建示例文档。
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
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# 1. Document 对象
# ============================================================

def demo_document_basics():
    print("=" * 50)
    print("Demo 6-1 (1/3): Document 对象")
    print("=" * 50)

    doc = Document(
        page_content="LangChain 是一个用于开发大语言模型应用的开源框架。",
        metadata={"source": "intro.txt", "topic": "LangChain"},
    )
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print(f"类型: {type(doc).__name__}")
    print()


# ============================================================
# 2. 创建示例文档并加载
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "knowledge"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_docs():
    """创建示例知识文档"""
    docs = {
        "langchain_intro.txt": """\
LangChain 是一个用于开发大语言模型应用的开源框架。
它提供了模块化的组件，让开发者可以轻松构建 LLM 应用。
核心组件包括：Model I/O、Retrieval、Chains、Agents、Memory。

LangChain 支持多种大语言模型，包括 OpenAI GPT、Anthropic Claude、
智谱 GLM 等。通过统一的接口，开发者可以轻松切换不同的模型。

LangChain 的设计理念是让 AI 应用的开发像搭积木一样简单。
开发者可以将不同的组件组合在一起，构建出复杂的应用。""",

        "rag_guide.txt": """\
RAG（检索增强生成）是一种结合了信息检索和文本生成的技术。
它的工作流程是：先从知识库中检索相关文档，然后将文档作为上下文
传递给大语言模型，由模型生成最终回答。

RAG 的优势在于可以让 LLM 利用外部知识，解决知识过时和幻觉问题。
常见的 RAG 应用包括：企业知识库问答、客服机器人、文档分析等。

构建 RAG 系统的关键步骤：文档加载、文本分割、向量化、向量存储、检索、生成。
每个步骤都有多种技术选择，需要根据具体场景进行优化。""",

        "agent_basics.txt": """\
AI Agent 是一种能够自主决策和执行任务的智能体。
它通过工具调用（Tool Use）来与外部世界交互。

Agent 的核心模式是 ReAct（Reason + Act）：
1. 推理（Reason）：分析当前情况和目标
2. 行动（Act）：选择并执行合适的工具
3. 观察（Observe）：获取工具执行的结果
4. 循环以上步骤直到任务完成

常见的 Agent 类型包括：ReAct Agent、Plan-and-Execute Agent、
Multi-Agent System 等。LangGraph 是构建复杂 Agent 的推荐框架。""",
    }

    for filename, content in docs.items():
        filepath = DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    return docs


def demo_load_documents():
    print("=" * 50)
    print("Demo 6-1 (2/3): 文档加载")
    print("=" * 50)

    create_sample_docs()

    # 使用 TextLoader 加载
    all_docs = []
    for f in sorted(DATA_DIR.glob("*.txt")):
        loader = TextLoader(str(f), encoding="utf-8")
        docs = loader.load()
        all_docs.extend(docs)
        print(f"加载: {f.name} → {len(docs)} 个文档")

    print(f"\n共加载 {len(all_docs)} 个文档")
    for doc in all_docs:
        print(f"  [{Path(doc.metadata['source']).name}] {len(doc.page_content)} 字符")
    print()

    return all_docs


def demo_text_splitter(docs):
    print("=" * 50)
    print("Demo 6-1 (3/3): 文本分割")
    print("=" * 50)

    # RecursiveCharacterTextSplitter（推荐）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    print(f"原始文档数: {len(docs)}")
    print(f"分割后块数: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks[:5]):
        source = Path(chunk.metadata["source"]).name
        print(f"--- Chunk {i+1} (来源: {source}, {len(chunk.page_content)} 字符) ---")
        print(f"{chunk.page_content[:120]}...")
        print()

    if len(chunks) > 5:
        print(f"... 还有 {len(chunks) - 5} 个 chunk")
    print()

    return chunks


if __name__ == "__main__":
    demo_document_basics()
    docs = demo_load_documents()
    chunks = demo_text_splitter(docs)

    print("=" * 50)
    print("Demo 6-1 完成!")
    print()
    print("RAG 数据准备流程:")
    print("  Document → TextLoader → RecursiveCharacterTextSplitter → chunks")
    print("  下一步: 向量化 → 存入向量数据库 (Demo 6-2)")
    print("=" * 50)
