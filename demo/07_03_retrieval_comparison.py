"""
第7章 Demo 3：实战 —— 检索策略效果对比

用同一组数据对比不同检索策略的召回效果。
可独立运行。
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
embeddings = OpenAIEmbeddings(model="text-embedding-v3", api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def build_dataset():
    """AI 技术知识库"""
    docs = [
        Document(page_content="GPT-4 是 OpenAI 的大语言模型，具备强大的推理和代码生成能力。", metadata={"topic": "gpt", "type": "model"}),
        Document(page_content="BERT 是 Google 提出的预训练语言模型，主要用于文本理解和分类任务。", metadata={"topic": "bert", "type": "model"}),
        Document(page_content="LLaMA 是 Meta 开源的大语言模型系列，社区活跃，衍生出许多微调版本。", metadata={"topic": "llama", "type": "model"}),
        Document(page_content="Stable Diffusion 是开源的图像生成模型，使用扩散模型架构。", metadata={"topic": "sd", "type": "vision"}),
        Document(page_content="Whisper 是 OpenAI 的语音识别模型，支持多语言转写。", metadata={"topic": "whisper", "type": "audio"}),
        Document(page_content="RAG 通过检索外部知识库增强 LLM 的回答质量，减少幻觉。", metadata={"topic": "rag", "type": "technique"}),
        Document(page_content="LoRA 是一种参数高效的微调方法，只需训练少量参数即可适配新任务。", metadata={"topic": "lora", "type": "technique"}),
        Document(page_content="Prompt Engineering 是设计提示词的技术，包括 Few-Shot、Chain-of-Thought 等。", metadata={"topic": "prompt", "type": "technique"}),
        Document(page_content="LangChain 是构建 LLM 应用的框架，支持 RAG、Agent、Chain 等模式。", metadata={"topic": "langchain", "type": "tool"}),
        Document(page_content="Hugging Face 是最大的 AI 模型和数据集社区，提供 Transformers 库。", metadata={"topic": "hf", "type": "platform"}),
        Document(page_content="FAISS 是 Facebook 开发的高效向量相似度搜索库，支持十亿级索引。", metadata={"topic": "faiss", "type": "tool"}),
        Document(page_content="LangGraph 用于构建有状态的 AI Agent，基于图结构管理工作流。", metadata={"topic": "langgraph", "type": "tool"}),
    ]
    return docs


def compare_strategies(query, docs, k=3):
    """对比不同检索策略"""
    # 构建向量库
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 策略1: Similarity
    sim_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # 策略2: MMR
    mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 10})

    # 策略3: BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    # 策略4: Ensemble (BM25 + Vector)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, sim_retriever],
        weights=[0.5, 0.5],
    )

    strategies = {
        "Similarity": sim_retriever,
        "MMR": mmr_retriever,
        "BM25": bm25_retriever,
        "Ensemble": ensemble_retriever,
    }

    print(f"查询: {query}\n")

    for name, retriever in strategies.items():
        results = retriever.invoke(query)
        print(f"  [{name}] 返回 {len(results)} 条:")
        for doc in results:
            topic = doc.metadata.get("topic", "?")
            print(f"    ({topic}) {doc.page_content[:60]}...")
        print()


def main():
    print("=" * 50)
    print(f"检索策略效果对比 [{QWEN_MODEL}]")
    print("=" * 50)
    print()

    docs = build_dataset()
    print(f"知识库: {len(docs)} 条文档")
    print()

    # 测试不同类型的查询
    queries = [
        "开源大语言模型有哪些？",       # 语义查询
        "FAISS",                        # 精确关键词
        "如何提高 LLM 的回答质量？",     # 概念查询
    ]

    for q in queries:
        print("=" * 50)
        compare_strategies(q, docs, k=3)

    print("=" * 50)
    print("检索策略选择指南:")
    print()
    print("  查询类型         推荐策略")
    print("  ──────────────   ────────────────────")
    print("  精确关键词        BM25 或 Ensemble")
    print("  语义/概念查询     Similarity 或 MMR")
    print("  结果重复          MMR")
    print("  综合效果最好      Ensemble (BM25+向量)")
    print("  召回不足          MultiQuery + Ensemble")
    print("=" * 50)


if __name__ == "__main__":
    main()
