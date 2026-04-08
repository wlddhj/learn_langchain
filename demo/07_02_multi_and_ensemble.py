"""
第7章 Demo 2：MultiQuery 与 Ensemble 混合检索

演示 MultiQueryRetriever 多角度查询、BM25+向量混合检索。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
embeddings = OpenAIEmbeddings(model="text-embedding-v3", api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


def build_test_data():
    """构建测试数据"""
    docs = [
        Document(page_content="Python 是一种解释型高级编程语言，以简洁的语法著称。", metadata={"id": 1}),
        Document(page_content="Java 是一种面向对象的编程语言，广泛用于企业级开发。", metadata={"id": 2}),
        Document(page_content="JavaScript 是 Web 前端的核心语言，也用于 Node.js 后端开发。", metadata={"id": 3}),
        Document(page_content="Rust 是一种系统级编程语言，注重内存安全和并发性能。", metadata={"id": 4}),
        Document(page_content="Go（Golang）由 Google 开发，擅长高并发和网络服务。", metadata={"id": 5}),
        Document(page_content="TypeScript 是 JavaScript 的超集，添加了静态类型检查。", metadata={"id": 6}),
        Document(page_content="C++ 是 C 语言的扩展，广泛用于系统开发、游戏引擎和高性能计算。", metadata={"id": 7}),
        Document(page_content="Swift 是 Apple 开发的编程语言，用于 iOS 和 macOS 应用开发。", metadata={"id": 8}),
        Document(page_content="Kotlin 是 JVM 语言，Google 推荐的 Android 开发首选语言。", metadata={"id": 9}),
        Document(page_content="SQL 是结构化查询语言，用于关系型数据库的数据操作。", metadata={"id": 10}),
    ]
    return docs


def demo_multi_query():
    """MultiQueryRetriever：用 LLM 从不同角度重写查询"""
    print("=" * 50)
    print(f"Demo 7-2 (1/2): MultiQueryRetriever [{QWEN_MODEL}]")
    print("=" * 50)

    docs = build_test_data()
    vectorstore = FAISS.from_documents(docs, embeddings)

    from langchain.retrievers.multi_query import MultiQueryRetriever

    # 自定义查询生成 prompt
    query_prompt = ChatPromptTemplate.from_template(
        """你是一个AI助手。请根据用户的问题，生成3个不同角度的搜索查询。

原始问题: {question}

要求：
- 从不同视角重新表述问题
- 使用不同的术语和表达
- 每行一个，不要编号

改写后的问题:"""
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm,
        prompt=query_prompt,
    )

    query = "适合做后端开发的语言有哪些？"
    print(f"原始查询: {query}")
    print()

    import logging
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)

    results = retriever.invoke(query)
    print(f"检索结果 ({len(results)} 条):")
    seen = set()
    for doc in results:
        if doc.metadata["id"] not in seen:
            seen.add(doc.metadata["id"])
            print(f"  [doc_{doc.metadata['id']}] {doc.page_content}")
    print()
    print("说明: MultiQuery 从多个角度重写查询，分别检索后合并去重，提高召回率")
    print()


def demo_ensemble_retriever():
    """EnsembleRetriever：混合 BM25 关键词 + 向量语义检索"""
    print("=" * 50)
    print(f"Demo 7-2 (2/2): EnsembleRetriever 混合检索")
    print("=" * 50)
    print("原理: BM25(精确关键词匹配) + 向量检索(语义相似) = 互补")
    print()

    docs = build_test_data()

    # BM25：关键词检索
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # 向量检索：语义检索
    vectorstore = FAISS.from_documents(docs, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    queries = [
        "Python",                         # 精确关键词
        "适合做手机应用开发的语言",          # 语义查询
        "高并发编程语言",                   # 语义查询
    ]

    for query in queries:
        print(f"查询: {query}")

        # BM25 结果
        bm25_results = bm25_retriever.invoke(query)
        print(f"  BM25:   {', '.join(doc.page_content[:20] for doc in bm25_results)}")

        # 向量检索结果
        vec_results = vector_retriever.invoke(query)
        print(f"  向量:   {', '.join(doc.page_content[:20] for doc in vec_results)}")

        # 混合结果
        ens_results = ensemble_retriever.invoke(query)
        seen = []
        for doc in ens_results:
            if doc.metadata["id"] not in [d.metadata["id"] for d in seen]:
                seen.append(doc)
        print(f"  混合:   {', '.join(doc.page_content[:20] for doc in seen)}")
        print()

    print("对比: BM25 擅长精确匹配，向量检索擅长语义理解，混合后两者互补")
    print()


if __name__ == "__main__":
    demo_multi_query()
    demo_ensemble_retriever()

    print("=" * 50)
    print("Demo 7-2 完成!")
    print()
    print("进阶检索策略:")
    print("  MultiQueryRetriever - 多角度重写查询，提高召回率")
    print("  EnsembleRetriever   - BM25 + 向量混合，互补优势")
    print("  选择建议: 召回不足用 MultiQuery，精确+语义用 Ensemble")
    print("=" * 50)
