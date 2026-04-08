"""
第6章 Demo 2：完整 RAG 管道

演示向量化、向量存储、相似度检索、完整 RAG chain（检索 + 生成）。
可独立运行，使用 Qwen 模型 + FAISS 向量库。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

# 使用 Qwen 的 Embedding 模型（通过 OpenAI 兼容接口）
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)


# ============================================================
# 1. 准备知识库数据
# ============================================================

def create_knowledge_base():
    """创建示例知识文档"""
    texts = [
        "LangChain 是一个用于开发大语言模型应用的开源框架。它提供了模块化的组件，包括 Model I/O、Retrieval、Chains、Agents 和 Memory。",
        "RAG（检索增强生成）结合了信息检索和文本生成。工作流程：检索相关文档 → 作为上下文传给 LLM → 生成回答。",
        "AI Agent 是能自主决策的智能体，通过工具调用与外部交互。核心模式是 ReAct：推理 → 行动 → 观察 → 循环。",
        "Embeddings 将文本转换为固定维度的向量，用于计算语义相似度。常见的 Embedding 模型有 text-embedding-ada-002、text-embedding-v3 等。",
        "向量数据库（Vector Store）存储文本向量的数据库，支持相似度检索。常用的有 Chroma、FAISS、Milvus、Pinecone 等。",
        "LangGraph 是 LangChain 的扩展框架，用于构建有状态的、多角色的 Agent 应用。基于图结构，支持循环和条件分支。",
        "Prompt 模板（PromptTemplate）让 prompt 可复用，支持变量注入和组合。ChatPromptTemplate 是最常用的对话模板。",
        "Output Parser 将 LLM 的文本输出转换为结构化数据。常用 StrOutputParser、JsonOutputParser、PydanticOutputParser。",
        "LCEL（LangChain Expression Language）使用管道操作符 | 连接组件，自动支持 invoke/batch/stream。",
        "多 Agent 系统通过 Supervisor 或 Swarm 模式协调多个专业 Agent。每个 Agent 有自己的 prompt 和工具。",
    ]
    docs = [Document(page_content=t, metadata={"source": f"doc_{i}"}) for i, t in enumerate(texts)]
    return docs


# ============================================================
# 2. Demo: 向量存储与检索
# ============================================================

def demo_vector_store():
    print("=" * 50)
    print(f"Demo 6-2 (1/3): 向量存储与检索 [{QWEN_MODEL}]")
    print("=" * 50)

    docs = create_knowledge_base()

    # 分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    print(f"文档数: {len(docs)}, 切分后: {len(chunks)}")

    # 创建 FAISS 向量库
    print("正在向量化并创建索引...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("向量库创建完成!")
    print()

    # 相似度检索
    queries = ["什么是 RAG？", "LangChain 有哪些核心组件？", "如何构建多 Agent 系统？"]

    for q in queries:
        results = vectorstore.similarity_search(q, k=2)
        print(f"问题: {q}")
        for doc in results:
            print(f"  → {doc.page_content[:80]}...")
        print()

    return vectorstore


# ============================================================
# 3. Demo: 完整 RAG Chain
# ============================================================

def demo_rag_chain(vectorstore):
    print("=" * 50)
    print(f"Demo 6-2 (2/3): 完整 RAG Chain [{QWEN_MODEL}]")
    print("=" * 50)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""基于以下上下文回答用户问题。如果上下文中没有相关信息，请说"根据已知信息无法回答"。

上下文：
{context}

问题：{question}

回答：""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "什么是 RAG？它有什么优势？",
        "LCEL 是什么？",
        "Python 的 GIL 是什么？",  # 知识库中没有的内容
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")
        print()

    return rag_chain


# ============================================================
# 4. Demo: 带分数的检索
# ============================================================

def demo_similarity_score(vectorstore):
    print("=" * 50)
    print("Demo 6-2 (3/3): 相似度分数")
    print("=" * 50)

    query = "什么是 Embeddings？"
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f"查询: {query}")
    print()
    for doc, score in results_with_scores:
        print(f"  分数: {score:.4f}")
        print(f"  内容: {doc.page_content[:80]}...")
        print()

    print("说明: 分数越低表示越相似（L2 距离）")
    print()


if __name__ == "__main__":
    vectorstore = demo_vector_store()
    demo_rag_chain(vectorstore)
    demo_similarity_score(vectorstore)

    print("=" * 50)
    print("Demo 6-2 完成!")
    print()
    print("RAG 完整流程:")
    print("  加载文档 → 文本分割 → 向量化 → 存入 FAISS")
    print("  查询 → 检索相关文档 → 注入 prompt → LLM 生成回答")
    print("=" * 50)
