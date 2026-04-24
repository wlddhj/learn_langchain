"""
第30章 Demo 2：检索引擎与答案生成

演示向量检索 + RAG 答案生成。
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

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


class DashScopeEmbeddings(Embeddings):
    """通义千问 Embeddings"""

    def __init__(self, api_key=None, model="text-embedding-v1"):
        self.api_key = api_key or os.environ["QWEN_API_KEY"]
        self.model = model
        import dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        from dashscope import TextEmbedding
        texts = [str(text) for text in texts]
        batch_size = 25
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            for text in batch:
                response = TextEmbedding.call(model=self.model, input=text)
                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    batch_embeddings.append(embedding)
                else:
                    batch_embeddings.append([0.0] * 1536)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


embeddings = DashScopeEmbeddings(api_key=QWEN_API_KEY)


def build_knowledge_base():
    """构建知识库"""
    docs = [
        Document(page_content="Python 是一种解释型高级编程语言，以简洁的语法著称。由 Guido van Rossum 于 1991 年发布。", metadata={"source": "python.txt"}),
        Document(page_content="Python 主要应用领域包括：Web开发、数据科学、机器学习、自动化脚本。", metadata={"source": "python_apps.txt"}),
        Document(page_content="Django 和 Flask 是 Python 最流行的 Web 开发框架。", metadata={"source": "web_frameworks.txt"}),
        Document(page_content="NumPy 和 Pandas 是 Python 数据科学的核心库。", metadata={"source": "data_libs.txt"}),
        Document(page_content="LangChain 是用于开发 LLM 应用的 Python 框架，支持链式调用和 Agent。", metadata={"source": "langchain.txt"}),
        Document(page_content="RAG 是检索增强生成技术，结合知识检索和 LLM 生成能力。", metadata={"source": "rag.txt"}),
    ]
    return docs


def demo_vector_search():
    """向量检索演示"""
    print("=" * 60)
    print(f"Demo 30-2 (1/3): 向量检索 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    docs = build_knowledge_base()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 检索测试
    queries = [
        "Python 适合做什么？",
        "Web 开发用什么框架？",
        "什么是 RAG？",
    ]

    print("向量检索测试：")
    print("-" * 60)

    for query in queries:
        results = vectorstore.similarity_search(query, k=2)

        print(f"查询: {query}")
        print(f"结果:")
        for i, doc in enumerate(results):
            print(f"  [{i+1}] {doc.page_content[:50]}...")
        print()


def demo_rag_answer():
    """RAG 答案生成"""
    print("=" * 60)
    print(f"Demo 30-2 (2/3): RAG 答案生成 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    docs = build_knowledge_base()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    question = "Python 可以用来做 Web 开发吗？"

    # 检索
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print("问题:", question)
    print()
    print("检索到的上下文:")
    print("-" * 60)
    for i, doc in enumerate(retrieved_docs):
        print(f"[来源 {i+1}] {doc.page_content}")
    print()

    # 生成答案
    rag_prompt = ChatPromptTemplate.from_template("""
基于以下参考资料回答问题：

{context}

问题：{question}

要求：
1. 使用参考资料中的信息
2. 如果资料中没有相关信息，请说明
3. 不要编造内容
""")

    answer = (rag_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
    })

    print("生成的答案:")
    print("-" * 60)
    print(answer)


def demo_with_source_citation():
    """带来源引用的答案"""
    print("=" * 60)
    print(f"Demo 30-2 (3/3): 带来源引用的答案 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    docs = build_knowledge_base()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    question = "介绍一下 Python"

    retrieved_docs = retriever.invoke(question)

    # 格式化上下文带来源
    context_with_source = "\n\n".join([
        f"[来源: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    cite_prompt = ChatPromptTemplate.from_template("""
基于以下参考资料回答问题，并引用来源：

{context}

问题：{question}

回答格式：
【回答内容】
[具体回答，并标注引用来源]

【来源引用】
- 来源1: [内容摘要]
- 来源2: [内容摘要]
""")

    answer = (cite_prompt | llm | StrOutputParser()).invoke({
        "context": context_with_source,
        "question": question,
    })

    print("问题:", question)
    print()
    print("答案:")
    print("-" * 60)
    print(answer)


if __name__ == "__main__":
    demo_vector_search()
    demo_rag_answer()
    demo_with_source_citation()

    print("=" * 60)
    print("Demo 30-2 完成!")
    print()
    print("知识问答系统核心流程：")
    print("  1. 文档加载 → Document 对象")
    print("  2. 向量嵌入 → Embeddings")
    print("  3. 存入向量库 → FAISS")
    print("  4. 检索相关文档 → similarity_search")
    print("  5. 构建 Prompt → 上下文 + 问题")
    print("  6. LLM 生成答案 → 最终输出")
    print()
    print("来源引用的重要性：")
    print("  - 用户可验证答案来源")
    print("  - 减少幻觉风险")
    print("  - 增强可信度")
    print("=" * 60)