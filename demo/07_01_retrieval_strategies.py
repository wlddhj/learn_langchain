"""
第7章 Demo 1：检索策略对比 —— Similarity vs MMR vs 阈值过滤

演示三种基础检索策略的区别和适用场景。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


class DashScopeEmbeddings(Embeddings):
    """通义千问 Embeddings 实现"""

    def __init__(self, api_key=None, model="text-embedding-v1"):
        self.api_key = api_key or os.environ["QWEN_API_KEY"]
        self.model = model

        # 配置 dashscope
        import dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        from dashscope import TextEmbedding

        # 确保所有文本都是字符串
        texts = [str(text) for text in texts]

        # 分批处理，每批最多25个文本
        batch_size = 25
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 调用 API - 每个文本单独调用
            batch_embeddings = []
            for text in batch:
                response = TextEmbedding.call(
                    model=self.model,
                    input=text  # 直接传入单个字符串
                )

                # 检查响应
                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    batch_embeddings.append(embedding)
                else:
                    print(f"Embedding API 错误: {response.code} - {response.message}")
                    # 返回零向量作为备选
                    zero_dim = 1536
                    batch_embeddings.append([0.0] * zero_dim)

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


# 使用自定义 embeddings
embeddings = DashScopeEmbeddings(api_key=QWEN_API_KEY)


def build_vectorstore():
    """构建测试用向量库"""
    docs = [
        Document(page_content="机器学习是人工智能的一个分支，通过数据训练模型来完成任务。", metadata={"topic": "ml"}),
        Document(page_content="深度学习是机器学习的子集，使用神经网络处理复杂模式。", metadata={"topic": "dl"}),
        Document(page_content="强化学习是一种通过奖励信号学习策略的机器学习方法。", metadata={"topic": "rl"}),
        Document(page_content="监督学习使用标注数据训练模型，如分类和回归任务。", metadata={"topic": "supervised"}),
        Document(page_content="无监督学习从无标注数据中发现模式，如聚类和降维。", metadata={"topic": "unsupervised"}),
        Document(page_content="卷积神经网络（CNN）擅长图像处理，广泛应用于计算机视觉。", metadata={"topic": "cnn"}),
        Document(page_content="循环神经网络（RNN）擅长序列数据处理，如自然语言处理。", metadata={"topic": "rnn"}),
        Document(page_content="Transformer 架构基于注意力机制，是 GPT 和 BERT 的基础。", metadata={"topic": "transformer"}),
        Document(page_content="迁移学习利用预训练模型的知识，在小数据集上也能获得好效果。", metadata={"topic": "transfer"}),
        Document(page_content="Python 是 AI 开发最常用的语言，有丰富的库如 PyTorch 和 TensorFlow。", metadata={"topic": "python"}),
    ]
    return FAISS.from_documents(docs, embeddings)


def demo_similarity():
    """Similarity：返回最相似的 k 个文档"""
    print("=" * 50)
    print("Demo 7-1 (1/3): Similarity 检索")
    print("=" * 50)
    print("特点: 返回与查询最相似的 k 个文档（默认策略）")
    print()

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    query = "什么是神经网络？"
    results = retriever.invoke(query)
    print(f"查询: {query}")
    print(f"返回 {len(results)} 条结果:")
    for doc in results:
        print(f"  [{doc.metadata['topic']}] {doc.page_content}")
    print()


def demo_mmr():
    """MMR：在相关性和多样性之间平衡"""
    print("=" * 50)
    print("Demo 7-1 (2/3): MMR 检索 (最大边际相关性)")
    print("=" * 50)
    print("特点: 在相关性和多样性之间平衡，避免结果过于重复")
    print()

    vectorstore = build_vectorstore()

    # 对比 similarity 和 mmr
    query = "机器学习方法"
    print(f"查询: {query}\n")

    # Similarity（可能返回很多相似内容）
    sim_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    print("--- Similarity (k=5) ---")
    sim_results = sim_retriever.invoke(query)
    for doc in sim_results:
        print(f"  [{doc.metadata['topic']}] {doc.page_content[:50]}...")

    print()

    # MMR（结果更多样）
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
    )
    print("--- MMR (k=5, fetch_k=10, lambda=0.5) ---")
    mmr_results = mmr_retriever.invoke(query)
    for doc in mmr_results:
        print(f"  [{doc.metadata['topic']}] {doc.page_content[:50]}...")

    print()
    print("对比: MMR 的结果覆盖更多不同话题，而 Similarity 可能返回重复内容")
    print()


def demo_score_threshold():
    """相似度阈值过滤"""
    print("=" * 50)
    print("Demo 7-1 (3/3): 相似度阈值过滤")
    print("=" * 50)
    print("特点: 只返回相似度高于阈值的文档")
    print()

    vectorstore = build_vectorstore()

    queries = [
        "深度学习和机器学习",
        "Python 的装饰器怎么用？",  # 语义距离较远
    ]

    for query in queries:
        # 带分数检索
        results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
        print(f"查询: {query}")
        for doc, score in results_with_scores:
            print(f"  [{doc.metadata['topic']}] 分数={score:.4f} → {doc.page_content[:40]}...")
        print()

    print("说明: FAISS 的 L2 距离越小越相似")
    print()


if __name__ == "__main__":
    demo_similarity()
    demo_mmr()
    demo_score_threshold()

    print("=" * 50)
    print("Demo 7-1 完成!")
    print()
    print("检索策略选择:")
    print("  similarity  - 简单场景，返回最相似的文档")
    print("  mmr         - 结果重复时，兼顾多样性和相关性")
    print("  score阈值   - 只返回足够相关的文档")
    print("=" * 50)
