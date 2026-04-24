"""
第32章 Demo 2：语义缓存实战

演示带监控的语义缓存实现。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key=None, model="text-embedding-v1"):
        self.api_key = api_key or os.environ["QWEN_API_KEY"]
        self.model = model
        import dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        from dashscope import TextEmbedding
        texts = [str(text) for text in texts]
        all_embeddings = []
        for text in texts:
            response = TextEmbedding.call(model=self.model, input=text)
            if response.status_code == 200:
                all_embeddings.append(response.output['embeddings'][0]['embedding'])
            else:
                all_embeddings.append([0.0] * 1536)
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


embeddings = DashScopeEmbeddings(api_key=QWEN_API_KEY)


def cosine_similarity(a, b):
    """计算余弦相似度"""
    from numpy import dot, norm
    return dot(a, b) / (norm(a) * norm(b))


class MonitoredSemanticCache:
    """带监控的语义缓存"""

    def __init__(self, threshold=0.90, ttl=3600):
        self.threshold = threshold
        self.ttl = ttl
        self.cache = {}

        # 监控指标
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, query: str) -> str | None:
        self.total_requests += 1
        query_embedding = embeddings.embed_query(query)

        for cached_embedding, (answer, timestamp) in self.cache.items():
            if time.time() - timestamp > self.ttl:
                continue

            similarity = cosine_similarity(query_embedding, cached_embedding)

            if similarity >= self.threshold:
                self.cache_hits += 1
                return answer

        self.cache_misses += 1
        return None

    def set(self, query: str, answer: str):
        query_embedding = embeddings.embed_query(query)
        self.cache[query_embedding] = (answer, time.time())

    def get_metrics(self) -> dict:
        hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


def demo_cache_with_monitoring():
    """带监控的缓存演示"""
    print("=" * 60)
    print(f"Demo 32-2: 带监控的语义缓存 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    cache = MonitoredSemanticCache(threshold=0.90, ttl=3600)

    # 测试查询
    queries = [
        ("产品价格是多少？", "初次查询"),
        ("价格是多少？", "相似查询"),
        ("产品多少钱？", "另一个相似查询"),
        ("退货政策是什么？", "新话题"),
        ("怎么退货？", "相似查询"),
    ]

    print("缓存测试：")
    print("-" * 60)

    for query, note in queries:
        cached_answer = cache.get(query)

        if cached_answer:
            print(f"查询: {query}")
            print(f"命中缓存! ({note})")
            print(f"答案: {cached_answer[:30]}...")
        else:
            # 模拟 LLM 调用
            print(f"查询: {query}")
            print(f"未命中缓存 ({note})，调用 LLM...")
            answer = llm.invoke(query).content
            cache.set(query, answer)
            print(f"答案: {answer[:30]}...")

        print()

    # 统计
    metrics = cache.get_metrics()
    print("缓存统计：")
    print("-" * 60)
    print(f"总请求: {metrics['total_requests']}")
    print(f"命中数: {metrics['cache_hits']}")
    print(f"未命中: {metrics['cache_misses']}")
    print(f"命中率: {metrics['hit_rate']:.2%}")
    print(f"缓存大小: {metrics['cache_size']}")


if __name__ == "__main__":
    demo_cache_with_monitoring()

    print("=" * 60)
    print("Demo 32-2 完成!")
    print()
    print("语义缓存监控指标：")
    print("  - total_requests: 总请求数")
    print("  - cache_hits: 命中数")
    print("  - cache_misses: 未命中数")
    print("  - hit_rate: 命中率")
    print()
    print("优化建议：")
    print("  - 命中率 < 30%: 降低阈值（如 0.85）")
    print("  - 命中率 > 80%: 可提高阈值保证准确性")
    print("=" * 60)