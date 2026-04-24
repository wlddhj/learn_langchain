"""
第32章 Demo 1：语义缓存基础

演示语义缓存的基本原理和实现。
"""

import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def demo_cache_comparison():
    """传统缓存 vs 语义缓存对比"""
    print("=" * 60)
    print("Demo 32-1: 语义缓存基础")
    print("=" * 60)
    print()

    print("传统缓存 vs 语义缓存：")
    print("-" * 60)
    print("""
| 特性 | 传统缓存 | 语义缓存 |
|------|---------|---------|
| 匹配方式 | 精确字符串匹配 | 语义相似度匹配 |
| 灵活性 | 低（必须完全相同） | 高（语义相似即可） |
| 命中率 | 低 | 高 |
| 适用场景 | 固定查询 | 自然语言查询 |
| 存储方式 | Key-Value | Vector + Metadata |
""")
    print()

    print("适用场景：")
    print("  ✅ 高价值场景：")
    print("     - 客服 FAQ 问答（常见问题重复）")
    print("     - 知识库问答（固定知识范围）")
    print("     - 产品咨询（标准化问题）")
    print()
    print("  ❌ 低价值场景：")
    print("     - 创意写作（每次需求不同）")
    print("     - 个性化对话（用户偏好差异）")
    print("     - 实时信息查询（信息变化快）")


def demo_semantic_cache_flow():
    """语义缓存工作流程"""
    print("=" * 60)
    print("Demo 32-1 (2/4): 工作流程")
    print("=" * 60)
    print()

    print("语义缓存工作流程：")
    print("-" * 60)
    print("""
用户查询
    ↓
生成查询向量 (Embedding)
    ↓
检索缓存向量库
    ↓
相似度计算 (Cosine Similarity)
    ↓
┌─────────────────────┐
│ 相似度 >= 阈值      │ → 返回缓存结果
│ 相似度 < 阈值       │ → 调用 LLM → 缓存结果 → 返回
└─────────────────────┘
""")
    print()


def demo_threshold_effect():
    """阈值参数说明"""
    print("=" * 60)
    print("Demo 32-1 (3/4): 阈值参数")
    print("=" * 60)
    print()

    print("核心参数说明：")
    print("-" * 60)
    print("""
| 参数 | 说明 | 建议值 |
|------|------|--------|
| 相似度阈值 | 值越高匹配越严格 | 0.85-0.95 |
| 向量维度 | Embedding 维度 | 1536 (OpenAI) |
| 缓存 TTL | 缓存有效期 | 1-24 小时 |
| 最大缓存数 | 缓存条目上限 | 根据内存设置 |

阈值选择建议：
- FAQ 类应用: 0.92（高阈值保证准确性）
- 客服应用: 0.88（平衡命中率和准确性）
- 知识问答: 0.85（提高命中率）
""")
    print()


def demo_simple_cache_implementation():
    """简单缓存实现示意"""
    print("=" * 60)
    print("Demo 32-1 (4/4): 简单实现示意")
    print("=" * 60)
    print()

    code = """
class SemanticCache:
    def __init__(self, threshold=0.95, ttl=3600):
        self.threshold = threshold
        self.ttl = ttl
        self.cache = {}  # embedding -> (answer, timestamp)

    def get(self, query: str) -> str | None:
        # 1. 生成查询向量
        query_embedding = embeddings.embed_query(query)

        # 2. 检查缓存相似度
        for cached_embedding, (answer, timestamp) in self.cache.items():
            # TTL 检查
            if time.time() - timestamp > self.ttl:
                continue

            # 相似度计算
            similarity = cosine_similarity(query_embedding, cached_embedding)

            # 阈值判断
            if similarity >= self.threshold:
                return answer  # 缓存命中

        return None  # 未命中

    def set(self, query: str, answer: str):
        # 缓存结果
        query_embedding = embeddings.embed_query(query)
        self.cache[query_embedding] = (answer, time.time())
"""
    print("简单语义缓存实现：")
    print("-" * 60)
    print(code)


if __name__ == "__main__":
    demo_cache_comparison()
    demo_semantic_cache_flow()
    demo_threshold_effect()
    demo_simple_cache_implementation()

    print("=" * 60)
    print("Demo 32-1 完成!")
    print()
    print("语义缓存核心要点：")
    print("  - 基于语义相似度而非精确匹配")
    print("  - 命中率比传统缓存高")
    print("  - 适合自然语言查询场景")
    print()
    print("关键参数：")
    print("  - threshold: 0.85-0.95")
    print("  - ttl: 1-24 小时")
    print("=" * 60)