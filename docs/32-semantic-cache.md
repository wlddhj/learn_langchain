# 第32章：语义缓存深入

## 32.1 为什么需要语义缓存

LLM API 调用成本高、延迟大，通过缓存可以显著降低成本和提升响应速度。

### 传统缓存 vs 语义缓存

| 特性 | 传统缓存 | 语义缓存 |
|------|---------|---------|
| **匹配方式** | 精确字符串匹配 | 语义相似度匹配 |
| **灵活性** | 低（必须完全相同） | 高（语义相似即可） |
| **命中率** | 低 | 高 |
| **适用场景** | 固定查询 | 自然语言查询 |
| **存储方式** | Key-Value | Vector + Metadata |

### 适用场景

```
高价值场景：
- 客服 FAQ 问答（常见问题重复）
- 知识库问答（固定知识范围）
- 产品咨询（标准化问题）
- API 封装（重复请求）

低价值场景：
- 创意写作（每次需求不同）
- 个性化对话（用户偏好差异）
- 实时信息查询（信息变化快）
```

## 32.2 语义缓存原理

### 工作流程

```
用户查询
    ↓
生成查询向量
    ↓
检索缓存向量库
    ↓
相似度计算（阈值判断）
    ↓
┌─────────────────────┐
│ 相似度 >= 阈值      │ → 返回缓存结果
│ 相似度 < 阈值       │ → 调用 LLM → 缓存结果 → 返回
└─────────────────────┘
```

### 核心参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| **相似度阈值** | 值越高匹配越严格 | 0.85-0.95 |
| **向量维度** | Embedding 维度 | 1536 (OpenAI) |
| **缓存 TTL** | 缓存有效期 | 1-24 小时 |
| **最大缓存数** | 缓存条目上限 | 根据内存设置 |

## 32.3 基础语义缓存实现

### 简单语义缓存

```python
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from numpy import dot
from numpy.linalg import norm
import time

class SemanticCache:
    """简单语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.95,
        ttl: int = 3600,  # 秒
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        
        # 缓存存储
        self.cache = {}  # query_embedding -> (answer, timestamp)
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        query_embedding = self.embeddings.embed_query(query)
        
        # 查找相似缓存
        for cached_embedding, (answer, timestamp) in self.cache.items():
            # 检查 TTL
            if time.time() - timestamp > self.ttl:
                continue
            
            # 计算相似度
            similarity = dot(query_embedding, cached_embedding) / (
                norm(query_embedding) * norm(cached_embedding)
            )
            
            if similarity >= self.threshold:
                return answer
        
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        query_embedding = self.embeddings.embed_query(query)
        self.cache[query_embedding] = (answer, time.time())
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def stats(self) -> dict:
        """缓存统计"""
        return {
            "cache_size": len(self.cache),
            "threshold": self.threshold,
            "ttl": self.ttl,
        }


# 使用示例
cache = SemanticCache(threshold=0.95, ttl=3600)
llm = ChatOpenAI(model="gpt-4o-mini")

def cached_chat(query: str) -> str:
    """带缓存的聊天"""
    # 检查缓存
    cached_answer = cache.get(query)
    
    if cached_answer:
        print("命中缓存！")
        return cached_answer
    
    # 调用 LLM
    print("调用 LLM...")
    answer = llm.invoke(query).content
    
    # 缓存结果
    cache.set(query, answer)
    
    return answer

# 测试
print(cached_chat("产品退货需要什么条件？"))
print(cached_chat("退货条件是什么？"))  # 应命中缓存
print(cache.stats())
```

### 基于向量库的语义缓存

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time

class VectorSemanticCache:
    """基于向量库的语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.90,
        ttl: int = 3600,
        persist_dir: str = "./cache_db",
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        
        # 创建向量库
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        # 检索最相似的缓存
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=1,
        )
        
        if not results:
            return None
        
        doc, distance = results[0]
        
        # 转换距离为相似度（Cosine similarity）
        similarity = 1 - distance
        
        # 检查阈值和 TTL
        timestamp = doc.metadata.get("timestamp", 0)
        
        if similarity >= self.threshold and time.time() - timestamp <= self.ttl:
            return doc.page_content
        
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        doc = Document(
            page_content=answer,
            metadata={
                "query": query,
                "timestamp": time.time(),
            },
        )
        
        self.vectorstore.add_documents([doc])
    
    def clear_expired(self):
        """清理过期缓存"""
        # 获取所有文档
        all_docs = self.vectorstore.get()
        
        expired_ids = []
        for i, doc_id in enumerate(all_docs["ids"]):
            timestamp = all_docs["metadatas"][i].get("timestamp", 0)
            
            if time.time() - timestamp > self.ttl:
                expired_ids.append(doc_id)
        
        # 删除过期缓存
        if expired_ids:
            self.vectorstore.delete(expired_ids)
    
    def stats(self) -> dict:
        """缓存统计"""
        return {
            "cache_size": self.vectorstore._collection.count(),
            "threshold": self.threshold,
            "ttl": self.ttl,
        }


# 使用示例
cache = VectorSemanticCache(threshold=0.90, ttl=3600)
llm = ChatOpenAI(model="gpt-4o-mini")

def cached_rag(query: str) -> str:
    """带缓存的 RAG"""
    cached_answer = cache.get(query)
    
    if cached_answer:
        return cached_answer
    
    # 实际 RAG 调用
    answer = llm.invoke(query).content
    cache.set(query, answer)
    
    return answer

# 测试
print(cached_rag("产品价格是多少？"))
print(cached_rag("价格是什么？"))  # 相似查询

# 定期清理
cache.clear_expired()
print(cache.stats())
```

## 32.4 LangChain 集成缓存

### 使用 LangChain 内置缓存

```python
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from langchain_core.caches import BaseCache

# 1. 内存缓存（精确匹配）
llm = ChatOpenAI(model="gpt-4o-mini")
llm.cache = InMemoryCache()

# 相同输入会缓存
response1 = llm.invoke("你好")
response2 = llm.invoke("你好")  # 从缓存返回

# 2. SQLite 缓存（持久化）
from langchain.cache import SQLiteCache

llm.cache = SQLiteCache(database_path=".langchain_cache.db")

# 3. Redis 缓存
from langchain_community.cache import RedisCache
from redis import Redis

redis_cache = RedisCache(redis_=Redis(host="localhost", port=6379))
llm.cache = redis_cache

# 4. GPTCache（语义缓存）
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.cache import GPTCache

def init_gptcache():
    cache = Cache()
    init_similar_cache(
        cache_obj=cache,
        embedding_func="openai",  # 使用 OpenAI Embedding
    )
    return cache

llm.cache = GPTCache(init_gptcache)

# 5. 自定义缓存
class CustomCache(BaseCache):
    def __init__(self):
        self.cache_data = {}
    
    def lookup(self, prompt, llm_string):
        key = (prompt, llm_string)
        return self.cache_data.get(key)
    
    def update(self, prompt, llm_string, return_val):
        key = (prompt, llm_string)
        self.cache_data[key] = return_val
    
    def clear(self):
        self.cache_data.clear()

llm.cache = CustomCache()
```

### GPTCache 详细配置

```python
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.adapter.langchain_models import LangChainChatOpenAI
from gptcache.embedding import OpenAI
from gptcache.similarity_evaluation import SearchDistanceEvaluation
from gptcache.manager import get_data_manager
from langchain_openai import ChatOpenAI

# 详细配置
cache = Cache()

# Embedding 配置
embedding = OpenAI(model="text-embedding-3-small")

# 相似度评估
similarity_eval = SearchDistanceEvaluation(max_distance=0.2)

# 数据管理（向量存储）
data_manager = get_data_manager(
    "sqlite",  # 本地 SQLite
    "chroma",  # Chroma 向量库
    vector_params={"dimension": 1536},
)

# 初始化缓存
cache.init(
    embedding_func=embedding.to_embeddings,
    similarity_evaluation=similarity_eval,
    data_manager=data_manager,
)

# 使用
llm = LangChainChatOpenAI(
    model="gpt-4o-mini",
    cache=cache,
)

# 相似查询会命中缓存
response1 = llm.invoke("产品价格是多少？")
response2 = llm.invoke("价格是多少？")  # 可能命中缓存
```

## 32.5 高级语义缓存

### 分层缓存策略

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import time

class TieredSemanticCache:
    """分层语义缓存"""
    
    def __init__(
        self,
        thresholds: list[float] = [0.95, 0.85],
        ttl_tiers: list[int] = [7200, 3600],
    ):
        self.embeddings = OpenAIEmbeddings()
        self.thresholds = thresholds  # 不同阈值层级
        self.ttl_tiers = ttl_tiers  # 不同 TTL
        
        # 多级缓存
        self.cache_tiers = [
            {} for _ in range(len(thresholds))
        ]
    
    def get(self, query: str) -> tuple[str | None, int]:
        """获取缓存（返回结果和命中层级）"""
        query_embedding = self.embeddings.embed_query(query)
        
        for tier_idx, threshold in enumerate(self.thresholds):
            for cached_embedding, (answer, timestamp) in self.cache_tiers[tier_idx].items():
                if time.time() - timestamp > self.ttl_tiers[tier_idx]:
                    continue
                
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity >= threshold:
                    return answer, tier_idx
        
        return None, -1
    
    def set(self, query: str, answer: str, tier: int = 0):
        """设置缓存到指定层级"""
        query_embedding = self.embeddings.embed_query(query)
        
        self.cache_tiers[tier][query_embedding] = (answer, time.time())
    
    def auto_tier_set(self, query: str, answer: str, confidence: float):
        """根据置信度自动选择层级"""
        # 高置信度 -> 高层级（更严格匹配）
        if confidence >= 0.9:
            tier = 0
        elif confidence >= 0.7:
            tier = 1
        else:
            tier = 1
        
        self.set(query, answer, tier)
    
    def _cosine_similarity(self, a, b):
        from numpy import dot, norm
        return dot(a, b) / (norm(a) * norm(b))
    
    def stats(self) -> dict:
        return {
            "tier_sizes": [len(tier) for tier in self.cache_tiers],
            "thresholds": self.thresholds,
            "ttl_tiers": self.ttl_tiers,
        }


# 使用示例
cache = TieredSemanticCache(
    thresholds=[0.95, 0.85],
    ttl_tiers=[7200, 3600],
)

llm = ChatOpenAI(model="gpt-4o-mini")

def tiered_cached_chat(query: str) -> str:
    cached_answer, tier = cache.get(query)
    
    if cached_answer:
        print(f"命中层级 {tier} 缓存")
        return cached_answer
    
    print("调用 LLM...")
    answer = llm.invoke(query).content
    
    # 自动选择层级
    cache.auto_tier_set(query, answer, confidence=0.8)
    
    return answer

# 测试
print(tiered_cached_chat("产品价格？"))
print(tiered_cached_chat("价格是多少？"))
print(cache.stats())
```

### 按类别缓存

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

class CategorySemanticCache:
    """分类语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.90,
        ttl: int = 3600,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        
        # 按类别存储缓存
        self.category_caches = {}
    
    def get(self, query: str, category: str) -> str | None:
        """获取指定类别的缓存"""
        if category not in self.category_caches:
            return None
        
        cache = self.category_caches[category]
        query_embedding = self.embeddings.embed_query(query)
        
        for cached_embedding, (answer, timestamp) in cache.items():
            if time.time() - timestamp > self.ttl:
                continue
            
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                return answer
        
        return None
    
    def set(self, query: str, answer: str, category: str):
        """设置指定类别缓存"""
        if category not in self.category_caches:
            self.category_caches[category] = {}
        
        query_embedding = self.embeddings.embed_query(query)
        self.category_caches[category][query_embedding] = (answer, time.time())
    
    def _cosine_similarity(self, a, b):
        from numpy import dot, norm
        return dot(a, b) / (norm(a) * norm(b))
    
    def stats(self) -> dict:
        return {
            "categories": list(self.category_caches.keys()),
            "category_sizes": {
                cat: len(cache) for cat, cache in self.category_caches.items()
            },
        }


# 使用示例
cache = CategorySemanticCache(threshold=0.90)
llm = ChatOpenAI(model="gpt-4o-mini")

# 分类器
def classify_query(query: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
    分类以下问题：
    {query}
    
    类别：product/order/tech/other
    只输出类别名称。
    """)
    
    return (prompt | llm).invoke({"query": query}).content

def category_cached_chat(query: str) -> str:
    # 分类
    category = classify_query(query)
    
    # 检查缓存
    cached_answer = cache.get(query, category)
    
    if cached_answer:
        print(f"命中 {category} 类别缓存")
        return cached_answer
    
    # 调用 LLM
    print(f"调用 LLM，类别: {category}")
    answer = llm.invoke(query).content
    
    # 缓存
    cache.set(query, answer, category)
    
    return answer

# 测试
print(category_cached_chat("产品价格？"))  # 类别: product
print(category_cached_chat("订单状态？"))  # 类别: order
print(cache.stats())
```

### 缓存预热

```python
class PreheatedSemanticCache:
    """带预热的语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.90,
        ttl: int = 3600,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        self.cache = {}
    
    def preheat(self, faq_pairs: list[tuple[str, str]]):
        """预热缓存"""
        print(f"预热 {len(faq_pairs)} 个常见问答...")
        
        for question, answer in faq_pairs:
            self.set(question, answer)
        
        print("预热完成")
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        query_embedding = self.embeddings.embed_query(query)
        
        for cached_embedding, (answer, timestamp) in self.cache.items():
            if time.time() - timestamp > self.ttl:
                continue
            
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                return answer
        
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        query_embedding = self.embeddings.embed_query(query)
        self.cache[query_embedding] = (answer, time.time())
    
    def _cosine_similarity(self, a, b):
        from numpy import dot, norm
        return dot(a, b) / (norm(a) * norm(b))
    
    def stats(self) -> dict:
        return {"cache_size": len(self.cache)}


# 使用示例
cache = PreheatedSemanticCache(threshold=0.90)

# FAQ 预热
faq_pairs = [
    ("产品价格是多少？", "产品价格 99 元。"),
    ("退货政策？", "7天内可退货。"),
    ("联系方式？", "客服电话：400-123-4567"),
    ("发货时间？", "下单后 24 小时内发货。"),
    ("支付方式？", "支持支付宝、微信支付。"),
]

cache.preheat(faq_pairs)

llm = ChatOpenAI(model="gpt-4o-mini")

def faq_chat(query: str) -> str:
    cached_answer = cache.get(query)
    
    if cached_answer:
        return cached_answer
    
    answer = llm.invoke(query).content
    cache.set(query, answer)
    
    return answer

# 测试 - 应命中预热缓存
print(faq_chat("价格？"))
print(faq_chat("怎么退货？"))
print(cache.stats())
```

## 32.6 缓存与 RAG 结合

### RAG 结果缓存

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

class RAGSemanticCache:
    """RAG 语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.88,
        ttl: int = 3600,
        persist_dir: str = "./rag_cache_db",
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        
        # 缓存向量库
        self.cache_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
    
    def get_cached_answer(self, query: str) -> str | None:
        """获取缓存答案"""
        results = self.cache_store.similarity_search_with_score(query, k=1)
        
        if not results:
            return None
        
        doc, distance = results[0]
        similarity = 1 - distance
        
        timestamp = doc.metadata.get("timestamp", 0)
        
        if similarity >= self.threshold and time.time() - timestamp <= self.ttl:
            return doc.page_content
        
        return None
    
    def cache_answer(self, query: str, answer: str, sources: list):
        """缓存答案"""
        from langchain_core.documents import Document
        
        doc = Document(
            page_content=answer,
            metadata={
                "query": query,
                "timestamp": time.time(),
                "sources": sources,
            },
        )
        
        self.cache_store.add_documents([doc])
    
    def stats(self) -> dict:
        return {"cache_size": self.cache_store._collection.count()}


# RAG 系统
class CachedRAGSystem:
    """带缓存的 RAG 系统"""
    
    def __init__(
        self,
        knowledge_db_path: str,
        cache_threshold: float = 0.88,
    ):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings()
        
        # 知识向量库
        self.knowledge_store = Chroma(
            persist_directory=knowledge_db_path,
            embedding_function=self.embeddings,
        )
        
        # 缓存
        self.cache = RAGSemanticCache(threshold=cache_threshold)
    
    def query(self, question: str) -> dict:
        """查询"""
        # 1. 检查缓存
        cached_answer = self.cache.get_cached_answer(question)
        
        if cached_answer:
            return {
                "answer": cached_answer,
                "cached": True,
                "sources": [],
            }
        
        # 2. RAG 检索
        retriever = self.knowledge_store.as_retriever()
        docs = retriever.invoke(question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. 生成答案
        prompt = ChatPromptTemplate.from_template("""
        基于以下内容回答问题：
        {context}
        
        问题：{question}
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question,
        })
        
        # 4. 缓存结果
        sources = [doc.metadata.get("source", "") for doc in docs]
        self.cache.cache_answer(question, answer, sources)
        
        return {
            "answer": answer,
            "cached": False,
            "sources": sources,
        }
    
    def cache_stats(self) -> dict:
        return self.cache.stats()


# 使用示例
rag_system = CachedRAGSystem(
    knowledge_db_path="./knowledge_db",
    cache_threshold=0.88,
)

# 测试
result1 = rag_system.query("产品退货政策？")
print(f"答案: {result1['answer']}")
print(f"缓存命中: {result1['cached']}")

result2 = rag_system.query("退货政策是什么？")  # 相似查询
print(f"答案: {result2['answer']}")
print(f"缓存命中: {result2['cached']}")

print(rag_system.cache_stats())
```

## 32.7 缓存监控与优化

### 缓存命中率监控

```python
class MonitoredSemanticCache:
    """带监控的语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.90,
        ttl: int = 3600,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        self.cache = {}
        
        # 监控指标
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        self.total_requests += 1
        
        query_embedding = self.embeddings.embed_query(query)
        
        for cached_embedding, (answer, timestamp) in self.cache.items():
            if time.time() - timestamp > self.ttl:
                continue
            
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                self.cache_hits += 1
                return answer
        
        self.cache_misses += 1
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        query_embedding = self.embeddings.embed_query(query)
        self.cache[query_embedding] = (answer, time.time())
    
    def _cosine_similarity(self, a, b):
        from numpy import dot, norm
        return dot(a, b) / (norm(a) * norm(b))
    
    def get_metrics(self) -> dict:
        """获取监控指标"""
        hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "threshold": self.threshold,
        }
    
    def optimize_threshold(self):
        """优化阈值"""
        metrics = self.get_metrics()
        
        # 建议阈值调整
        if metrics["hit_rate"] < 0.3:
            suggestion = "命中率低，建议降低阈值（如 0.85）"
        elif metrics["hit_rate"] > 0.8:
            suggestion = "命中率高，可适当提高阈值保证准确性"
        else:
            suggestion = "阈值适中，无需调整"
        
        return {
            "metrics": metrics,
            "suggestion": suggestion,
        }


# 使用示例
cache = MonitoredSemanticCache(threshold=0.90)
llm = ChatOpenAI(model="gpt-4o-mini")

# 多次调用
queries = [
    "产品价格？",
    "价格是多少？",
    "多少钱？",
    "产品介绍",
    "介绍一下产品",
]

for q in queries:
    cached = cache.get(q)
    if cached:
        print(f"缓存命中: {q}")
    else:
        answer = llm.invoke(q).content
        cache.set(q, answer)
        print(f"调用 LLM: {q}")

# 获取指标
print(cache.get_metrics())
print(cache.optimize_threshold())
```

### 缓存淘汰策略

```python
class SemanticCacheWithEviction:
    """带淘汰策略的语义缓存"""
    
    def __init__(
        self,
        threshold: float = 0.90,
        ttl: int = 3600,
        max_size: int = 1000,
        eviction_policy: str = "lru",  # lru/lfu/fifo
    ):
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        
        self.cache = {}  # embedding -> (answer, timestamp, access_count)
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        query_embedding = self.embeddings.embed_query(query)
        
        for cached_embedding, (answer, timestamp, access_count) in self.cache.items():
            if time.time() - timestamp > self.ttl:
                continue
            
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                # 更新访问计数
                self.cache[cached_embedding] = (answer, timestamp, access_count + 1)
                return answer
        
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        # 检查是否需要淘汰
        if len(self.cache) >= self.max_size:
            self._evict()
        
        query_embedding = self.embeddings.embed_query(query)
        self.cache[query_embedding] = (answer, time.time(), 0)
    
    def _evict(self):
        """执行淘汰"""
        if self.eviction_policy == "lru":
            # 最近最少使用
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            self.cache.pop(oldest[0])
        
        elif self.eviction_policy == "lfu":
            # 最少频率使用
            least_frequent = min(self.cache.items(), key=lambda x: x[1][2])
            self.cache.pop(least_frequent[0])
        
        elif self.eviction_policy == "fifo":
            # 先进先出
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            self.cache.pop(oldest[0])
    
    def _cosine_similarity(self, a, b):
        from numpy import dot, norm
        return dot(a, b) / (norm(a) * norm(b))
    
    def stats(self) -> dict:
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "eviction_policy": self.eviction_policy,
        }


# 使用示例
cache = SemanticCacheWithEviction(
    threshold=0.90,
    max_size=100,
    eviction_policy="lfu",
)

llm = ChatOpenAI(model="gpt-4o-mini")

for i in range(150):  # 超过最大缓存数
    query = f"问题 {i}"
    cached = cache.get(query)
    
    if not cached:
        answer = llm.invoke(query).content
        cache.set(query, answer)

print(cache.stats())
```

## 32.8 分布式语义缓存

### Redis 语义缓存

```python
import redis
import json
from langchain_openai import OpenAIEmbeddings

class RedisSemanticCache:
    """Redis 分布式语义缓存"""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        threshold: float = 0.90,
        ttl: int = 3600,
    ):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        query_embedding = self.embeddings.embed_query(query)
        
        # 从 Redis 获取所有缓存
        cache_keys = self.redis.keys("semantic_cache:*")
        
        for key in cache_keys:
            cached_data = json.loads(self.redis.get(key))
            
            cached_embedding = cached_data["embedding"]
            answer = cached_data["answer"]
            timestamp = cached_data["timestamp"]
            
            # TTL 检查
            if time.time() - timestamp > self.ttl:
                self.redis.delete(key)
                continue
            
            # 相似度计算
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                return answer
        
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        query_embedding = self.embeddings.embed_query(query)
        
        key = f"semantic_cache:{hash(query)}"
        
        cached_data = {
            "query": query,
            "embedding": query_embedding,
            "answer": answer,
            "timestamp": time.time(),
        }
        
        self.redis.set(key, json.dumps(cached_data), ex=self.ttl)
    
    def _cosine_similarity(self, a, b):
        from numpy import dot, norm
        return dot(a, b) / (norm(a) * norm(b))
    
    def clear(self):
        """清空缓存"""
        keys = self.redis.keys("semantic_cache:*")
        if keys:
            self.redis.delete(*keys)
    
    def stats(self) -> dict:
        keys = self.redis.keys("semantic_cache:*")
        return {"cache_size": len(keys)}
```

### 结合 Redis 和向量库

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import redis
import json

class HybridDistributedCache:
    """混合分布式缓存"""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        vector_dir: str = "./distributed_cache",
        threshold: float = 0.90,
        ttl: int = 3600,
    ):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold
        self.ttl = ttl
        
        # 向量库（用于高效检索）
        self.vectorstore = Chroma(
            persist_directory=vector_dir,
            embedding_function=self.embeddings,
        )
    
    def get(self, query: str) -> str | None:
        """获取缓存"""
        # 使用向量库快速检索
        results = self.vectorstore.similarity_search_with_score(query, k=5)
        
        for doc, distance in results:
            similarity = 1 - distance
            
            if similarity >= self.threshold:
                cache_key = doc.metadata.get("cache_key")
                
                # 从 Redis 获取完整答案
                cached_data = self.redis.get(cache_key)
                
                if cached_data:
                    data = json.loads(cached_data)
                    
                    # TTL 检查
                    if time.time() - data["timestamp"] <= self.ttl:
                        return data["answer"]
        
        return None
    
    def set(self, query: str, answer: str):
        """设置缓存"""
        query_embedding = self.embeddings.embed_query(query)
        
        cache_key = f"semantic_cache:{hash(query)}"
        
        # 存到 Redis
        cached_data = {
            "query": query,
            "answer": answer,
            "timestamp": time.time(),
        }
        
        self.redis.set(cache_key, json.dumps(cached_data), ex=self.ttl)
        
        # 存到向量库（用于检索）
        from langchain_core.documents import Document
        
        doc = Document(
            page_content=query,
            metadata={
                "cache_key": cache_key,
                "timestamp": time.time(),
            },
        )
        
        self.vectorstore.add_documents([doc])
```

## 32.9 生产最佳实践

### 缓存配置建议

```python
# 推荐配置
RECOMMENDED_CONFIG = {
    # FAQ 类应用
    "faq": {
        "threshold": 0.92,  # 高阈值保证准确性
        "ttl": 86400,      # 24 小时
        "max_size": 5000,  # 大容量
    },
    
    # 客服应用
    "customer_service": {
        "threshold": 0.88,
        "ttl": 3600,       # 1 小时
        "max_size": 1000,
    },
    
    # 知识问答
    "knowledge_qa": {
        "threshold": 0.85,
        "ttl": 7200,       # 2 小时
        "max_size": 3000,
    },
    
    # API 服务
    "api_service": {
        "threshold": 0.90,
        "ttl": 1800,       # 30 分钟
        "max_size": 10000,
    },
}

# 根据场景选择配置
def create_cache_for_scenario(scenario: str):
    config = RECOMMENDED_CONFIG.get(scenario, RECOMMENDED_CONFIG["customer_service"])
    
    return SemanticCacheWithEviction(
        threshold=config["threshold"],
        ttl=config["ttl"],
        max_size=config["max_size"],
    )
```

### 缓存预热脚本

```python
def preheat_cache_from_faq_file(file_path: str, cache: SemanticCache):
    """从 FAQ 文件预热缓存"""
    import csv
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            question = row['question']
            answer = row['answer']
            
            cache.set(question, answer)
    
    print(f"预热完成，缓存大小: {cache.stats()}")

# FAQ 文件格式
# question,answer
# 产品价格是多少？,产品价格 99 元。
# 退货政策是什么？,7天内可退货。
```

## 32.10 本章小结

- 语义缓存：基于语义相似度而非精确匹配
- 原理：查询向量 → 检索缓存 → 相似度判断 → 返回/调用 LLM
- 基础实现：内存缓存、向量库缓存
- LangChain 集成：InMemoryCache、SQLiteCache、GPTCache
- 高级策略：分层缓存、分类缓存、预热缓存
- RAG 结合：缓存检索结果和生成答案
- 监控优化：命中率监控、阈值优化、淘汰策略
- 分布式缓存：Redis + 向量库混合方案
- 最佳实践：按场景配置阈值、TTL、容量
- 适用场景：FAQ、客服、知识问答（高命中率）