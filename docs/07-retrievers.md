# 第7章：Retriever 检索器进阶

## 7.1 Retriever vs VectorStore

| 组件 | 职责 | 接口方法 |
|------|------|----------|
| VectorStore | 存储 + 检索向量 | `similarity_search()` |
| Retriever | 检索的统一接口 | `invoke()` |

```python
# VectorStore 的检索方法
results = vectorstore.similarity_search("query", k=3)  # 返回 Document 列表

# 转换为 Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("query")  # 同样返回 Document 列表
```

> **推荐**：在 chain 中始终使用 Retriever 接口，它更灵活、可替换。

## 7.2 基础检索策略

### 相似度检索 (Similarity)

```python
# 默认策略，返回最相似的 k 个文档
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
```

### MMR 检索 (Maximal Marginal Relevance)

在**相关性**和**多样性**之间取得平衡，避免返回内容重复的文档：

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,                  # 最终返回 3 个
        "fetch_k": 10,           # 先检索 10 个候选
        "lambda_mult": 0.5,      # 0=最大多样性，1=最大相关性
    },
)
```

**何时使用 MMR？**
- 检索结果经常重复时
- 需要覆盖多个角度时
- 知识库中有大量相似文档时

### 相似度分数阈值

```python
# 只返回相似度高于阈值的文档
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # 只返回分数 > 0.8 的
        "k": 5,
    },
)
```

## 7.3 MultiQueryRetriever（多查询检索）

用 LLM 从不同角度重写用户查询，提高召回率：

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# 工作流程：
# 1. 用户查询: "什么是深度学习？"
# 2. LLM 生成多个变体查询:
#    - "深度学习的定义是什么？"
#    - "请解释深度学习的基本概念"
#    - "深度学习和机器学习有什么关系？"
# 3. 对每个查询分别检索
# 4. 合并去重结果
```

### 自定义查询生成 prompt

```python
from langchain_core.prompts import ChatPromptTemplate

query_prompt = ChatPromptTemplate.from_template(
    """你是一个AI助手。请根据用户的问题，生成3个不同角度的版本。

原始问题: {question}

要求：
- 从不同视角重新表述问题
- 使用不同的术语和表达
- 每行一个，不要编号

改写后的问题:"""
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
    prompt=query_prompt,
)
```

## 7.4 ContextualCompressionRetriever（上下文压缩）

先检索，再压缩——只提取与查询相关的片段：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 用 LLM 从检索到的文档中提取相关部分
compressor = LLMChainExtractor.from_llm(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

# 效果：原本长篇文档被压缩为只包含答案相关的片段
compressed_docs = compression_retriever.invoke("什么是向量数据库？")
```

### 嵌入器过滤（EmbeddingsFilter，更快更便宜）

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76,  # 相似度阈值
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)
```

## 7.5 EnsembleRetriever（混合检索/Hybrid Search）

混合检索（Hybrid Search）结合**向量检索**（语义匹配）和**关键词检索**（精确匹配），是目前效果最好的检索策略。

### 为什么需要混合检索？

| 检索方式 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| **向量检索** | 理解语义相似性 | 遗漏精确关键词匹配 | 语义相近、表述不同 |
| **BM25检索** | 精确关键词匹配 | 不理解语义关系 | 专业术语、精确匹配 |
| **混合检索** | 两者优势互补 | 计算成本稍高 | 生产环境推荐 |

### 完整混合检索实现

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 准备文档
docs = [
    Document(page_content="RAG是检索增强生成技术，结合检索和生成"),
    Document(page_content="LangChain是一个LLM应用开发框架"),
    Document(page_content="向量数据库用于存储文本的向量表示"),
    Document(page_content="BM25是经典的关键词检索算法"),
    Document(page_content="语义检索可以理解文本的语义相似性"),
]

# 1. 创建 BM25 检索器（关键词检索）
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5  # 返回 top 5

# 2. 创建向量检索器（语义检索）
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. 混合检索器 - 结合两者
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # BM25 40%，向量 60%
)

# 使用
results = ensemble_retriever.invoke("RAG是什么技术？")
for doc in results:
    print(doc.page_content)
```

### 权重调整策略

权重分配取决于应用场景：

```python
# 场景1：专业术语多 → 提高 BM25 权重
technical_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.6, 0.4],  # BM25 优先
)

# 场景2：自然语言问答 → 提高向量权重
qa_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # 向量优先
)

# 场景3：均衡场景 → 平衡权重
balanced_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # 均衡
)
```

### 混合检索原理：Reciprocal Rank Fusion (RRF)

EnsembleRetriever 使用 RRF 算法合并结果：

```
RRF 公式：
score(d) = Σ (1 / (k + rank(d, retriever_i)))

其中：
- d：文档
- k：常数（默认60）
- rank(d, retriever_i)：文档在检索器 i 中的排名
```

```python
# RRF 算法示意（简化版）
def reciprocal_rank_fusion(results_list, k=60):
    """
    多个检索结果合并
    
    results_list: 每个检索器的结果列表
    k: RRF 常数，影响排名差异的权重
    """
    rrf_scores = {}
    
    for results in results_list:
        for rank, doc in enumerate(results, 1):
            doc_id = doc.metadata.get("id", doc.page_content)
            
            # RRF 分数计算
            rrf_score = 1 / (k + rank)
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "doc": doc}
            
            rrf_scores[doc_id]["score"] += rrf_score
    
    # 按分数排序
    sorted_docs = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    return [item["doc"] for item in sorted_docs]
```

### 效果对比实验

```python
import time

# 测试查询
queries = [
    "RAG技术原理",       # 专业术语
    "如何开发LLM应用",   # 自然语言
    "向量数据库是什么",  # 概念查询
    "BM25算法参数k",     # 精确关键词
]

def compare_retrievers(queries, retrievers):
    """对比不同检索器的效果"""
    results = {}
    
    for name, retriever in retrievers.items():
        results[name] = []
        
        for query in queries:
            docs = retriever.invoke(query)
            results[name].append({
                "query": query,
                "doc_count": len(docs),
                "top_doc": docs[0].page_content[:50] if docs else "无结果",
            })
    
    return results

# 对比实验
retrievers = {
    "BM25": bm25_retriever,
    "Vector": vector_retriever,
    "Ensemble": ensemble_retriever,
}

comparison = compare_retrievers(queries, retrievers)

# 输出对比表
print("检索策略对比：")
print("-" * 80)
for query in queries:
    print(f"\n查询: {query}")
    for name in retrievers:
        result = [r for r in comparison[name] if r["query"] == query][0]
        print(f"  {name}: {result['doc_count']}个结果, 首个: {result['top_doc']}")
```

### 混合检索效果评估

| 查询类型 | BM25 | 向量检索 | 混合检索 |
|----------|------|----------|----------|
| 精确术语查询 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 语义相近查询 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 自然语言问答 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 模糊概念查询 | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 多路召回扩展

可以添加更多检索路径：

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# 多查询向量检索
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_retriever,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

# 三路混合：BM25 + 向量 + 多查询向量
advanced_ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever, multi_query_retriever],
    weights=[0.3, 0.4, 0.3],
)
```

### 生产环境建议

```python
# 推荐的生产配置
def create_production_retriever(vectorstore, docs):
    """生产环境混合检索配置"""
    
    # BM25 配置
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 10  # 多取一些候选
    
    # 向量检索配置
    vector = vectorstore.as_retriever(
        search_type="mmr",  # 使用 MMR 增加多样性
        search_kwargs={"k": 10, "fetch_k": 20}
    )
    
    # 混合
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6],
    )
    
    return ensemble
```

**混合检索核心优势总结**：
- 向量检索理解语义，但可能遗漏精确关键词
- BM25 检索精确匹配关键词，但不理解语义
- RRF 算法智能合并两者结果
- 生产环境推荐：混合检索 + Re-ranking 后处理

## 7.6 Re-ranking（重排序）

先粗检索大量候选，再用 LLM/Cross-encoder 精细排序：

```python
# 使用 Cohere Reranker（需要 API Key）
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# 先取 10 个候选，再用 Cross-encoder 重排取 top 3
```

## 7.7 ParentDocumentRetriever（父子文档检索）

检索小块，返回大块（保留完整上下文）：

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# 子文档分割器（用于检索）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

# 父文档分割器（用于返回）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 存储父子关系
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

# 检索时：匹配小片段 → 返回对应的完整大文档
results = retriever.invoke("查询内容")
```

## 7.8 Self-Querying Retriever（自查询检索）

自动从自然语言查询中提取元数据过滤条件：

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(name="genre", description="电影类型", type="string"),
    AttributeInfo(name="year", description="上映年份", type="integer"),
    AttributeInfo(name="rating", description="评分", type="float"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectorstore,
    document_contents="电影信息",
    metadata_field_info=metadata_field_info,
)

# 用户说 "2023年评分高于8.0的科幻电影"
# 自动生成: vector_search("科幻电影") + filter(year=2023, rating>8.0)
results = retriever.invoke("2023年评分高于8.0的科幻电影")
```

## 7.9 检索策略选择指南

```
简单场景 → similarity 检索
结果重复 → MMR
召回不足 → MultiQueryRetriever + EnsembleRetriever
结果太长 → ContextualCompression
需要精确 + 语义 → Ensemble (BM25 + Vector)
有元数据 → Self-Querying
需要上下文 → ParentDocument
```

## 7.10 本章小结

- 基础检索：`similarity`（精确）、`mmr`（多样性）、阈值过滤
- **MultiQueryRetriever**：多角度重写查询，提高召回
- **EnsembleRetriever（混合检索）**：BM25关键词 + 向量语义，RRF算法合并，生产环境推荐
- **ContextualCompression**：压缩无关内容，减少 token 消耗
- **ParentDocument**：小块检索，大块返回，保留完整上下文
- **Self-Querying**：自动从查询中提取元数据过滤条件
- **Re-ranking**：粗检索后精细排序，进一步提升精度
- 混合检索权重调整：专业术语场景提高BM25权重，自然语言场景提高向量权重
