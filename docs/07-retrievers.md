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

## 7.5 EnsembleRetriever（混合检索）

结合多种检索策略，取长补短：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25：关键词检索（擅长精确匹配）
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# 向量检索（擅长语义相似）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合检索，权重各占50%
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("RAG是什么技术？")
```

**为什么混合检索效果好？**
- 向量检索：理解语义，但可能遗漏精确关键词
- BM25 检索：精确匹配关键词，但不理解语义
- 混合后：两者优势互补

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
- **EnsembleRetriever**：混合关键词+向量检索，效果最好
- **ContextualCompression**：压缩无关内容，减少 token 消耗
- **ParentDocument**：小块检索，大块返回
- **Self-Querying**：自动从查询中提取元数据过滤条件
