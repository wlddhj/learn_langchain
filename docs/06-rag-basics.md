# 第6章：RAG 检索增强生成（基础）

## 6.1 什么是 RAG

RAG (Retrieval-Augmented Generation) = 检索增强生成，核心思路：

```
用户提问 → 从知识库检索相关文档 → 将文档作为上下文喂给 LLM → LLM 生成回答
```

**为什么需要 RAG？**
- LLM 的知识有截止日期，无法获取最新信息
- LLM 不了解你的私有数据（公司文档、内部知识库等）
- 直接把所有文档塞进 prompt 会超出 token 限制

## 6.2 RAG 的完整流程

```
[离线索引阶段]
文档 → Document Loader → Text Splitter → Embeddings → Vector Store

[在线查询阶段]
用户问题 → Embeddings → Vector Store 检索 → 相关文档 + 问题 → LLM → 回答
```

## 6.3 Document Loaders（文档加载器）

将各种格式的文件加载为 LangChain Document 对象：

```python
from langchain_core.documents import Document

# Document 结构
doc = Document(
    page_content="文档的文本内容",
    metadata={"source": "file.pdf", "page": 1}
)
```

### 常用 Loader

```python
# 文本文件
from langchain_community.document_loaders import TextLoader
loader = TextLoader("notes.txt")
docs = loader.load()

# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("report.pdf")
docs = loader.load()  # 每页一个 Document

# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()

# Web 网页
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# 目录下的所有文件
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
```

### 懒加载（大文件场景）

```python
# 使用 lazy_load 逐条生成，避免一次性加载全部到内存
for doc in loader.lazy_load():
    process(doc)
```

## 6.4 Text Splitters（文本分割器）

将长文档切分成合适大小的块（chunk）：

### RecursiveCharacterTextSplitter（推荐）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # 每块最大字符数
    chunk_overlap=200,     # 块之间的重叠字符数
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 分隔符优先级
)

chunks = splitter.split_documents(docs)
print(f"原始文档数: {len(docs)}")
print(f"切分后块数: {len(chunks)}")
print(f"第一块内容: {chunks[0].page_content[:100]}")
```

> **为什么需要 overlap？** 重叠部分确保信息不会在切割处丢失。

### 其他分割器

```python
# 按字符分割（简单粗暴）
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
)

# 按 Token 分割（精确控制 token 数）
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(
    chunk_size=500,       # 每块最大 token 数
    chunk_overlap=50,
)

# Markdown 分割（按标题层级）
from langchain_text_splitters import MarkdownHeaderTextSplitter
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

# 代码分割
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)
```

### Chunk 大小的选择

| 场景 | 推荐 chunk_size | chunk_overlap |
|------|----------------|---------------|
| 问答 | 500-1000 | 50-200 |
| 摘要 | 2000-4000 | 200-400 |
| 精确检索 | 300-500 | 50-100 |

## 6.5 Embeddings（文本向量化）

将文本转换为固定维度的向量（数组），用于计算相似度：

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 单文本向量化
vector = embeddings.embed_query("什么是机器学习？")
print(len(vector))   # 1536（向量维度）
print(vector[:5])    # [-0.0023, 0.0089, ...]

# 批量文本向量化
vectors = embeddings.embed_documents([
    "机器学习是AI的分支",
    "深度学习是机器学习的分支",
])
```

### 相似度计算

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

q_vec = embeddings.embed_query("什么是AI？")
d1_vec = embeddings.embed_query("人工智能是计算机科学的分支")
d2_vec = embeddings.embed_query("今天天气真好")

print(f"相关文档相似度: {cosine_similarity(q_vec, d1_vec):.4f}")  # ~0.8+
print(f"无关文档相似度: {cosine_similarity(q_vec, d2_vec):.4f}")  # ~0.3
```

## 6.6 Vector Stores（向量数据库）

存储向量并支持相似度检索：

### Chroma（本地开发推荐）

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 从文档创建向量库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化目录
)

# 相似度检索
results = vectorstore.similarity_search("什么是机器学习？", k=3)
for doc in results:
    print(doc.page_content[:100])
    print(doc.metadata)
    print("---")

# 带分数的相似度检索
results_with_scores = vectorstore.similarity_search_with_score(
    "什么是机器学习？", k=3
)
for doc, score in results_with_scores:
    print(f"分数: {score:.4f}")
    print(doc.page_content[:100])
    print("---")
```

### FAISS（高性能，适合大规模数据）

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)

# 保存/加载
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
```

### 加载已有向量库

```python
# 不用每次重新创建
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

## 6.7 构建一个完整的 RAG Chain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = TextLoader("knowledge.txt")
docs = loader.load()

# 2. 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. 创建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. 构建 RAG prompt
prompt = ChatPromptTemplate.from_template("""基于以下上下文回答用户问题。如果上下文中没有相关信息，请回答"我不知道"。

上下文：
{context}

问题：{question}

回答：""")

# 6. 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. 组装 chain
llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 8. 使用
answer = rag_chain.invoke("什么是LangChain？")
print(answer)
```

## 6.8 本章小结

- RAG = 检索相关文档 + 喂给 LLM 生成回答
- Document Loader 加载各种格式的文件
- Text Splitter 将长文档切成合适大小的块
- Embeddings 将文本转换为向量用于相似度计算
- Vector Store 存储向量并支持检索
- 完整 RAG chain：`retriever → prompt → llm → parser`
