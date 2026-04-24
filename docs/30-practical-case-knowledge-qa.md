# 第30章：知识问答系统实战

## 30.1 项目概述

### 系统目标

构建一个企业级知识问答系统，具备以下功能：
- 多格式文档导入（PDF、Word、TXT、Markdown）
- 智能文档切分与索引
- 高精度语义检索
- 多轮问答对话
- 知识溯源与引用
- 知识更新与管理
- 用户反馈与优化

### 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                     用户界面层                           │
│         (Web UI / API / 移动端 / 企业微信集成)          │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     应用服务层                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 问答引擎  │  │ 文档管理  │  │ 用户管理  │           │
│  └───────────┘  └───────────┘  └───────────┘           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 反馈系统  │  │ 权限控制  │  │ 日志追踪  │           │
│  └───────────┘  └───────────┘  └───────────┘           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     核心处理层                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 文档解析  │  │ 文档切分  │  │ 向量嵌入  │           │
│  └───────────┘  └───────────┘  └───────────┘           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 检索引擎  │  │ 答案生成  │  │ 溯源引用  │           │
│  └───────────┘  └───────────┘  └───────────┘           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     数据存储层                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 向量数据库│  │ 文档数据库│  │ 元数据存储│           │
│  └───────────┘  └───────────┘  └───────────┘           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 会话存储  │  │ 反馈记录  │  │ 日志存储  │           │
│  └───────────┘  └───────────┘  └───────────┘           │
└─────────────────────────────────────────────────────────┘
```

## 30.2 核心模块设计

### 数据模型

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(BaseModel):
    """文档模型"""
    id: str
    title: str
    file_type: str  # pdf/docx/txt/md
    file_path: str
    content: Optional[str] = None
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime
    updated_at: datetime
    metadata: dict = {}

class DocumentChunk(BaseModel):
    """文档片段模型"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    metadata: dict = {}

class Question(BaseModel):
    """问题模型"""
    id: str
    user_id: str
    content: str
    created_at: datetime

class Answer(BaseModel):
    """答案模型"""
    id: str
    question_id: str
    content: str
    confidence: float
    sources: List[dict] = []  # 引用的文档片段
    created_at: datetime

class Feedback(BaseModel):
    """反馈模型"""
    id: str
    question_id: str
    answer_id: str
    user_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    is_correct: Optional[bool] = None
    created_at: datetime
```

### 目录结构

```
knowledge_qa_system/
├── core/
│   ├── document_processor.py    # 文档处理
│   ├── text_splitter.py         # 文本切分
│   ├── embedder.py              # 向量嵌入
│   ├── retriever.py             # 检索引擎
│   ├── answer_generator.py      # 答案生成
│   └── source_tracer.py         # 溯源引用
├── loaders/
│   ├── pdf_loader.py            # PDF 加载
│   ├── docx_loader.py           # Word 加载
│   ├── txt_loader.py            # TXT 加载
│   └── markdown_loader.py       # Markdown 加载
├── storage/
│   ├── vector_store.py          # 向量存储
│   ├── document_store.py        # 文档存储
│   ├── session_store.py         # 会话存储
│   └── feedback_store.py        # 反馈存储
├── services/
│   ├── qa_service.py            # 问答服务
│   ├── document_service.py      # 文档管理
│   ├── feedback_service.py      # 反反馈管理
│   └── user_service.py          # 用户管理
├── api/
│   ├── main.py                  # API 主入口
│   ├── routes/
│   │   ├── qa.py                # 问答路由
│   │   ├── documents.py         # 文档路由
│   │   ├── feedback.py          # 反馈路由
│   │   └── admin.py             # 管理路由
│   └── middleware/
│   │   ├── auth.py              # 认证
│   │   ├── rate_limit.py        # 限流
│   └── websocket.py             # WebSocket
├── web/
│   ├── static/                  # 静态文件
│   ├── templates/               # HTML模板
│   └── app.py                   # Web应用
├── monitoring/
│   ├── metrics.py               # 监控指标
│   ├── analytics.py             # 分析统计
│   └── dashboard.py             # 监控面板
├── config/
│   ├── settings.py              # 配置
│   └── prompts.py               # Prompt配置
├── tests/
│   ├── test_processor.py
│   ├── test_retriever.py
│   └── test_api.py
└── requirements.txt
```

## 30.3 文档处理模块

### 多格式文档加载

```python
# loaders/document_loader.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from typing import List
import os

class DocumentLoader:
    """多格式文档加载器"""
    
    def __init__(self):
        self.loaders = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader,
            "md": UnstructuredMarkdownLoader,
        }
    
    def load(self, file_path: str) -> List[Document]:
        """加载文档"""
        file_type = self._get_file_type(file_path)
        
        if file_type not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        loader_class = self.loaders[file_type]
        
        # 根据类型选择加载方式
        if file_type == "pdf":
            loader = loader_class(file_path)
        else:
            loader = loader_class(file_path)
        
        documents = loader.load()
        
        # 添加元数据
        for doc in documents:
            doc.metadata["source_file"] = file_path
            doc.metadata["file_type"] = file_type
            doc.metadata["load_time"] = datetime.now().isoformat()
        
        return documents
    
    def load_batch(self, file_paths: List[str]) -> List[Document]:
        """批量加载文档"""
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self.load(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return all_documents
    
    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """加载目录下所有文档"""
        file_paths = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if self._is_supported(file):
                        file_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if self._is_supported(file):
                    file_paths.append(os.path.join(directory, file))
        
        return self.load_batch(file_paths)
    
    def _get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        return os.path.splitext(file_path)[1].lower().replace(".", "")
    
    def _is_supported(self, file_name: str) -> bool:
        """检查文件是否支持"""
        file_type = self._get_file_type(file_name)
        return file_type in self.loaders


# 使用示例
loader = DocumentLoader()

# 加载单个文档
docs = loader.load("./documents/product_manual.pdf")
print(f"加载了 {len(docs)} 页")

# 加载目录
docs = loader.load_directory("./documents/")
print(f"加载了 {len(docs)} 个文档")
```

### 智能文本切分

```python
# core/text_splitter.py
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_core.documents import Document
from typing import List, Optional
import re

class SmartTextSplitter:
    """智能文本切分器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        
        # 不同策略的切分器
        self.splitters = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            ),
            "markdown": MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "header1"),
                    ("##", "header2"),
                    ("###", "header3"),
                ],
            ),
            "sentence": SentenceTransformersTokenTextSplitter(
                chunk_overlap=chunk_overlap,
                tokens_per_chunk=chunk_size,
            ),
        }
    
    def split(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        if self.strategy == "markdown":
            return self._split_markdown(documents)
        elif self.strategy == "sentence":
            return self._split_by_sentence(documents)
        else:
            return self._split_recursive(documents)
    
    def _split_recursive(self, documents: List[Document]) -> List[Document]:
        """递归切分"""
        splitter = self.splitters["recursive"]
        
        chunks = []
        for doc in documents:
            split_docs = splitter.split_documents([doc])
            
            for i, chunk in enumerate(split_docs):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_strategy"] = "recursive"
                chunks.append(chunk)
        
        return chunks
    
    def _split_markdown(self, documents: List[Document]) -> List[Document]:
        """Markdown 按标题切分"""
        md_splitter = self.splitters["markdown"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        chunks = []
        for doc in documents:
            # 先按标题切分
            if doc.metadata.get("file_type") == "md":
                header_splits = md_splitter.split_text(doc.page_content)
                
                # 再按长度切分
                for header_split in header_splits:
                    content = header_split[1] if isinstance(header_split, tuple) else str(header_split)
                    text_chunks = text_splitter.split_text(content)
                    
                    for chunk in text_chunks:
                        chunks.append(Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                "header": header_split[0] if isinstance(header_split, tuple) else "",
                            },
                        ))
            else:
                # 非 Markdown 文件使用普通切分
                chunks.extend(self._split_recursive([doc]))
        
        return chunks
    
    def _split_by_sentence(self, documents: List[Document]) -> List[Document]:
        """按句子切分"""
        splitter = self.splitters["sentence"]
        
        chunks = []
        for doc in documents:
            sentences = self._extract_sentences(doc.page_content)
            
            # 组合成合适大小的块
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(Document(
                            page_content=current_chunk,
                            metadata={**doc.metadata, "chunk_index": chunk_index},
                        ))
                        chunk_index += 1
                    current_chunk = sentence
            
            # 最后一块
            if current_chunk:
                chunks.append(Document(
                    page_content=current_chunk,
                    metadata={**doc.metadata, "chunk_index": chunk_index},
                ))
        
        return chunks
    
    def _extract_sentences(self, text: str) -> List[str]:
        """提取句子"""
        # 中文句子切分
        pattern = r'[^。！？；]+[。！？；]'
        sentences = re.findall(pattern, text)
        
        if not sentences:
            # 英文句子切分
            pattern = r'[^.!?]+[.!?]'
            sentences = re.findall(pattern, text)
        
        return sentences


# 混合切分策略
class HybridSplitter:
    """混合切分策略"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, documents: List[Document]) -> List[Document]:
        """混合切分"""
        chunks = []
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "")
            
            # 根据文件类型选择策略
            if file_type == "md":
                splitter = SmartTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    strategy="markdown",
                )
            elif file_type == "pdf":
                # PDF 通常需要递归切分
                splitter = SmartTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    strategy="recursive",
                )
            else:
                splitter = SmartTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    strategy="recursive",
                )
            
            chunks.extend(splitter.split([doc]))
        
        return chunks


# 使用示例
loader = DocumentLoader()
splitter = HybridSplitter(chunk_size=500, chunk_overlap=50)

# 加载并切分
docs = loader.load("./documents/manual.pdf")
chunks = splitter.split(docs)

print(f"原文档 {len(docs)} 页")
print(f"切分后 {len(chunks)} 个片段")
```

### 向量嵌入与存储

```python
# core/embedder.py
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import os

class DocumentEmbedder:
    """文档向量嵌入"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        persist_dir: Optional[str] = None,
    ):
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.persist_dir = persist_dir
        
        # 初始化向量库
        if persist_dir and os.path.exists(persist_dir):
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )
        else:
            self.vectorstore = None
    
    def embed_documents(self, documents: List[Document]) -> Chroma:
        """嵌入文档并创建向量库"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """添加新文档"""
        if self.vectorstore:
            self.vectorstore.add_documents(documents)
        else:
            self.embed_documents(documents)
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """获取向量库"""
        return self.vectorstore
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本的向量"""
        return self.embeddings.embed_documents(texts)
    
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量"""
        return self.embeddings.embed_query(text)


# 使用示例
loader = DocumentLoader()
splitter = HybridSplitter()
embedder = DocumentEmbedder(persist_dir="./vector_db")

# 处理流程
docs = loader.load_directory("./documents/")
chunks = splitter.split(docs)
vectorstore = embedder.embed_documents(chunks)

print(f"向量库中有 {vectorstore._collection.count()} 个向量")
```

## 30.4 检索引擎

### 多策略检索

```python
# core/retriever.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Callable
from pydantic import BaseModel

class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = 5
    score_threshold: float = 0.7
    search_type: str = "similarity"  # similarity/mmr/similarity_score_threshold

class MultiStrategyRetriever:
    """多策略检索器"""
    
    def __init__(
        self,
        vectorstore: Chroma,
        config: RetrievalConfig = None,
    ):
        self.vectorstore = vectorstore
        self.config = config or RetrievalConfig()
        self.embeddings = OpenAIEmbeddings()
    
    def retrieve(self, query: str) -> List[Document]:
        """检索文档"""
        if self.config.search_type == "similarity":
            return self._similarity_search(query)
        elif self.config.search_type == "mmr":
            return self._mmr_search(query)
        elif self.config.search_type == "similarity_score_threshold":
            return self._threshold_search(query)
        else:
            return self._similarity_search(query)
    
    def _similarity_search(self, query: str) -> List[Document]:
        """相似度检索"""
        return self.vectorstore.similarity_search(
            query=query,
            k=self.config.top_k,
        )
    
    def _mmr_search(self, query: str) -> List[Document]:
        """MMR 检索（多样性）"""
        return self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=self.config.top_k,
            fetch_k=self.config.top_k * 3,
        )
    
    def _threshold_search(self, query: str) -> List[Document]:
        """阈值检索"""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.config.top_k,
        )
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """检索并返回分数"""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.config.top_k,
        )
    
    def hybrid_retrieve(
        self,
        query: str,
        keyword_filter: Optional[Callable] = None,
    ) -> List[Document]:
        """混合检索（向量 + 关键词）"""
        # 向量检索
        vector_results = self._similarity_search(query)
        
        # 关键词检索（可选）
        if keyword_filter:
            filtered_results = [doc for doc in vector_results if keyword_filter(doc)]
            if filtered_results:
                return filtered_results
        
        return vector_results
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """重排序检索结果"""
        # 使用 LLM 重排序
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        rerank_prompt = ChatPromptTemplate.from_template("""
        对以下检索结果按与查询的相关性重新排序：
        
        查询：{query}
        
        检索结果：
        {documents}
        
        请输出排序后的编号列表（最相关的在前）。
        """)
        
        doc_text = "\n".join([
            f"[{i+1}] {doc.page_content[:200]}..."
            for i, doc in enumerate(documents)
        ])
        
        ranking = (rerank_prompt | llm).invoke({
            "query": query,
            "documents": doc_text,
        }).content
        
        # 解析排序结果
        import re
        indices = re.findall(r'\[(\d+)\]', ranking)
        sorted_docs = []
        
        for idx in indices:
            idx = int(idx) - 1
            if idx < len(documents):
                sorted_docs.append(documents[idx])
        
        # 补充未排序的文档
        for doc in documents:
            if doc not in sorted_docs:
                sorted_docs.append(doc)
        
        return sorted_docs


# 使用示例
embedder = DocumentEmbedder(persist_dir="./vector_db")
vectorstore = embedder.get_vectorstore()

retriever = MultiStrategyRetriever(
    vectorstore,
    config=RetrievalConfig(top_k=5, search_type="mmr"),
)

results = retriever.retrieve("产品退货政策是什么？")
for doc in results:
    print(doc.page_content[:100])
    print("---")

# 带分数检索
results_with_scores = retriever.retrieve_with_scores("退货政策")
for doc, score in results_with_scores:
    print(f"分数: {score:.3f}, 内容: {doc.page_content[:100]}")
```

## 30.5 答案生成模块

### RAG 答案生成

```python
# core/answer_generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List, Optional
from pydantic import BaseModel

class AnswerResult(BaseModel):
    """答案结果"""
    answer: str
    sources: List[dict]
    confidence: float
    reasoning: Optional[str] = None

class RAGAnswerGenerator:
    """RAG 答案生成器"""
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0,
    ):
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        
        # RAG Prompt
        self.rag_prompt = ChatPromptTemplate.from_template("""
        你是一个专业的知识问答助手。请基于以下参考资料回答问题。
        
        参考资料：
        {context}
        
        问题：{question}
        
        回答要求：
        1. 优先使用参考资料中的信息
        2. 如果资料中没有相关信息，明确说明"参考资料中暂无相关信息"
        3. 不要添加参考资料之外的推断或猜测
        4. 引用具体来源（标注来源编号）
        5. 如果问题涉及多个方面，请分别回答
        
        回答格式：
        ```
        【回答内容】
        [基于来源 X 的信息，引用具体内容]
        
        【来源引用】
        - 来源 X：[具体内容摘要]
        
        【补充说明】（如有）
        ```
        """)
    
    def generate(
        self,
        question: str,
        documents: List[Document],
    ) -> AnswerResult:
        """生成答案"""
        # 格式化上下文
        context = self._format_documents(documents)
        
        # 构建 Chain
        chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # 生成答案
        answer = chain.invoke({
            "context": context,
            "question": question,
        })
        
        # 计算置信度
        confidence = self._calculate_confidence(question, documents, answer)
        
        return AnswerResult(
            answer=answer,
            sources=self._extract_sources(documents),
            confidence=confidence,
        )
    
    def _format_documents(self, documents: List[Document]) -> str:
        """格式化文档"""
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            source_info = doc.metadata.get("source_file", "未知来源")
            formatted.append(f"[来源 {i}] (文件: {source_info})\n{doc.page_content}")
        
        return "\n\n".join(formatted)
    
    def _extract_sources(self, documents: List[Document]) -> List[dict]:
        """提取来源信息"""
        sources = []
        
        for i, doc in enumerate(documents, 1):
            sources.append({
                "index": i,
                "source_file": doc.metadata.get("source_file", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "content_preview": doc.page_content[:200],
                "metadata": doc.metadata,
            })
        
        return sources
    
    def _calculate_confidence(
        self,
        question: str,
        documents: List[Document],
        answer: str,
    ) -> float:
        """计算置信度"""
        # 基于检索结果质量评估
        if not documents:
            return 0.0
        
        # 使用 LLM 评估答案质量
        eval_prompt = ChatPromptTemplate.from_template("""
        评估答案的可信度（0-1 分）：
        
        问题：{question}
        参考资料：{context}
        答案：{answer}
        
        评分标准：
        - 答案是否基于参考资料？
        - 是否存在幻觉（编造信息）？
        - 是否完整回答问题？
        
        只输出分数（0-1 的数字）。
        """)
        
        score_text = (eval_prompt | self.llm).invoke({
            "question": question,
            "context": self._format_documents(documents),
            "answer": answer,
        }).content
        
        # 解析分数
        try:
            import re
            score_match = re.search(r'[0-9.]+', score_text)
            if score_match:
                return float(score_match.group())
        except:
            pass
        
        return 0.5


# 多轮对话增强
class ConversationalAnswerGenerator(RAGAnswerGenerator):
    """支持多轮对话的答案生成器"""
    
    def __init__(self, llm_model: str = "gpt-4o-mini", temperature: float = 0):
        super().__init__(llm_model, temperature)
        
        self.conversation_prompt = ChatPromptTemplate.from_template("""
        你是一个专业的知识问答助手。请基于以下参考资料和对话历史回答问题。
        
        对话历史：
        {history}
        
        参考资料：
        {context}
        
        当前问题：{question}
        
        回答要求：
        1. 结合对话历史理解问题意图
        2. 优先使用参考资料中的信息
        3. 如果是追问，请联系上文回答
        4. 不要添加参考资料之外的推断
        5. 引用具体来源
        
        回答格式同上。
        """)
    
    def generate_with_history(
        self,
        question: str,
        documents: List[Document],
        history: List[tuple],  # [(role, content), ...]
    ) -> AnswerResult:
        """带历史对话的答案生成"""
        # 格式化历史
        history_text = "\n".join([
            f"{role}: {content}" for role, content in history
        ])
        
        context = self._format_documents(documents)
        
        chain = self.conversation_prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "history": history_text,
            "context": context,
            "question": question,
        })
        
        confidence = self._calculate_confidence(question, documents, answer)
        
        return AnswerResult(
            answer=answer,
            sources=self._extract_sources(documents),
            confidence=confidence,
        )


# 使用示例
generator = RAGAnswerGenerator()

question = "产品退货需要什么条件？"
docs = retriever.retrieve(question)

result = generator.generate(question, docs)
print("答案:", result.answer)
print("置信度:", result.confidence)
print("来源数量:", len(result.sources))
```

## 30.6 完整问答系统

```python
# services/qa_service.py
from core.document_loader import DocumentLoader
from core.text_splitter import HybridSplitter
from core.embedder import DocumentEmbedder
from core.retriever import MultiStrategyRetriever, RetrievalConfig
from core.answer_generator import RAGAnswerGenerator, ConversationalAnswerGenerator
from langchain_core.documents import Document
from typing import List, Optional, Dict
from datetime import datetime
import uuid
import os

class KnowledgeQASystem:
    """知识问答系统"""
    
    def __init__(
        self,
        documents_dir: str,
        persist_dir: str = "./vector_db",
        config: Dict = None,
    ):
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.config = config or {}
        
        # 初始化各模块
        self.loader = DocumentLoader()
        self.splitter = HybridSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
        )
        self.embedder = DocumentEmbedder(persist_dir=persist_dir)
        self.retriever_config = RetrievalConfig(
            top_k=self.config.get("top_k", 5),
            search_type=self.config.get("search_type", "mmr"),
        )
        self.generator = ConversationalAnswerGenerator()
        
        # 会话管理
        self.sessions = {}
        
        # 初始化向量库
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """初始化向量库"""
        if os.path.exists(self.persist_dir):
            # 加载已有向量库
            self.vectorstore = self.embedder.get_vectorstore()
            self.retriever = MultiStrategyRetriever(
                self.vectorstore,
                self.retriever_config,
            )
        else:
            # 创建新向量库
            self._build_vectorstore()
    
    def _build_vectorstore(self):
        """构建向量库"""
        # 加载文档
        documents = self.loader.load_directory(self.documents_dir)
        
        # 切分文档
        chunks = self.splitter.split(documents)
        
        # 嵌入并存储
        self.vectorstore = self.embedder.embed_documents(chunks)
        
        # 创建检索器
        self.retriever = MultiStrategyRetriever(
            self.vectorstore,
            self.retriever_config,
        )
        
        print(f"已处理 {len(documents)} 个文档，{len(chunks)} 个片段")
    
    def create_session(self, user_id: str) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "history": [],
            "created_at": datetime.now(),
        }
        return session_id
    
    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """提问"""
        # 检索
        documents = self.retriever.retrieve(question)
        
        # 重排序（可选）
        if self.config.get("enable_rerank", False):
            documents = self.retriever.rerank(question, documents)
        
        # 生成答案
        if session_id and session_id in self.sessions:
            history = self.sessions[session_id]["history"]
            result = self.generator.generate_with_history(
                question, documents, history
            )
            
            # 更新历史
            self.sessions[session_id]["history"].append(("user", question))
            self.sessions[session_id]["history"].append(("assistant", result.answer))
        else:
            result = self.generator.generate(question, documents)
        
        return {
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "session_id": session_id,
        }
    
    def add_document(self, file_path: str) -> dict:
        """添加新文档"""
        # 加载
        documents = self.loader.load(file_path)
        
        # 切分
        chunks = self.splitter.split(documents)
        
        # 添加到向量库
        self.embedder.add_documents(chunks)
        
        return {
            "file_path": file_path,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "status": "success",
        }
    
    def update_knowledge_base(self):
        """更新知识库"""
        self._build_vectorstore()
    
    def get_session_history(self, session_id: str) -> List[tuple]:
        """获取会话历史"""
        if session_id in self.sessions:
            return self.sessions[session_id]["history"]
        return []
    
    def close_session(self, session_id: str):
        """关闭会话"""
        if session_id in self.sessions:
            self.sessions[session_id]["closed_at"] = datetime.now()
    
    def search_knowledge(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[dict]:
        """搜索知识库"""
        results = self.retriever.retrieve_with_scores(query)
        
        return [
            {
                "content": doc.page_content,
                "score": score,
                "source": doc.metadata.get("source_file", ""),
            }
            for doc, score in results[:top_k]
        ]
    
    def get_stats(self) -> dict:
        """获取系统统计"""
        return {
            "vector_count": self.vectorstore._collection.count(),
            "session_count": len(self.sessions),
            "documents_dir": self.documents_dir,
        }


# 使用示例
qa_system = KnowledgeQASystem(
    documents_dir="./knowledge_documents",
    persist_dir="./vector_db",
    config={
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "search_type": "mmr",
        "enable_rerank": True,
    },
)

# 创建会话
session_id = qa_system.create_session("user_123")

# 多轮问答
questions = [
    "产品退货需要什么条件？",
    "退货后多久能收到退款？",
    "如果是质量问题退货，流程有什么不同？",
]

for q in questions:
    result = qa_system.ask(q, session_id)
    print(f"问: {q}")
    print(f"答: {result['answer']}")
    print(f"置信度: {result['confidence']}")
    print("---")

# 添加新文档
qa_system.add_document("./new_document.pdf")

# 搜索知识库
search_results = qa_system.search_knowledge("退款", top_k=10)
for r in search_results:
    print(f"分数: {r['score']:.3f}, 来源: {r['source']}")
```

## 30.7 用户反馈系统

```python
# services/feedback_service.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class Feedback(BaseModel):
    """用户反馈"""
    id: str
    question_id: str
    answer_id: str
    user_id: str
    rating: int  # 1-5 星
    is_correct: Optional[bool] = None
    comment: Optional[str] = None
    created_at: datetime

class FeedbackAnalyzer:
    """反馈分析器"""
    
    def __init__(self):
        self.feedbacks = {}
    
    def record_feedback(
        self,
        question_id: str,
        answer_id: str,
        user_id: str,
        rating: int,
        is_correct: Optional[bool] = None,
        comment: Optional[str] = None,
    ) -> str:
        """记录反馈"""
        feedback_id = str(uuid.uuid4())
        
        feedback = Feedback(
            id=feedback_id,
            question_id=question_id,
            answer_id=answer_id,
            user_id=user_id,
            rating=rating,
            is_correct=is_correct,
            comment=comment,
            created_at=datetime.now(),
        )
        
        self.feedbacks[feedback_id] = feedback
        
        return feedback_id
    
    def analyze_feedbacks(self) -> dict:
        """分析反馈统计"""
        if not self.feedbacks:
            return {}
        
        ratings = [f.rating for f in self.feedbacks.values()]
        correct_count = sum(1 for f in self.feedbacks.values() if f.is_correct)
        incorrect_count = sum(1 for f in self.feedbacks.values() if f.is_correct == False)
        
        return {
            "total_feedbacks": len(self.feedbacks),
            "average_rating": sum(ratings) / len(ratings),
            "correct_rate": correct_count / (correct_count + incorrect_count) if (correct_count + incorrect_count) > 0 else 0,
            "rating_distribution": {
                "5_star": sum(1 for r in ratings if r == 5),
                "4_star": sum(1 for r in ratings if r == 4),
                "3_star": sum(1 for r in ratings if r == 3),
                "2_star": sum(1 for r in ratings if r == 2),
                "1_star": sum(1 for r in ratings if r == 1),
            },
        }
    
    def get_improvement_suggestions(self) -> List[str]:
        """获取改进建议"""
        suggestions = []
        
        # 分析低评分反馈
        low_rating_feedbacks = [
            f for f in self.feedbacks.values() if f.rating <= 2
        ]
        
        for f in low_rating_feedbacks:
            if f.comment:
                suggestions.append(f"低评分反馈: {f.comment}")
        
        # 分析错误标记
        incorrect_feedbacks = [
            f for f in self.feedbacks.values() if f.is_correct == False
        ]
        
        if incorrect_feedbacks:
            suggestions.append(f"有 {len(incorrect_feedbacks)} 个答案被标记为错误")
        
        return suggestions


# 集成到问答系统
class QASystemWithFeedback(KnowledgeQASystem):
    """带反馈的问答系统"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_service = FeedbackAnalyzer()
        self.qa_records = {}
    
    def ask(self, question: str, session_id: Optional[str] = None) -> dict:
        """提问（带记录）"""
        result = super().ask(question, session_id)
        
        # 记录问答
        qa_id = str(uuid.uuid4())
        self.qa_records[qa_id] = {
            "id": qa_id,
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "session_id": session_id,
            "created_at": datetime.now(),
        }
        
        result["qa_id"] = qa_id
        return result
    
    def submit_feedback(
        self,
        qa_id: str,
        rating: int,
        is_correct: Optional[bool] = None,
        comment: Optional[str] = None,
    ) -> str:
        """提交反馈"""
        user_id = "user_123"  # 实际应从 session 获取
        
        feedback_id = self.feedback_service.record_feedback(
            question_id=qa_id,
            answer_id=qa_id,
            user_id=user_id,
            rating=rating,
            is_correct=is_correct,
            comment=comment,
        )
        
        return feedback_id
    
    def get_feedback_stats(self) -> dict:
        """获取反馈统计"""
        return self.feedback_service.analyze_feedbacks()


# 使用示例
qa_system = QASystemWithFeedback(
    documents_dir="./knowledge_documents",
    persist_dir="./vector_db",
)

# 提问
result = qa_system.ask("退货政策是什么？")
qa_id = result["qa_id"]

# 提交反馈
feedback_id = qa_system.submit_feedback(
    qa_id=qa_id,
    rating=4,
    is_correct=True,
    comment="回答比较准确，但可以更详细",
)

# 查看统计
stats = qa_system.get_feedback_stats()
print("反馈统计:", stats)
```

## 30.8 API 服务化

```python
# api/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import os

from services.qa_service import KnowledgeQASystem

app = FastAPI(title="知识问答系统 API")

# 初始化系统
qa_system = KnowledgeQASystem(
    documents_dir=os.getenv("DOCUMENTS_DIR", "./documents"),
    persist_dir=os.getenv("PERSIST_DIR", "./vector_db"),
)

# 请求/响应模型
class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AnswerResponse(BaseModel):
    qa_id: str
    question: str
    answer: str
    sources: List[dict]
    confidence: float
    session_id: Optional[str]

class FeedbackRequest(BaseModel):
    qa_id: str
    rating: int
    is_correct: Optional[bool] = None
    comment: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    file_name: str
    document_count: int
    chunk_count: int
    status: str

# API 路由
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """问答接口"""
    result = qa_system.ask(request.question, request.session_id)
    
    return AnswerResponse(
        qa_id=result.get("qa_id", ""),
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        session_id=result["session_id"],
    )

@app.post("/session/create")
async def create_session(user_id: str):
    """创建会话"""
    session_id = qa_system.create_session(user_id)
    return {"session_id": session_id}

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """提交反馈"""
    feedback_id = qa_system.submit_feedback(
        qa_id=request.qa_id,
        rating=request.rating,
        is_correct=request.is_correct,
        comment=request.comment,
    )
    return {"feedback_id": feedback_id, "status": "success"}

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    # 保存文件
    file_path = f"./uploads/{file.filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 添加到系统
    result = qa_system.add_document(file_path)
    
    return DocumentUploadResponse(
        file_name=file.filename,
        document_count=result["document_count"],
        chunk_count=result["chunk_count"],
        status=result["status"],
    )

@app.get("/search")
async def search_knowledge(query: str, top_k: int = 10):
    """搜索知识库"""
    results = qa_system.search_knowledge(query, top_k)
    return {"results": results}

@app.get("/stats")
async def get_stats():
    """获取统计"""
    stats = qa_system.get_stats()
    feedback_stats = qa_system.get_feedback_stats()
    
    return {
        "system_stats": stats,
        "feedback_stats": feedback_stats,
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# 流式响应
@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    """流式问答"""
    from fastapi.responses import StreamingResponse
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    # 检索
    documents = qa_system.retriever.retrieve(request.question)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = ChatPromptTemplate.from_template("""
    基于以下内容回答问题：
    {context}
    
    问题：{question}
    """)
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    
    async def generate():
        async for chunk in chain.astream({
            "context": context,
            "question": request.question,
        }):
            if chunk.content:
                yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")

# 运行: uvicorn api.main:app --reload
```

## 30.9 监控与评估

```python
# monitoring/qa_metrics.py
from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time

# 指标定义
qa_requests = Counter('qa_requests_total', 'Total QA requests')
qa_latency = Histogram('qa_latency_seconds', 'QA response latency')
qa_confidence = Histogram('qa_confidence_score', 'Answer confidence scores')
feedback_count = Counter('feedback_total', 'Total feedback submissions', ['rating'])
document_count = Gauge('documents_total', 'Total documents in system')
vector_count = Gauge('vectors_total', 'Total vectors in store')

def monitor_qa(func):
    """QA 监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        qa_requests.inc()
        qa_latency.observe(time.time() - start_time)
        qa_confidence.observe(result['confidence'])
        
        return result
    
    return wrapper
```

## 30.10 本章小结

- 文档处理：多格式加载、智能切分、向量嵌入
- 检索引擎：相似度检索、MMR、阈值检索、重排序
- 答案生成：RAG Prompt、多轮对话支持、置信度计算
- 完整系统：集成所有模块、会话管理、知识更新
- 反馈系统：评分、正确性标记、改进建议
- API 服务化：问答、文档上传、反馈、统计接口
- 监控评估：请求统计、延迟监控、置信度分布
- 扩展方向：多语言支持、权限控制、知识图谱