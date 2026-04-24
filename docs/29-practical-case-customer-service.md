# 第29章：智能客服系统实战

## 29.1 项目概述

### 系统目标

构建一个完整的智能客服系统，具备以下功能：
- 多轮对话管理
- 知识库问答（RAG）
- 意图识别与分类
- 工单系统对接
- 情绪分析与响应
- 实时监控与评估

### 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                     用户界面层                           │
│         (Web/移动端/API/社交媒体)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     接入层                               │
│         (API Gateway / 负载均衡)                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     核心服务层                           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 意图识别  │  │ 对话管理  │  │ 知识问答  │           │
│  └───────────┘  └───────────┘  └───────────┘           │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 情绪分析  │  │ 工单系统  │  │ Agent执行 │           │
│  └───────────┘  └───────────┘  └───────────┘           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     数据层                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ 向量库    │  │ 会话存储  │  │ 工单数据库│           │
│  └───────────┘  └───────────┘  └───────────┘           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     基础设施层                           │
│         (监控 / 日志 / 缓存 / 消息队列)                 │
└─────────────────────────────────────────────────────────┘
```

## 29.2 系统设计

### 数据模型设计

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class IntentType(str, Enum):
    """意图类型"""
    PRODUCT_QUERY = "product_query"       # 产品咨询
    ORDER_STATUS = "order_status"         # 订单查询
    COMPLAINT = "complaint"               # 投诉建议
    TECH_SUPPORT = "tech_support"         # 技术支持
    RETURN_REFUND = "return_refund"       # 退换货
    GENERAL_CHAT = "general_chat"         # 一般闲聊
    UNKNOWN = "unknown"                   # 未识别

class SentimentType(str, Enum):
    """情绪类型"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"

class Message(BaseModel):
    """消息模型"""
    id: str
    role: str  # user/assistant/system
    content: str
    timestamp: datetime
    intent: Optional[IntentType] = None
    sentiment: Optional[SentimentType] = None
    metadata: dict = {}

class Conversation(BaseModel):
    """会话模型"""
    id: str
    user_id: str
    messages: List[Message] = []
    status: str = "active"  # active/closed/transferred
    created_at: datetime
    updated_at: datetime
    metadata: dict = {}

class Ticket(BaseModel):
    """工单模型"""
    id: str
    conversation_id: str
    user_id: str
    type: IntentType
    priority: str  # low/medium/high/urgent
    status: str  # open/in_progress/resolved/closed
    title: str
    description: str
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    resolution: Optional[str] = None
```

### 目录结构

```
customer_service_system/
├── core/
│   ├── intent_classifier.py      # 意图识别
│   ├── sentiment_analyzer.py     # 情绪分析
│   ├── conversation_manager.py   # 对话管理
│   ├── knowledge_retriever.py    # 知识检索
│   └── response_generator.py     # 响应生成
├── agents/
│   ├── order_agent.py            # 订单查询 Agent
│   ├── product_agent.py          # 产品咨询 Agent
│   ├── support_agent.py          # 技术支持 Agent
│   └── ticket_agent.py           # 工单管理 Agent
├── tools/
│   ├── database_tools.py         # 数据库工具
│   ├── api_tools.py              # API 调用工具
│   └── notification_tools.py     # 通知工具
├── data/
│   ├── knowledge_base/           # 知识库文档
│   ├── intent_samples.json       # 意图识别样本
│   └── response_templates.json   # 响应模板
├── storage/
│   ├── vector_store.py           # 向量库管理
│   ├── session_store.py          # 会话存储
│   └── ticket_store.py           # 工单存储
├── api/
│   ├── main.py                   # FastAPI 主入口
│   ├── routes/
│   │   ├── chat.py               # 聊天路由
│   │   ├── tickets.py            # 工单路由
│   │   └── analytics.py          # 分析路由
│   └── middleware/
│   │   ├── auth.py               # 认证中间件
│   │   ├── rate_limit.py         # 限流中间件
├── monitoring/
│   ├── metrics.py                # 监控指标
│   ├── alerts.py                 # 告警系统
│   └── dashboard.py              # 监控仪表盘
├── config/
│   ├── settings.py               # 配置管理
│   └── prompts.py                # Prompt 配置
├── tests/
│   ├── test_intent.py
│   ├── test_conversation.py
│   └── test_api.py
└── requirements.txt
```

## 29.3 核心模块实现

### 意图识别器

```python
# core/intent_classifier.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List
from enum import Enum

class IntentType(str, Enum):
    PRODUCT_QUERY = "product_query"
    ORDER_STATUS = "order_status"
    COMPLAINT = "complaint"
    TECH_SUPPORT = "tech_support"
    RETURN_REFUND = "return_refund"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"

class IntentResult(BaseModel):
    intent: IntentType
    confidence: float
    sub_intents: List[str] = []
    keywords: List[str] = []

class IntentClassifier:
    """意图识别器"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        self.prompt = ChatPromptTemplate.from_template("""
        分析用户消息的意图，选择最合适的分类：
        
        用户消息：{message}
        
        可选意图类型：
        - product_query: 产品咨询（询问产品信息、功能、价格等）
        - order_status: 订单查询（查询订单状态、物流信息等）
        - complaint: 投诉建议（投诉服务问题、提出建议等）
        - tech_support: 技术支持（技术问题、使用问题等）
        - return_refund: 退换货（退货、换货、退款请求等）
        - general_chat: 一般闲聊（问候、闲聊等）
        - unknown: 未识别
        
        请输出：
        1. 主要意图类型
        2. 置信度（0-1）
        3. 子意图列表
        4. 关键词列表
        
        输出 JSON 格式。
        """)
    
    def classify(self, message: str) -> IntentResult:
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=IntentResult)
        
        chain = self.prompt | self.llm | parser
        return chain.invoke({"message": message})
    
    def classify_batch(self, messages: List[str]) -> List[IntentResult]:
        """批量识别"""
        return [self.classify(msg) for msg in messages]
    
    def get_intent_handler(self, intent: IntentType) -> str:
        """获取意图对应的处理器"""
        handlers = {
            IntentType.PRODUCT_QUERY: "product_agent",
            IntentType.ORDER_STATUS: "order_agent",
            IntentType.COMPLAINT: "ticket_agent",
            IntentType.TECH_SUPPORT: "support_agent",
            IntentType.RETURN_REFUND: "ticket_agent",
            IntentType.GENERAL_CHAT: "chat_agent",
            IntentType.UNKNOWN: "default_agent",
        }
        return handlers.get(intent, "default_agent")


# 使用示例
classifier = IntentClassifier()

# 测试意图识别
test_messages = [
    "我想查询我的订单状态",
    "这个产品多少钱？",
    "我要退货",
    "怎么使用这个功能？",
    "你好",
]

for msg in test_messages:
    result = classifier.classify(msg)
    print(f"消息: {msg}")
    print(f"意图: {result.intent}, 置信度: {result.confidence}")
    print(f"处理器: {classifier.get_intent_handler(result.intent)}")
    print("---")
```

### 情绪分析器

```python
# core/sentiment_analyzer.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"

class SentimentResult(BaseModel):
    sentiment: SentimentType
    score: float  # -1 到 1
    keywords: List[str] = []
    suggestion: str = ""

class SentimentAnalyzer:
    """情绪分析器"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        self.prompt = ChatPromptTemplate.from_template("""
        分析用户消息的情绪：
        
        用户消息：{message}
        
        可选情绪类型：
        - positive: 积极情绪（满意、感谢、开心等）
        - neutral: 中性情绪（一般询问、无明显情绪等）
        - negative: 负面情绪（不满、抱怨等）
        - angry: 愤怒情绪（强烈不满、投诉等）
        
        请输出：
        1. 情绪类型
        2. 情绪分数（-1到1，-1为极度负面，1为极度正面）
        3. 关键词列表（表达情绪的关键词）
        4. 响应建议（如何更好地回应这个情绪）
        
        输出 JSON 格式。
        """)
    
    def analyze(self, message: str) -> SentimentResult:
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=SentimentResult)
        
        chain = self.prompt | self.llm | parser
        return chain.invoke({"message": message})
    
    def get_response_strategy(self, sentiment: SentimentType) -> str:
        """根据情绪获取响应策略"""
        strategies = {
            SentimentType.POSITIVE: "友好热情回应，鼓励用户",
            SentimentType.NEUTRAL: "专业高效回应，简洁明了",
            SentimentType.NEGATIVE: "耐心安抚，提供解决方案",
            SentimentType.ANGRY: "优先安抚，快速处理，必要时转人工",
        }
        return strategies.get(sentiment, "专业回应")
    
    def should_transfer_to_human(self, result: SentimentResult) -> bool:
        """判断是否需要转人工"""
        # 情绪分数低于 -0.5 时转人工
        return result.score < -0.5


# 使用示例
analyzer = SentimentAnalyzer()

test_messages = [
    "非常满意你们的服务！",
    "我想问一下这个产品",
    "服务太差了，等了很久",
    "这什么垃圾产品！我要投诉！",
]

for msg in test_messages:
    result = analyzer.analyze(msg)
    print(f"消息: {msg}")
    print(f"情绪: {result.sentiment}, 分数: {result.score}")
    print(f"策略: {analyzer.get_response_strategy(result.sentiment)}")
    print(f"需转人工: {analyzer.should_transfer_to_human(result)}")
    print("---")
```

### 对话管理器

```python
# core/conversation_manager.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, Optional
from datetime import datetime
import uuid

class ConversationManager:
    """对话管理器"""
    
    def __init__(self, llm, max_history: int = 10):
        self.llm = llm
        self.max_history = max_history
        self.conversations = {}  # session_id -> messages
    
    def create_conversation(self, user_id: str) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = {
            "user_id": user_id,
            "messages": [],
            "created_at": datetime.now(),
            "metadata": {},
        }
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息"""
        if session_id not in self.conversations:
            raise ValueError("Session not found")
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
        }
        
        self.conversations[session_id]["messages"].append(message)
        
        # 截断历史，避免过长
        if len(self.conversations[session_id]["messages"]) > self.max_history * 2:
            # 保留最近的对话
            self.conversations[session_id]["messages"] = \
                self.conversations[session_id]["messages"][-self.max_history * 2:]
    
    def get_history(self, session_id: str) -> List[BaseMessage]:
        """获取对话历史"""
        if session_id not in self.conversations:
            return []
        
        messages = []
        for msg in self.conversations[session_id]["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        return messages
    
    def get_context_summary(self, session_id: str) -> str:
        """获取对话摘要"""
        history = self.get_history(session_id)
        
        if not history:
            return "无历史对话"
        
        summary_prompt = ChatPromptTemplate.from_template("""
        总结以下对话历史的关键信息：
        
        {history}
        
        请输出：
        1. 用户主要需求
        2. 已解决的问题
        3. 待解决的问题
        4. 用户情绪变化
        """)
        
        history_text = "\n".join([
            f"{msg.type}: {msg.content}" for msg in history
        ])
        
        return (summary_prompt | self.llm).invoke({"history": history_text}).content
    
    def close_conversation(self, session_id: str):
        """关闭会话"""
        if session_id in self.conversations:
            self.conversations[session_id]["status"] = "closed"
            self.conversations[session_id]["closed_at"] = datetime.now()
    
    def get_conversation_stats(self, session_id: str) -> dict:
        """获取会话统计"""
        if session_id not in self.conversations:
            return {}
        
        conv = self.conversations[session_id]
        messages = conv["messages"]
        
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m["role"] == "user"]),
            "assistant_messages": len([m for m in messages if m["role"] == "assistant"]),
            "duration": (datetime.now() - conv["created_at"]).total_seconds(),
        }


# 使用示例
manager = ConversationManager(ChatOpenAI(model="gpt-4o-mini"))

# 创建会话
session_id = manager.create_conversation("user_123")

# 添加对话
manager.add_message(session_id, "user", "我想查询我的订单")
manager.add_message(session_id, "assistant", "好的，请提供您的订单号")

# 继续对话
manager.add_message(session_id, "user", "订单号是 12345")
manager.add_message(session_id, "assistant", "您的订单正在配送中...")

# 获取历史
history = manager.get_history(session_id)
print("对话历史:", [msg.content for msg in history])

# 获取摘要
summary = manager.get_context_summary(session_id)
print("对话摘要:", summary)
```

### 知识检索器

```python
# core/knowledge_retriever.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List

class KnowledgeRetriever:
    """知识检索器"""
    
    def __init__(
        self,
        knowledge_dir: str,
        persist_dir: str = "./chroma_db",
    ):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.persist_dir = persist_dir
        
        # 加载或创建向量库
        self.vectorstore = self._load_or_create_vectorstore(knowledge_dir)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # RAG Chain
        self.rag_prompt = ChatPromptTemplate.from_template("""
        基于以下知识库内容回答用户问题。如果内容中没有相关信息，请说明。
        
        知识库内容：
        {context}
        
        用户问题：{question}
        
        请：
        1. 基于知识库内容回答
        2. 引用具体来源
        3. 如果知识库中没有相关信息，明确说明"知识库中暂无此信息"
        """)
        
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _load_or_create_vectorstore(self, knowledge_dir: str) -> Chroma:
        """加载或创建向量库"""
        try:
            # 尝试加载已有向量库
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
        except:
            # 创建新向量库
            return self._create_vectorstore(knowledge_dir)
    
    def _create_vectorstore(self, knowledge_dir: str) -> Chroma:
        """创建向量库"""
        # 加载文档
        loader = DirectoryLoader(
            knowledge_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        docs = loader.load()
        
        # 切分文档
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(docs)
        
        # 创建向量库
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
    
    def _format_docs(self, docs: List) -> str:
        """格式化文档"""
        return "\n\n".join([
            f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        ])
    
    def query(self, question: str) -> dict:
        """查询知识库"""
        # 检索
        docs = self.retriever.invoke(question)
        
        # 生成回答
        answer = self.chain.invoke(question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata for doc in docs],
            "retrieved_content": [doc.page_content for doc in docs],
        }
    
    def add_knowledge(self, content: str, metadata: dict = {}):
        """添加新知识"""
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata],
        )
    
    def update_knowledge(self, knowledge_dir: str):
        """更新知识库"""
        # 重建向量库
        self.vectorstore = self._create_vectorstore(knowledge_dir)
        self.retriever = self.vectorstore.as_retriever()


# 使用示例
retriever = KnowledgeRetriever("./knowledge_base")

# 查询
result = retriever.query("退货政策是什么？")
print("回答:", result["answer"])
print("来源:", result["sources"])

# 添加新知识
retriever.add_knowledge(
    "新产品 X 于 2024 年发布，价格 999 元。",
    {"source": "product_update.txt"},
)
```

### 响应生成器

```python
# core/response_generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Optional

class ResponseGenerator:
    """响应生成器"""
    
    def __init__(self, llm, templates: dict = None):
        self.llm = llm
        self.templates = templates or self._default_templates()
    
    def _default_templates(self) -> dict:
        """默认响应模板"""
        return {
            "greeting": "您好！我是智能客服助手，很高兴为您服务。请问有什么可以帮助您？",
            "farewell": "感谢您的咨询，祝您生活愉快！如有其他问题，随时欢迎咨询。",
            "transfer": "为了更好地解决您的问题，我将为您转接人工客服，请稍候...",
            "unknown": "抱歉，我暂时无法理解您的问题。请您详细描述一下，或选择转接人工客服。",
            "apology": "抱歉给您带来不便，我们会尽快为您处理这个问题。",
        }
    
    def generate(
        self,
        user_message: str,
        intent: str,
        sentiment: str,
        knowledge_answer: Optional[str] = None,
        history: List = None,
    ) -> str:
        """生成响应"""
        
        # 根据意图选择生成策略
        if intent == "general_chat":
            return self._generate_chat_response(user_message, history)
        
        elif intent in ["product_query", "tech_support"]:
            if knowledge_answer:
                return self._format_knowledge_response(
                    knowledge_answer, sentiment
                )
            else:
                return self.templates["unknown"]
        
        elif intent == "complaint":
            return self._generate_complaint_response(
                user_message, sentiment, history
            )
        
        elif intent == "order_status":
            # 这里需要调用订单查询 Agent
            return "正在查询您的订单信息..."
        
        elif intent == "return_refund":
            return self._generate_return_response(user_message, sentiment)
        
        else:
            return self.templates["unknown"]
    
    def _generate_chat_response(self, message: str, history: List) -> str:
        """生成闲聊响应"""
        prompt = ChatPromptTemplate.from_template("""
        作为友好的客服助手，回复用户的闲聊消息。
        
        对话历史：{history}
        用户消息：{message}
        
        要求：
        - 友好、热情
        - 简洁、自然
        - 可以适当引导用户咨询具体问题
        """)
        
        history_text = "\n".join([
            f"{msg.type}: {msg.content}" for msg in (history or [])
        ])
        
        return (prompt | self.llm).invoke({
            "history": history_text,
            "message": message,
        }).content
    
    def _format_knowledge_response(
        self,
        knowledge_answer: str,
        sentiment: str,
    ) -> str:
        """格式化知识库回答"""
        # 根据情绪调整语气
        if sentiment == "negative":
            prefix = "感谢您的耐心等待，以下是相关信息：\n"
        elif sentiment == "positive":
            prefix = "很高兴为您解答：\n"
        else:
            prefix = "以下是相关信息：\n"
        
        return prefix + knowledge_answer
    
    def _generate_complaint_response(
        self,
        message: str,
        sentiment: str,
        history: List,
    ) -> str:
        """生成投诉响应"""
        # 强负面情绪需要安抚
        if sentiment == "angry":
            return self.templates["apology"] + "\n我们非常重视您的反馈，正在为您创建工单处理..."
        
        prompt = ChatPromptTemplate.from_template("""
        作为客服助手，妥善处理用户的投诉建议。
        
        用户投诉：{message}
        
        要求：
        - 表达理解和歉意
        - 提供解决方案或处理承诺
        - 询问是否需要进一步帮助
        """)
        
        return (prompt | self.llm).invoke({"message": message}).content
    
    def _generate_return_response(
        self,
        message: str,
        sentiment: str,
    ) -> str:
        """生成退换货响应"""
        prompt = ChatPromptTemplate.from_template("""
        处理用户的退换货请求。
        
        用户消息：{message}
        
        请：
        1. 确认退换货意向
        2. 说明退换货流程
        3. 询问订单号或购买信息
        
        如果用户情绪不佳，先表达歉意和耐心。
        """)
        
        return (prompt | self.llm).invoke({"message": message}).content
    
    def get_template(self, template_name: str) -> str:
        """获取模板响应"""
        return self.templates.get(template_name, "")


# 使用示例
generator = ResponseGenerator(ChatOpenAI(model="gpt-4o-mini"))

response = generator.generate(
    user_message="这个产品质量太差了",
    intent="complaint",
    sentiment="negative",
)
print("响应:", response)
```

## 29.4 Agent 模块实现

### 订单查询 Agent

```python
# agents/order_agent.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional

class OrderInfo(BaseModel):
    order_id: str
    status: str
    items: list
    total_price: float
    shipping_info: dict
    estimated_delivery: Optional[str]

# 工具定义
@tool
def query_order_by_id(order_id: str) -> str:
    """根据订单号查询订单信息"""
    # 模拟数据库查询
    mock_orders = {
        "12345": {
            "order_id": "12345",
            "status": "配送中",
            "items": ["产品A", "产品B"],
            "total_price": 199.0,
            "shipping_info": {"快递": "顺丰", "快递单号": "SF123456"},
            "estimated_delivery": "2024-01-15",
        },
        "67890": {
            "order_id": "67890",
            "status": "已签收",
            "items": ["产品C"],
            "total_price": 99.0,
            "shipping_info": {"快递": "京东", "快递单号": "JD678901"},
            "estimated_delivery": "2024-01-10",
        },
    }
    
    order = mock_orders.get(order_id)
    if order:
        return str(order)
    return f"未找到订单号 {order_id} 的信息"

@tool
def query_orders_by_user(user_id: str) -> str:
    """根据用户ID查询所有订单"""
    # 模拟查询
    return f"用户 {user_id} 的订单列表: [12345, 67890]"

@tool
def check_shipping_status(tracking_number: str) -> str:
    """查询物流状态"""
    return f"快递单号 {tracking_number}: 正在派送，预计今天送达"

class OrderAgent:
    """订单查询 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent = create_react_agent(
            model=self.llm,
            tools=[query_order_by_id, query_orders_by_user, check_shipping_status],
        )
    
    def query(self, user_input: str, user_id: str = None) -> dict:
        """查询订单"""
        # 构建输入
        if user_id:
            prompt = f"用户ID: {user_id}, 用户问题: {user_input}"
        else:
            prompt = user_input
        
        result = self.agent.invoke({"messages": [("user", prompt)]})
        
        return {
            "response": result["messages"][-1].content,
            "tool_calls": self._extract_tool_calls(result),
        }
    
    def _extract_tool_calls(self, result: dict) -> list:
        """提取工具调用记录"""
        calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                    })
        return calls


# 使用示例
order_agent = OrderAgent()

# 查询具体订单
result = order_agent.query("查询订单 12345 的状态")
print("回答:", result["response"])

# 查询用户所有订单
result = order_agent.query("我最近有什么订单？", user_id="user_123")
print("回答:", result["response"])
```

### 产品咨询 Agent

```python
# agents/product_agent.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_products(keyword: str) -> str:
    """搜索产品"""
    # 模拟产品数据库
    products = [
        {"name": "产品A", "price": 99, "category": "电子产品"},
        {"name": "产品B", "price": 199, "category": "电子产品"},
        {"name": "产品C", "price": 49, "category": "家居用品"},
    ]
    
    results = [p for p in products if keyword.lower() in p["name"].lower()]
    return str(results) if results else "未找到相关产品"

@tool
def get_product_detail(product_id: str) -> str:
    """获取产品详情"""
    mock_details = {
        "A001": {
            "name": "产品A",
            "price": 99,
            "description": "高质量电子产品",
            "stock": 100,
            "specs": {"颜色": "黑色", "尺寸": "中号"},
        },
    }
    return str(mock_details.get(product_id, "未找到产品详情"))

@tool
def check_product_availability(product_id: str, region: str) -> str:
    """检查产品是否有货"""
    return f"产品 {product_id} 在 {region} 地区有货，库存充足"

class ProductAgent:
    """产品咨询 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent = create_react_agent(
            model=self.llm,
            tools=[search_products, get_product_detail, check_product_availability],
        )
    
    def consult(self, user_input: str) -> dict:
        """产品咨询"""
        result = self.agent.invoke({"messages": [("user", user_input)]})
        
        return {
            "response": result["messages"][-1].content,
            "tool_calls": self._extract_tool_calls(result),
        }
    
    def _extract_tool_calls(self, result: dict) -> list:
        calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append({"tool": tc["name"], "args": tc["args"]})
        return calls


# 使用示例
product_agent = ProductAgent()

result = product_agent.consult("我想买个电子产品，推荐一下")
print("回答:", result["response"])
```

### 工单管理 Agent

```python
# agents/ticket_agent.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime
import uuid

# 模拟工单数据库
tickets_db = {}

@tool
def create_ticket(
    user_id: str,
    title: str,
    description: str,
    priority: str = "medium",
) -> str:
    """创建工单"""
    ticket_id = str(uuid.uuid4())[:8]
    
    ticket = {
        "id": ticket_id,
        "user_id": user_id,
        "title": title,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }
    
    tickets_db[ticket_id] = ticket
    return f"工单已创建，工单号: {ticket_id}"

@tool
def query_ticket(ticket_id: str) -> str:
    """查询工单状态"""
    ticket = tickets_db.get(ticket_id)
    if ticket:
        return str(ticket)
    return f"未找到工单 {ticket_id}"

@tool
def update_ticket_status(ticket_id: str, status: str) -> str:
    """更新工单状态"""
    if ticket_id in tickets_db:
        tickets_db[ticket_id]["status"] = status
        tickets_db[ticket_id]["updated_at"] = datetime.now().isoformat()
        return f"工单 {ticket_id} 状态已更新为 {status}"
    return f"未找到工单 {ticket_id}"

@tool
def escalate_ticket(ticket_id: str, reason: str) -> str:
    """升级工单"""
    if ticket_id in tickets_db:
        tickets_db[ticket_id]["priority"] = "urgent"
        tickets_db[ticket_id]["escalation_reason"] = reason
        return f"工单 {ticket_id} 已升级为紧急，原因: {reason}"
    return f"未找到工单 {ticket_id}"

class TicketAgent:
    """工单管理 Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent = create_react_agent(
            model=self.llm,
            tools=[create_ticket, query_ticket, update_ticket_status, escalate_ticket],
        )
    
    def handle_issue(self, user_input: str, user_id: str) -> dict:
        """处理问题"""
        result = self.agent.invoke({
            "messages": [("user", f"用户 {user_id}: {user_input}")]
        })
        
        return {
            "response": result["messages"][-1].content,
            "tool_calls": self._extract_tool_calls(result),
        }
    
    def _extract_tool_calls(self, result: dict) -> list:
        calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append({"tool": tc["name"], "args": tc["args"]})
        return calls


# 使用示例
ticket_agent = TicketAgent()

result = ticket_agent.handle_issue(
    "我要投诉产品质量问题",
    "user_123"
)
print("回答:", result["response"])
print("工具调用:", result["tool_calls"])
```

## 29.5 统一客服系统

### 主控制器

```python
# core/customer_service.py
from langchain_openai import ChatOpenAI
from core.intent_classifier import IntentClassifier
from core.sentiment_analyzer import SentimentAnalyzer
from core.conversation_manager import ConversationManager
from core.knowledge_retriever import KnowledgeRetriever
from core.response_generator import ResponseGenerator
from agents.order_agent import OrderAgent
from agents.product_agent import ProductAgent
from agents.ticket_agent import TicketAgent
from typing import Optional
import uuid

class CustomerServiceSystem:
    """智能客服系统"""
    
    def __init__(
        self,
        knowledge_dir: str = "./knowledge_base",
    ):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 核心模块
        self.intent_classifier = IntentClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_manager = ConversationManager(self.llm)
        self.knowledge_retriever = KnowledgeRetriever(knowledge_dir)
        self.response_generator = ResponseGenerator(self.llm)
        
        # Agent 模块
        self.order_agent = OrderAgent()
        self.product_agent = ProductAgent()
        self.ticket_agent = TicketAgent()
    
    def create_session(self, user_id: str) -> str:
        """创建会话"""
        return self.conversation_manager.create_conversation(user_id)
    
    def chat(
        self,
        session_id: str,
        user_message: str,
        user_id: Optional[str] = None,
    ) -> dict:
        """处理用户消息"""
        # 1. 获取对话历史
        history = self.conversation_manager.get_history(session_id)
        
        # 2. 意图识别
        intent_result = self.intent_classifier.classify(user_message)
        
        # 3. 情绪分析
        sentiment_result = self.sentiment_analyzer.analyze(user_message)
        
        # 4. 根据意图选择处理路径
        response = self._route_by_intent(
            intent_result.intent,
            user_message,
            sentiment_result.sentiment,
            history,
            user_id,
        )
        
        # 5. 记录消息
        self.conversation_manager.add_message(session_id, "user", user_message)
        self.conversation_manager.add_message(session_id, "assistant", response)
        
        # 6. 检查是否需要转人工
        transfer_needed = self.sentiment_analyzer.should_transfer_to_human(
            sentiment_result
        )
        
        return {
            "session_id": session_id,
            "response": response,
            "intent": intent_result.intent,
            "sentiment": sentiment_result.sentiment,
            "confidence": intent_result.confidence,
            "transfer_needed": transfer_needed,
        }
    
    def _route_by_intent(
        self,
        intent: str,
        message: str,
        sentiment: str,
        history: list,
        user_id: Optional[str],
    ) -> str:
        """根据意图路由到对应处理器"""
        
        if intent == "order_status":
            # 订单查询
            result = self.order_agent.query(message, user_id)
            return result["response"]
        
        elif intent == "product_query":
            # 产品咨询 - 先查知识库，再调用 Agent
            knowledge_result = self.knowledge_retriever.query(message)
            
            if "知识库中暂无此信息" not in knowledge_result["answer"]:
                return self.response_generator._format_knowledge_response(
                    knowledge_result["answer"],
                    sentiment,
                )
            else:
                agent_result = self.product_agent.consult(message)
                return agent_result["response"]
        
        elif intent in ["complaint", "return_refund"]:
            # 投诉/退换货 - 创建工单
            result = self.ticket_agent.handle_issue(message, user_id or "unknown")
            
            # 添加情绪安抚
            if sentiment in ["negative", "angry"]:
                prefix = self.response_generator.get_template("apology") + "\n"
                return prefix + result["response"]
            
            return result["response"]
        
        elif intent == "tech_support":
            # 技术支持 - 知识库问答
            result = self.knowledge_retriever.query(message)
            return self.response_generator._format_knowledge_response(
                result["answer"],
                sentiment,
            )
        
        elif intent == "general_chat":
            # 闲聊
            return self.response_generator.generate(
                message,
                intent,
                sentiment,
                None,
                history,
            )
        
        else:
            # 未识别意图
            return self.response_generator.get_template("unknown")
    
    def close_session(self, session_id: str):
        """关闭会话"""
        self.conversation_manager.close_conversation(session_id)
    
    def get_session_stats(self, session_id: str) -> dict:
        """获取会话统计"""
        return self.conversation_manager.get_conversation_stats(session_id)
    
    def get_session_summary(self, session_id: str) -> str:
        """获取会话摘要"""
        return self.conversation_manager.get_context_summary(session_id)


# 使用示例
system = CustomerServiceSystem("./knowledge_base")

# 创建会话
session_id = system.create_session("user_123")

# 多轮对话
messages = [
    "你好",
    "我想查询我的订单 12345",
    "这个订单什么时候能到？",
    "我对这个产品质量不满意，我要投诉",
]

for msg in messages:
    result = system.chat(session_id, msg, "user_123")
    print(f"用户: {msg}")
    print(f"客服: {result['response']}")
    print(f"意图: {result['intent']}, 情绪: {result['sentiment']}")
    print("---")

# 获取统计
stats = system.get_session_stats(session_id)
print("会话统计:", stats)
```

## 29.6 API 服务化

```python
# api/main.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os

from core.customer_service import CustomerServiceSystem

app = FastAPI(title="智能客服 API")

# 初始化系统
system = CustomerServiceSystem(
    knowledge_dir=os.getenv("KNOWLEDGE_DIR", "./knowledge_base"),
)

# 请求模型
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    sentiment: str
    transfer_needed: bool

class SessionStatsResponse(BaseModel):
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int
    duration_seconds: float

# API 路由
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    # 创建或使用现有会话
    session_id = request.session_id or system.create_session(request.user_id)
    
    # 处理消息
    result = system.chat(
        session_id,
        request.message,
        request.user_id,
    )
    
    return ChatResponse(**result)

@app.post("/session/create")
async def create_session(user_id: str):
    """创建新会话"""
    session_id = system.create_session(user_id)
    return {"session_id": session_id}

@app.post("/session/close")
async def close_session(session_id: str):
    """关闭会话"""
    system.close_session(session_id)
    return {"status": "closed"}

@app.get("/session/stats/{session_id}", response_model=SessionStatsResponse)
async def get_session_stats(session_id: str):
    """获取会话统计"""
    stats = system.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStatsResponse(session_id=session_id, **stats)

@app.get("/session/summary/{session_id}")
async def get_session_summary(session_id: str):
    """获取会话摘要"""
    summary = system.get_session_summary(session_id)
    return {"session_id": session_id, "summary": summary}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# 运行: uvicorn api.main:app --reload
```

## 29.7 监控与评估

### 监控指标

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time

# 定义指标
chat_requests = Counter(
    'customer_service_chat_requests_total',
    'Total chat requests'
)

intent_distribution = Counter(
    'customer_service_intent_total',
    'Intent distribution',
    ['intent_type']
)

sentiment_distribution = Counter(
    'customer_service_sentiment_total',
    'Sentiment distribution',
    ['sentiment_type']
)

response_latency = Histogram(
    'customer_service_response_latency_seconds',
    'Response latency'
)

transfer_count = Counter(
    'customer_service_transfers_total',
    'Human transfers count'
)

active_sessions = Gauge(
    'customer_service_active_sessions',
    'Active sessions count'
)

def monitor_response(func):
    """响应监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        # 记录指标
        chat_requests.inc()
        response_latency.observe(time.time() - start_time)
        intent_distribution.labels(intent_type=result['intent']).inc()
        sentiment_distribution.labels(sentiment_type=result['sentiment']).inc()
        
        if result['transfer_needed']:
            transfer_count.inc()
        
        return result
    
    return wrapper

# 使用装饰器
class MonitoredCustomerServiceSystem(CustomerServiceSystem):
    @monitor_response
    def chat(self, session_id, user_message, user_id=None):
        return super().chat(session_id, user_message, user_id)
```

### 评估系统

```python
# monitoring/evaluation.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List

class ResponseEvaluation(BaseModel):
    accuracy: float  # 回答准确性
    helpfulness: float  # 帮助程度
    politeness: float  # 礼貌程度
    efficiency: float  # 效率（简洁程度）
    overall: float  # 综合评分
    issues: List[str]  # 问题列表

class CustomerServiceEvaluator:
    """客服系统评估器"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.eval_prompt = ChatPromptTemplate.from_template("""
        评估以下客服回答的质量：
        
        用户问题：{question}
        客服回答：{response}
        用户意图：{intent}
        用户情绪：{sentiment}
        
        请评估（0-1 分）：
        1. accuracy: 回答准确性（信息是否正确）
        2. helpfulness: 帮助程度（是否解决用户问题）
        3. politeness: 礼貌程度（语气是否友好）
        4. efficiency: 效率（回答是否简洁有效）
        5. overall: 综合评分
        
        同时列出存在的问题。
        
        输出 JSON 格式。
        """)
    
    def evaluate(
        self,
        question: str,
        response: str,
        intent: str,
        sentiment: str,
    ) -> ResponseEvaluation:
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=ResponseEvaluation)
        
        chain = self.eval_prompt | self.llm | parser
        return chain.invoke({
            "question": question,
            "response": response,
            "intent": intent,
            "sentiment": sentiment,
        })
    
    def evaluate_session(self, session_messages: List[dict]) -> dict:
        """评估整个会话"""
        evaluations = []
        
        for i in range(0, len(session_messages), 2):
            if i + 1 < len(session_messages):
                user_msg = session_messages[i]
                assistant_msg = session_messages[i + 1]
                
                eval_result = self.evaluate(
                    question=user_msg["content"],
                    response=assistant_msg["content"],
                    intent=user_msg.get("intent", "unknown"),
                    sentiment=user_msg.get("sentiment", "neutral"),
                )
                evaluations.append(eval_result)
        
        # 计算平均分数
        if evaluations:
            avg_scores = {
                "accuracy": sum(e.accuracy for e in evaluations) / len(evaluations),
                "helpfulness": sum(e.helpfulness for e in evaluations) / len(evaluations),
                "politeness": sum(e.politeness for e in evaluations) / len(evaluations),
                "efficiency": sum(e.efficiency for e in evaluations) / len(evaluations),
                "overall": sum(e.overall for e in evaluations) / len(evaluations),
            }
        else:
            avg_scores = {}
        
        return {
            "message_count": len(session_messages),
            "evaluations": evaluations,
            "average_scores": avg_scores,
        }
```

## 29.8 本章小结

- 系统架构：接入层、核心服务层、数据层、基础设施层
- 核心模块：意图识别、情绪分析、对话管理、知识检索、响应生成
- Agent 模块：订单查询、产品咨询、工单管理
- 统一系统：整合所有模块，实现智能路由
- API 服务化：FastAPI 封装，支持会话管理
- 监控评估：Prometheus 指标、响应质量评估
- 扩展方向：语音客服、多语言支持、个性化服务