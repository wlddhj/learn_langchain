# 第8章：Memory 记忆管理

## 8.1 为什么需要记忆

LLM 是无状态的，每次调用都像"失忆"。记忆机制让应用能记住之前的对话：

```
用户: 我叫小明
AI: 你好小明！

用户: 我叫什么名字？
AI: ??? （没有记忆的话，AI 不知道答案）
```

## 8.2 现代方案：直接管理消息历史（推荐）

> **最佳实践**：使用 `ChatMessageHistory` + `MessagesPlaceholder`，这是当前 LangChain 推荐的方式。

```python
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 1. 存储会话历史
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 2. 创建带历史占位的 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    MessagesPlaceholder(variable_name="history"),  # 注入历史消息
    ("human", "{question}"),
])

# 3. 构建 chain
chain = prompt | ChatOpenAI(model="gpt-4o-mini")

# 4. 包装为带记忆的 chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 5. 使用（不同 session_id 隔离对话）
response1 = chain_with_history.invoke(
    {"question": "我叫小明"},
    config={"configurable": {"session_id": "user-1"}},
)
print(response1.content)

response2 = chain_with_history.invoke(
    {"question": "我叫什么名字？"},
    config={"configurable": {"session_id": "user-1"}},
)
print(response2.content)  # "你叫小明"

# 不同 session 互不影响
response3 = chain_with_history.invoke(
    {"question": "我叫什么名字？"},
    config={"configurable": {"session_id": "user-2"}},
)
print(response3.content)  # "抱歉，我不知道你的名字"（新会话没有历史）
```

## 8.3 消息历史的持久化

### Redis 持久化

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
    )
```

### 文件持久化

```python
from langchain_community.chat_message_histories import FileChatMessageHistory

def get_session_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(f"./chat_history/{session_id}.json")
```

### SQLAlchemy 持久化

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db",
    )
```

## 8.4 摘要记忆 (Conversation Summary)

当对话过长时，用 LLM 自动总结历史，节省 token：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.summarize import load_summarize_chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 摘要记忆的核心思路：
# 1. 当历史消息超过阈值时
# 2. 用 LLM 将早期对话总结为一条摘要消息
# 3. 用摘要 + 最近的消息作为完整历史

def summarize_messages(messages, llm, max_messages=10):
    """当消息数超过阈值时，总结早期消息"""
    if len(messages) <= max_messages:
        return messages

    # 需要总结的消息
    to_summarize = messages[:-max_messages]
    recent = messages[-max_messages:]

    # 生成摘要
    summary_prompt = "请总结以下对话内容，保留关键信息：\n\n"
    for msg in to_summarize:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        summary_prompt += f"{role}: {msg.content}\n"

    summary = llm.invoke(summary_prompt).content

    # 构建新的消息列表：摘要 + 最近消息
    return [
        SystemMessage(content=f"之前的对话摘要：{summary}")
    ] + recent
```

## 8.5 窗口记忆 (Sliding Window)

只保留最近 N 轮对话：

```python
from langchain_core.messages import HumanMessage, AIMessage

def trim_messages(messages, max_tokens=4000, llm=None):
    """保留最近的消息，确保不超过 token 限制"""
    trimmed = []
    total_tokens = 0

    # 从最新消息开始，向前累加
    for msg in reversed(messages):
        estimated_tokens = len(msg.content) // 4  # 粗略估算
        if total_tokens + estimated_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total_tokens += estimated_tokens

    return trimmed
```

## 8.6 在 RAG 中使用记忆

让 RAG 系统能理解上下文相关的追问：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", """基于以下上下文回答用户问题。
如果上下文中没有相关信息，请回答"根据已知信息无法回答"。

上下文：
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 注意：这里需要调整以适配 RunnableWithMessageHistory
# 因为 chain 的输入需要同时包含 context 和 question
```

## 8.7 记忆策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 完整历史 | 信息不丢失 | token 消耗大 | 短对话 |
| 窗口截断 | 简单高效 | 丢失早期信息 | 一般对话 |
| 摘要记忆 | 节省 token | 摘要可能丢失细节 | 长对话 |
| 混合策略 | 兼顾效果和成本 | 实现复杂 | 生产环境 |

## 8.8 实际生产建议

1. **设置 token 上限**：始终限制历史消息的 token 数
2. **使用持久化存储**：不要用 InMemory，用 Redis/DB
3. **会话过期**：设置会话超时时间，清理过期历史
4. **按用户隔离**：用 `session_id` 区分不同用户/会话
5. **混合策略**：最近 5 轮完整保留 + 早期对话摘要

## 8.9 本章小结

- 使用 `RunnableWithMessageHistory` 管理对话记忆
- `MessagesPlaceholder` 在 prompt 中预留历史消息位置
- 多种持久化方案：内存、文件、Redis、SQL
- 长对话使用摘要记忆或窗口截断控制 token 消耗
- 生产环境务必使用持久化存储 + token 限制
