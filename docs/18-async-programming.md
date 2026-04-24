# 第18章：异步编程深入

## 18.1 为什么需要异步

| 场景 | 同步问题 | 异步优势 |
|------|---------|---------|
| **并发请求** | 串行等待，响应慢 | 同时处理，吞吐量高 |
| **流式输出** | 需完整返回才显示 | 边生成边显示 |
| **I/O 操作** | 阻塞等待 | 不阻塞主线程 |
| **资源利用** | 单线程空闲等待 | 多任务并行 |

异步编程是**生产环境的标配**。

## 18.2 Runnable 的异步方法

所有 Runnable 组件都提供异步接口：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 同步方法
result = llm.invoke("你好")           # 单次调用
results = llm.batch(["你好", "再见"])  # 批量调用
for chunk in llm.stream("你好"):       # 流式输出
    print(chunk)

# 异步方法（对应）
result = await llm.ainvoke("你好")           # 异步单次
results = await llm.abatch(["你好", "再见"])  # 异步批量
async for chunk in llm.astream("你好"):       # 异步流式
    print(chunk)
```

### 方法对照表

| 同步方法 | 异步方法 | 说明 |
|---------|---------|------|
| `invoke()` | `ainvoke()` | 单次调用 |
| `batch()` | `abatch()` | 批量调用 |
| `stream()` | `astream()` | 流式输出 |
| `transform()` | `atransform()` | 转换输入 |

## 18.3 异步基础示例

### 单次异步调用

```python
import asyncio
from langchain_openai import ChatOpenAI

async def simple_async():
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 异步调用
    response = await llm.ainvoke("什么是机器学习？")
    print(response.content)

asyncio.run(simple_async())
```

### 异步批量调用

```python
async def async_batch():
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 异步批量：所有请求并行执行
    questions = [
        "什么是 RAG？",
        "什么是 Agent？",
        "什么是 LangGraph？",
    ]

    responses = await llm.abatch(questions)

    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r.content}\n")

asyncio.run(async_batch())
```

### 异步流式输出

```python
async def async_stream():
    llm = ChatOpenAI(model="gpt-4o-mini")

    print("流式输出:")
    async for chunk in llm.astream("写一首关于AI的短诗"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(async_stream())
```

## 18.4 异步 Chain

LCEL 自动支持异步：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义 chain（同步方式）
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")
parser = StrOutputParser()

chain = prompt | llm | parser

# 异步调用 chain
async def async_chain_example():
    result = await chain.ainvoke({"topic": "量子计算"})
    print(result)

asyncio.run(async_chain_example())

# 异步批量
async def async_chain_batch():
    topics = ["机器学习", "深度学习", "强化学习"]
    results = await chain.abatch([{"topic": t} for t in topics])
    for r in results:
        print(r)

asyncio.run(async_chain_batch())
```

## 18.5 异步工具

### 异步工具定义

```python
import httpx
from langchain_core.tools import tool

@tool
async def async_fetch_url(url: str) -> str:
    """异步获取网页内容。url: 网页地址"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        return response.text[:500]


# 使用 StructuredTool 定义异步工具
from langchain_core.tools import StructuredTool

async def async_search(query: str) -> str:
    """异步搜索"""
    await asyncio.sleep(1)  # 模拟异步操作
    return f"搜索结果: {query}"

async_search_tool = StructuredTool.from_function(
    coroutine=async_search,  # 异步函数
    name="async_search",
    description="异步搜索信息",
)
```

### 在 Agent 中使用异步工具

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[async_fetch_url],
)

# 异步运行 Agent
async def async_agent_example():
    result = await agent.ainvoke({
        "messages": [("user", "获取 https://example.com 的内容")]
    })
    print(result["messages"][-1].content)

asyncio.run(async_agent_example())
```

## 18.6 异步 RAG

### 异步文档加载

```python
from langchain_community.document_loaders import AsyncHtmlLoader

# 异步加载网页
loader = AsyncHtmlLoader(["https://example.com", "https://example.org"])
docs = await loader.aload()
```

### 异步向量检索

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

async def async_retrieval():
    embeddings = OpenAIEmbeddings()

    # 创建向量库（同步操作，通常在初始化时完成）
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # 异步检索
    retriever = vectorstore.as_retriever()
    docs = await retriever.ainvoke("什么是 RAG？")

    for doc in docs:
        print(doc.page_content[:100])

asyncio.run(async_retrieval())
```

### 异步 RAG Chain

```python
async def async_rag_chain():
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template("""
    基于以下内容回答问题：
    {context}

    问题：{question}
    """)

    def format_docs(docs):
        return "\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 异步调用
    result = await chain.ainvoke("什么是向量数据库？")
    print(result)

asyncio.run(async_rag_chain())
```

## 18.7 异步 LangGraph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def async_node(state: State):
    """异步节点"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", async_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# 异步运行
async def async_graph_example():
    result = await graph.ainvoke({
        "messages": [("user", "你好")]
    })
    print(result["messages"][-1].content)

asyncio.run(async_graph_example())

# 异步流式
async def async_graph_stream():
    async for event in graph.astream({
        "messages": [("user", "写一首诗")]
    }):
        for node, output in event.items():
            print(f"[{node}] {output}")

asyncio.run(async_graph_stream())
```

## 18.8 并发控制

### 同时运行多个独立任务

```python
async def concurrent_tasks():
    """并发执行多个独立任务"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 创建多个异步任务
    tasks = [
        llm.ainvoke("问题1：什么是AI？"),
        llm.ainvoke("问题2：什么是ML？"),
        llm.ainvoke("问题3：什么是DL？"),
    ]

    # 并发执行（同时发起）
    responses = await asyncio.gather(*tasks)

    for i, r in enumerate(responses, 1):
        print(f"问题{i}: {r.content[:50]}")

asyncio.run(concurrent_tasks())
```

### 带超时的异步调用

```python
async def with_timeout():
    """带超时控制的异步调用"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    try:
        response = await asyncio.wait_for(
            llm.ainvoke("写一个长故事"),
            timeout=30.0  # 30秒超时
        )
        print(response.content)
    except asyncio.TimeoutError:
        print("请求超时")

asyncio.run(with_timeout())
```

### 限制并发数量

```python
async def limited_concurrency():
    """限制并发数量"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    questions = [f"问题{i}" for i in range(20)]

    # 使用 semaphore 限制并发数
    semaphore = asyncio.Semaphore(5)  # 最多5个并发

    async def call_with_limit(q):
        async with semaphore:
            return await llm.ainvoke(q)

    tasks = [call_with_limit(q) for q in questions]
    responses = await asyncio.gather(*tasks)

    print(f"完成 {len(responses)} 个请求")

asyncio.run(limited_concurrency())
```

## 18.9 异步事件流

`astream_events` 提供最详细的异步追踪：

```python
async def async_events():
    """详细的异步事件追踪"""
    chain = prompt | llm | parser

    async for event in chain.astream_events(
        {"topic": "机器学习"},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_prompt_start":
            print("[Prompt 构建]")

        elif kind == "on_llm_start":
            print("[LLM 开始思考]")

        elif kind == "on_llm_stream":
            token = event["data"]["chunk"].content
            if token:
                print(token, end="", flush=True)

        elif kind == "on_llm_end":
            print("\n[LLM 完成]")

        elif kind == "on_parser_start":
            print("[解析开始]")

        elif kind == "on_parser_end":
            print("[解析完成]")

asyncio.run(async_events())
```

## 18.10 异步 vs 同步性能对比

```python
import time

async def compare_performance():
    """对比同步和异步的性能"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    questions = ["问题1", "问题2", "问题3", "问题4", "问题5"]

    # 同步批量（实际上是串行）
    start = time.time()
    sync_results = llm.batch(questions)
    sync_time = time.time() - start
    print(f"同步批量: {sync_time:.2f}s")

    # 异步批量（真正的并发）
    start = time.time()
    async_results = await llm.abatch(questions)
    async_time = time.time() - start
    print(f"异步批量: {async_time:.2f}s")

    print(f"性能提升: {sync_time / async_time:.1f}x")

asyncio.run(compare_performance())
```

典型结果：异步批量比同步批量快 **3-5倍**。

## 18.11 异步编程最佳实践

### 1. 全异步或全同步，避免混用

```python
# ❌ 错误：在异步函数中调用同步方法
async def bad_example():
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke("你好")  # 同步方法阻塞！
    return response

# ✅ 正确：全部使用异步方法
async def good_example():
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = await llm.ainvoke("你好")  # 异步方法
    return response
```

### 2. 使用 asyncio.gather 并发独立任务

```python
# ❌ 错误：串行 await
async def serial_await():
    r1 = await llm.ainvoke("问题1")
    r2 = await llm.ainvoke("问题2")  # 等待 r1 完成才开始
    return [r1, r2]

# ✅ 正确：并发 await
async def parallel_await():
    r1, r2 = await asyncio.gather(
        llm.ainvoke("问题1"),
        llm.ainvoke("问题2"),  # 同时开始
    )
    return [r1, r2]
```

### 3. Web 应用中使用异步

```python
# FastAPI 示例
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    # FastAPI 是异步框架，使用异步调用
    response = await llm.ainvoke(query.question)
    return {"answer": response.content}

@app.post("/stream")
async def stream_chat(query: Query):
    # 流式响应
    async def generate():
        async for chunk in llm.astream(query.question):
            if chunk.content:
                yield chunk.content

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/plain")
```

### 4. 合理设置超时

```python
async def with_retries():
    """带重试和超时的异步调用"""
    max_retries = 3
    timeout = 30.0

    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                llm.ainvoke("复杂问题"),
                timeout=timeout,
            )
            return response
        except asyncio.TimeoutError:
            print(f"第 {attempt + 1} 次超时")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)  # 等待后重试
```

## 18.12 常见异步陷阱

### 陷阱1：忘记 await

```python
# ❌ 错误：忘记 await
async def forgot_await():
    result = llm.ainvoke("你好")  # 没有 await，得到的是 coroutine 对象
    print(result)  # 打印 coroutine，不是结果

# ✅ 正确
async def correct_await():
    result = await llm.ainvoke("你好")
    print(result)
```

### 陷阱2：在同步代码中直接调用异步

```python
# ❌ 错误：同步函数直接调用异步
def sync_calling_async():
    result = llm.ainvoke("你好")  # 没有 await，不会执行
    return result

# ✅ 正确：使用 asyncio.run
def sync_wrapper():
    async def async_func():
        return await llm.ainvoke("你好")
    return asyncio.run(async_func())
```

### 陷阱3：阻塞异步循环

```python
# ❌ 错误：在异步循环中使用同步调用
async def bad_loop():
    for chunk in llm.stream("写诗"):  # 同步 stream！
        print(chunk)

# ✅ 正确：使用异步流
async def good_loop():
    async for chunk in llm.astream("写诗"):
        print(chunk)
```

## 18.13 本章小结

- 所有 Runnable 都有 `ainvoke`/`abatch`/`astream` 异步方法
- 使用 `await` 等待异步操作完成
- `asyncio.gather()` 实现并发执行
- LCEL chain 自动支持异步调用
- 工具可以用 `async def` 或 `StructuredTool(coroutine=...)` 定义异步版本
- LangGraph 支持异步节点和异步运行
- 使用 `asyncio.wait_for()` 设置超时
- 使用 `asyncio.Semaphore()` 限制并发数量
- `astream_events()` 提供详细的事件追踪
- Web 应用中应全链路使用异步
- 避免：忘记 await、同步阻塞异步、混用同步和异步