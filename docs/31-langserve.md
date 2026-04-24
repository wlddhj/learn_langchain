# 第31章：LangServe 服务化

## 31.1 LangServe 简介

LangServe 是 LangChain 官方提供的服务化工具，帮助开发者快速将 LangChain 应用部署为 REST API。

### 核心特性

| 特性 | 说明 |
|------|------|
| **快速部署** | 几行代码即可创建 API |
| **类型安全** | Pydantic 自动生成 Schema |
| **流式支持** | 内置 SSE 流式响应 |
| ** Playground** | 内置交互式 UI |
| **兼容 LangChain** | 直接集成 Runnable |
| **异步支持** | 高性能异步处理 |

### 安装

```bash
pip install langserve
pip install langserve[all]  # 安装所有依赖
```

## 31.2 快速开始

### 最简服务

```python
# server.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 创建 FastAPI 应用
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="简单的 LangChain API 服务",
)

# 创建 LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 添加路由
add_routes(app, llm, path="/chat")

# 运行: uvicorn server:app --reload
```

### 访问方式

```bash
# 调用 API
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "你好"}'

# 流式调用
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "你好"}'

# Playground UI
http://localhost:8000/chat/playground
```

## 31.3 路由配置详解

### Runnable 类型映射

| Runnable 类型 | API Endpoint | 说明 |
|---------------|-------------|------|
| `LLM` | `/invoke` | 单次调用 |
| `LLM` | `/stream` | 流式调用 |
| `Chain` | `/invoke` | Chain 执行 |
| `Chain` | `/stream` | Chain 流式 |
| `Agent` | `/invoke` | Agent 执行 |
| `Agent` | `/stream` | Agent 流式 |

### 基础路由配置

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

app = FastAPI(title="LangChain API")

# 1. LLM 路由
llm = ChatOpenAI(model="gpt-4o-mini")
add_routes(app, llm, path="/llm")

# 2. Chain 路由
prompt = ChatPromptTemplate.from_template("翻译成{language}: {text}")
chain = prompt | llm | StrOutputParser()
add_routes(app, chain, path="/translate")

# 3. 多个 Chain
summary_chain = (
    ChatPromptTemplate.from_template("用一句话总结: {text}")
    | llm
    | StrOutputParser()
)
add_routes(app, summary_chain, path="/summary")

# 4. 配置选项
add_routes(
    app,
    llm,
    path="/configured_llm",
    config_keys=["model", "temperature"],  # 可配置的参数
    input_type=str,
    output_type=str,
)

# 运行: uvicorn server:app --reload
```

### 带 Pydantic Schema

```python
from pydantic import BaseModel

# 输入类型
class TranslateInput(BaseModel):
    text: str
    language: str = "英文"

# 输出类型
class TranslateOutput(BaseModel):
    translation: str
    original: str

# Chain with Schema
def create_translate_chain():
    prompt = ChatPromptTemplate.from_template(
        "翻译成{language}: {text}"
    )
    
    def format_output(original, translation):
        return TranslateOutput(
            translation=translation,
            original=original,
        )
    
    return (
        {
            "text": lambda x: x.text,
            "language": lambda x: x.language,
        }
        | prompt
        | llm
        | StrOutputParser()
        | (lambda x: format_output(TranslateInput.__dict__, x))
    )

add_routes(app, create_translate_chain(), path="/translate_schema")
```

## 31.4 API 接口详解

### Invoke 接口

```python
# 客户端调用
import requests

# 调用 invoke
response = requests.post(
    "http://localhost:8000/llm/invoke",
    json={"input": "你好，介绍一下自己"},
)

result = response.json()
print(result["output"])

# 带配置调用
response = requests.post(
    "http://localhost:8000/llm/invoke",
    json={
        "input": "你好",
        "config": {
            "model": "gpt-4o",
            "temperature": 0.5,
        },
    },
)
```

### Stream 接口

```python
# SSE 流式调用
import httpx

async def stream_chat():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/llm/stream",
            json={"input": "讲一个故事"},
        ) as response:
            for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = line[5:]  # 去掉 "data:"
                    print(data)

# 使用客户端库
from langserve.client import RemoteRunnable

client = RemoteRunnable("http://localhost:8000/llm")

# 流式调用
for chunk in client.stream("讲一个故事"):
    print(chunk.content, end="", flush=True)
```

### Batch 接口

```python
# 批量调用
response = requests.post(
    "http://localhost:8000/llm/batch",
    json={
        "inputs": [
            "翻译成英文：你好",
            "翻译成英文：世界",
            "翻译成英文：再见",
        ],
    },
)

results = response.json()
for r in results["output"]:
    print(r)
```

## 31.5 Chain 服务化

### RAG Chain 服务

```python
# rag_server.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langserve import add_routes

app = FastAPI(title="RAG API Server")

# 初始化
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever()

# 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_prompt = ChatPromptTemplate.from_template("""
基于以下内容回答问题：
{context}

问题：{question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 添加路由
add_routes(app, rag_chain, path="/rag")

# 运行: uvicorn rag_server:app --reload
```

### Agent 服务

```python
# agent_server.py
from fastapi import FastAPI
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langserve import add_routes

app = FastAPI(title="Agent API Server")

# 工具定义
@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

@tool
def calculate(expression: str) -> str:
    """计算工具"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 创建 Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search, calculate],
)

# 添加路由
add_routes(app, agent, path="/agent")

# 运行: uvicorn agent_server:app --reload
```

### 多 Chain 服务

```python
# multi_chain_server.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

app = FastAPI(title="Multi-Chain API Server")

llm = ChatOpenAI(model="gpt-4o-mini")

# 翻译 Chain
translate_prompt = ChatPromptTemplate.from_template(
    "翻译成{language}: {text}"
)
translate_chain = translate_prompt | llm | StrOutputParser()
add_routes(app, translate_chain, path="/translate")

# 摘要 Chain
summary_prompt = ChatPromptTemplate.from_template(
    "用一句话总结: {text}"
)
summary_chain = summary_prompt | llm | StrOutputParser()
add_routes(app, summary_chain, path="/summary")

# 翻译+摘要 Chain（组合）
combined_chain = {
    "translation": translate_chain,
    "summary": summary_chain,
}
add_routes(app, combined_chain, path="/combined")

# 问答 Chain
qa_prompt = ChatPromptTemplate.from_template("""
请回答问题，如果不确定请说明：
{question}
""")
qa_chain = qa_prompt | llm | StrOutputParser()
add_routes(app, qa_chain, path="/qa")

# 运行: uvicorn multi_chain_server:app --reload
```

## 31.6 配置与自定义

### 配置选项

```python
from langserve import add_routes

# 可配置参数
add_routes(
    app,
    llm,
    path="/llm",
    config_keys=["model", "temperature", "max_tokens"],  # 可配置项
    enabled_endpoints=["invoke", "stream", "batch"],     # 启用的端点
    playground_type="chat",                               # Playground 类型
)

# 配置调用示例
response = requests.post(
    "http://localhost:8000/llm/invoke",
    json={
        "input": "你好",
        "config": {
            "configurable": {
                "model": "gpt-4o",
                "temperature": 0.7,
            },
        },
    },
)
```

### 自定义端点

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langserve import add_routes

app = FastAPI(title="Custom Endpoints Server")

llm = ChatOpenAI(model="gpt-4o-mini")

# 标准 LangServe 路由
add_routes(app, llm, path="/llm")

# 自定义端点
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.0

class ChatResponse(BaseModel):
    response: str
    tokens_used: int

@app.post("/custom_chat", response_model=ChatResponse)
async def custom_chat(request: ChatRequest):
    """自定义聊天端点"""
    llm_configured = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=request.temperature,
    )
    
    response = await llm_configured.ainvoke(request.message)
    
    return ChatResponse(
        response=response.content,
        tokens_used=response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
    )

# 流式自定义端点
@app.post("/custom_stream")
async def custom_stream(request: ChatRequest):
    """自定义流式端点"""
    from fastapi.responses import StreamingResponse
    
    llm_configured = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=request.temperature,
    )
    
    async def generate():
        async for chunk in llm_configured.astream(request.message):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 输入输出验证

```python
from pydantic import BaseModel, Field
from typing import List

# 输入 Schema
class RAGInput(BaseModel):
    question: str = Field(description="用户问题")
    top_k: int = Field(default=5, description="检索文档数量")
    language: str = Field(default="中文", description="回答语言")

# 输出 Schema
class RAGOutput(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# Chain with Schema
def create_rag_chain_with_schema():
    # ... Chain 定义
    
    # 添加验证
    return rag_chain.with_types(
        input_type=RAGInput,
        output_type=RAGOutput,
    )

add_routes(
    app,
    create_rag_chain_with_schema(),
    path="/rag_schema",
)
```

## 31.7 Playground UI

### Playground 配置

```python
from langserve import add_routes

# Chat Playground
add_routes(
    app,
    llm,
    path="/chat",
    playground_type="chat",  # 聊天类型
)

# Default Playground
add_routes(
    app,
    chain,
    path="/chain",
    playground_type="default",  # 默认类型
)

# 自定义 Playground 标题
add_routes(
    app,
    llm,
    path="/chat",
    playground_config={
        "title": "AI 助手",
        "description": "智能对话助手",
        "examples": [
            {"input": "你好"},
            {"input": "介绍一下自己"},
        ],
    },
)
```

### Playground 功能

```
Playground 提供的功能：
- invoke: 单次调用测试
- stream: 流式调用测试
- batch: 批量调用测试
- config: 参数配置
- history: 调用历史
```

## 31.8 客户端使用

### Python 客户端

```python
from langserve.client import RemoteRunnable

# 创建客户端
client = RemoteRunnable("http://localhost:8000/llm")

# invoke 调用
result = client.invoke("你好")
print(result)

# stream 调用
for chunk in client.stream("讲一个故事"):
    print(chunk.content, end="", flush=True)

# batch 调用
results = client.batch(["你好", "再见"])
for r in results:
    print(r)

# 带配置调用
result = client.invoke(
    "你好",
    config={
        "configurable": {
            "temperature": 0.5,
        },
    },
)
```

### 异步客户端

```python
from langserve.client import RemoteRunnable
import asyncio

async def async_chat():
    client = RemoteRunnable("http://localhost:8000/llm")
    
    # 异步 invoke
    result = await client.ainvoke("你好")
    print(result)
    
    # 异步 stream
    async for chunk in client.astream("讲一个故事"):
        print(chunk.content, end="", flush=True)
    
    # 异步 batch
    results = await client.abatch(["你好", "再见"])
    for r in results:
        print(r)

asyncio.run(async_chat())
```

### TypeScript/JavaScript 客户端

```typescript
// 使用 fetch API
const response = await fetch('http://localhost:8000/llm/invoke', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ input: '你好' }),
});

const data = await response.json();
console.log(data.output);

// 流式调用
const streamResponse = await fetch('http://localhost:8000/llm/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ input: '讲一个故事' }),
});

const reader = streamResponse.body?.getReader();
while (true) {
  const { done, value } = await reader?.read();
  if (done) break;
  console.log(new TextDecoder().decode(value));
}
```

## 31.9 生产部署

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 环境变量
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV PORT=8000

# 运行
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  langserve:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
```

### Kubernetes 部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langserve-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langserve
  template:
    metadata:
      labels:
        app: langserve
    spec:
      containers:
      - name: langserve
        image: my-registry/langserve:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: langserve-service
spec:
  selector:
    app: langserve
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 生产配置

```python
# production_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langserve import add_routes
import os

# 创建应用
app = FastAPI(
    title="Production LangServe",
    version="1.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}

# LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("TEMPERATURE", "0")),
)

# 添加路由
add_routes(
    app,
    llm,
    path="/chat",
    config_keys=["model", "temperature"],
)

# 运行配置
# uvicorn production_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 31.10 监控与日志

### 集成监控

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

app = FastAPI(title="LangServe with Monitoring")

# 指标
request_count = Counter('langserve_requests_total', 'Total requests')
response_time = Histogram('langserve_response_time', 'Response time')

# 监控中间件
@app.middleware("http")
async def monitor_middleware(request, call_next):
    import time
    start = time.time()
    
    response = await call_next(request)
    
    request_count.inc()
    response_time.observe(time.time() - start)
    
    return response

# 指标端点
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

# LangServe 路由
llm = ChatOpenAI(model="gpt-4o-mini")
add_routes(app, llm, path="/chat")
```

### LangSmith 集成

```python
import os

# 配置 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "langserve-production"

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes

app = FastAPI(title="LangServe with LangSmith")

llm = ChatOpenAI(model="gpt-4o-mini")
add_routes(app, llm, path="/chat")

# 所有调用自动追踪到 LangSmith
```

## 31.11 完整示例

### 完整服务示例

```python
# complete_server.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langserve import add_routes
import os

# 配置
app = FastAPI(
    title="Complete LangServe API",
    version="1.0",
    description="完整的 LangChain API 服务",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# 1. 简单聊天
add_routes(app, llm, path="/chat", playground_type="chat")

# 2. 翻译 Chain
translate_prompt = ChatPromptTemplate.from_template("翻译成{language}: {text}")
translate_chain = translate_prompt | llm | StrOutputParser()
add_routes(app, translate_chain, path="/translate")

# 3. 摘要 Chain
summary_prompt = ChatPromptTemplate.from_template("用一句话总结: {text}")
summary_chain = summary_prompt | llm | StrOutputParser()
add_routes(app, summary_chain, path="/summary")

# 4. RAG Chain
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
基于以下内容回答问题：
{context}

问题：{question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
add_routes(app, rag_chain, path="/rag")

# 5. Agent
@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索: {query}"

@tool
def calculate(expression: str) -> str:
    """计算工具"""
    try:
        return str(eval(expression))
    except:
        return "错误"

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search, calculate],
)
add_routes(app, agent, path="/agent")

# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}

# API Key 认证（可选）
async def verify_api_key(api_key: str = Depends(lambda: None)):
    # 实际实现认证逻辑
    pass

# 运行
# uvicorn complete_server:app --host 0.0.0.0 --port 8000 --reload
```

## 31.12 本章小结

- LangServe：快速将 LangChain 应用服务化
- add_routes：几行代码添加 API 路由
- API 端点：invoke/stream/batch 自动生成
- Playground：内置交互式 UI 测试
- 配置选项：config_keys、enabled_endpoints
- Schema：Pydantic 自动生成输入输出 Schema
- 客户端：RemoteRunnable Python 客户端
- 生产部署：Docker、Kubernetes、生产配置
- 监控：Prometheus、LangSmith 集成
- 完整示例：Chat、翻译、摘要、RAG、Agent 全覆盖