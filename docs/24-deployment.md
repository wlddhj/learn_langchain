# 第24章：部署与生产化

## 24.1 从开发到生产

| 维度 | 开发环境 | 生产环境 |
|------|---------|---------|
| **稳定性** | 允许失败 | 必须稳定 |
| **性能** | 不关心 | 必须优化 |
| **成本** | 不限制 | 必须控制 |
| **安全** | 不关注 | 必须保障 |
| **监控** | 可选 | 必须完善 |
| **部署** | 本地运行 | 服务化 |

## 24.2 API 服务化

### FastAPI 基础服务

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

app = FastAPI(title="LLM API Service")

# 初始化模型（启动时）
llm = ChatOpenAI(model="gpt-4o-mini")

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    try:
        response = await llm.ainvoke(request.message)
        return ChatResponse(
            response=response.content,
            session_id=request.session_id or "default",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    from fastapi.responses import StreamingResponse

    async def generate():
        async for chunk in llm.astream(request.message):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")


# 运行: uvicorn main:app --reload
```

### 带 Chain 的服务

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 预定义 Chain（启动时初始化）
prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")
chain = prompt | llm | StrOutputParser()

class ExplainRequest(BaseModel):
    topic: str

@app.post("/explain")
async def explain(request: ExplainRequest):
    """解释接口"""
    try:
        result = await chain.ainvoke({"topic": request.topic})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Agent API 服务

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

# 初始化 Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search],
)

class AgentRequest(BaseModel):
    message: str
    max_steps: int = 5

@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    """Agent 接口"""
    try:
        result = await agent.ainvoke(
            {"messages": [("user", request.message)]},
            config={"recursion_limit": request.max_steps},
        )
        return {
            "response": result["messages"][-1].content,
            "steps": len(result["messages"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### RAG API 服务

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 预初始化向量库（启动时）
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever()

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

class RAGRequest(BaseModel):
    question: str

@app.post("/rag")
async def rag_endpoint(request: RAGRequest):
    """RAG 接口"""
    try:
        result = await rag_chain.ainvoke(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 24.3 容器化部署

### Dockerfile

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
ENV LANGCHAIN_TRACING_V2=true
ENV PYTHONUNBUFFERED=1

# 运行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```txt
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langgraph>=0.2.0
langchain-chroma>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=production
    volumes:
      - ./data:/app/data
    depends_on:
      - redis

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### 构建和运行

```bash
# 构建
docker-compose build

# 运行
docker-compose up -d

# 查看日志
docker-compose logs -f api

# 停止
docker-compose down
```

## 24.4 水平扩展

### 负载均衡配置

```yaml
# docker-compose.yml with scaling
version: '3.8'

services:
  api:
    build: .
    deploy:
      replicas: 3  # 运行 3 个实例
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

  redis:
    image: redis:alpine
```

### nginx.conf

```nginx
# nginx.conf
upstream api_servers {
    least_conn;
    server api:8000;
    server api:8001;
    server api:8002;
}

server {
    listen 80;

    location / {
        proxy_pass http://api_servers;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
    }
}
```

### Kubernetes 部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: api
        image: my-registry/llm-api:latest
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
          requests:
            memory: "256Mi"
            cpu: "250m"

---
apiVersion: v1
kind: Service
metadata:
  name: llm-api-service
spec:
  selector:
    app: llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 24.5 持久化存储

### Redis 会话存储

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379",
    )

# 包装 Chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="message",
    history_messages_key="history",
)

@app.post("/chat/history")
async def chat_with_history(request: ChatRequest):
    """带历史记忆的聊天"""
    result = await chain_with_history.ainvoke(
        {"message": request.message},
        config={"configurable": {"session_id": request.session_id}},
    )
    return {"response": result}
```

### PostgreSQL 状态存储

```python
from langgraph.checkpoint.postgres import PostgresSaver

# LangGraph 状态持久化
with PostgresSaver.from_conn_string("postgresql://user:pass@localhost/db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    result = await graph.ainvoke(
        {"messages": [("user", request.message)]},
        config={"configurable": {"thread_id": request.session_id}},
    )
```

## 24.6 监控和日志

### 结构化日志

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "extra": getattr(record, 'extra', {}),
        })

logger = logging.getLogger("llm_api")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 使用
@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info("Chat request", extra={
        "session_id": request.session_id,
        "message_length": len(request.message),
    })

    result = await llm.ainvoke(request.message)

    logger.info("Chat response", extra={
        "session_id": request.session_id,
        "response_length": len(result.content),
    })

    return {"response": result.content}
```

### Prometheus 监控

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# 定义指标
request_count = Counter('llm_requests_total', 'Total LLM requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')
token_count = Counter('llm_tokens_total', 'Total tokens used', ['type'])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time
    start = time.time()

    response = await call_next(request)

    latency = time.time() - start
    request_latency.observe(latency)
    request_count.inc()

    return response

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")
```

### 健康检查

```python
@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

@app.get("/health/ready")
async def ready():
    """就绪检查"""
    # 检查依赖服务
    try:
        # 检查 Redis
        redis_client.ping()

        # 检查 LLM API
        await llm.ainvoke("ping")

        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}
```

## 24.7 错误处理

### 全局异常处理

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal error occurred",
            "request_id": request.headers.get("X-Request-ID"),
        },
    )
```

### 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # 每分钟最多 10 次
async def chat(request: Request, chat_request: ChatRequest):
    result = await llm.ainvoke(chat_request.message)
    return {"response": result.content}
```

### 超时处理

```python
import asyncio

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await asyncio.wait_for(
            llm.ainvoke(request.message),
            timeout=30.0,
        )
        return {"response": response.content}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
```

## 24.8 安全配置

### API Key 管理

```python
import os
from fastapi import Header, HTTPException

# 环境变量存储 API Key
API_KEYS = os.getenv("API_KEYS", "").split(",")

async def verify_api_key(x_api_key: str = Header(None)):
    """验证 API Key"""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/chat")
async def chat(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    result = await llm.ainvoke(request.message)
    return {"response": result.content}
```

### HTTPS 配置

```yaml
# docker-compose.yml with SSL
services:
  api:
    build: .
    ports:
      - "8000:8000"

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
```

## 24.9 CI/CD 配置

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build Docker image
      run: docker build -t my-registry/llm-api:${{ github.sha }} .

    - name: Push to registry
      run: |
        docker login -u ${{ secrets.REGISTRY_USER }} -p ${{ secrets.REGISTRY_PASS }} my-registry
        docker push my-registry/llm-api:${{ github.sha }}

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/llm-api api=my-registry/llm-api:${{ github.sha }}
```

## 24.10 生产配置清单

| 配置项 | 说明 |
|--------|------|
| **API 服务** | FastAPI/Flask 封装 |
| **容器化** | Docker + Docker Compose |
| **水平扩展** | 负载均衡 + 多实例 |
| **持久化** | Redis/PostgreSQL |
| **监控** | Prometheus + Grafana |
| **日志** | 结构化日志 |
| **健康检查** | /health + /health/ready |
| **错误处理** | 全局异常处理 |
| **速率限制** | slowapi 或自定义 |
| **超时设置** | asyncio.wait_for |
| **安全** | API Key + HTTPS |
| **CI/CD** | GitHub Actions/Jenkins |

## 24.11 本章小结

- 使用 FastAPI 封装 LLM 应用为 API 服务
- Docker 容器化部署，便于移植和扩展
- Docker Compose 或 Kubernetes 实现水平扩展
- Redis/PostgreSQL 持久化会话和状态
- Prometheus + 结构化日志实现监控
- 配置健康检查、错误处理、速率限制、超时
- API Key 认证和 HTTPS 保障安全
- CI/CD 实现自动化部署
- 生产化需要：稳定性、性能、成本、安全、监控全面考虑