"""
第24章 Demo 1：FastAPI 服务化基础

演示将 LangChain 应用封装为 FastAPI 服务。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def demo_fastapi_structure():
    """FastAPI 服务结构"""
    print("=" * 60)
    print("Demo 24-1 (1/3): FastAPI 服务结构")
    print("=" * 60)
    print()

    code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

# 创建 FastAPI 应用
app = FastAPI(title="LLM API Service")

# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 请求模型
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

# 响应模型
class ChatResponse(BaseModel):
    response: str
    session_id: str

# API 路由
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = await llm.ainvoke(request.message)
        return ChatResponse(
            response=response.content,
            session_id=request.session_id or "default",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}

# 运行: uvicorn main:app --reload
"""

    print("基础 FastAPI 服务代码：")
    print("-" * 60)
    print(code)


def demo_api_design():
    """API 设计要点"""
    print("=" * 60)
    print("Demo 24-1 (2/3): API 设计")
    print("=" * 60)
    print()

    print("API 设计最佳实践：")
    print("-" * 60)
    print("""
| 设计要点 | 说明 |
|---------|------|
| Pydantic 模型 | 类型安全的请求/响应 |
| 异步处理 | 使用 ainvoke 提高性能 |
| 错误处理 | HTTPException 返回错误 |
| 健康检查 | /health 端点 |
| 流式响应 | SSE 支持实时输出 |
| 速率限制 | 防止 API 过载 |
""")


def demo_streaming_api():
    """流式 API"""
    print("=" * 60)
    print("Demo 24-1 (3/3): 流式 API")
    print("=" * 60)
    print()

    code = """
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in llm.astream(request.message):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")
"""

    print("流式 API 代码：")
    print("-" * 60)
    print(code)


if __name__ == "__main__":
    demo_fastapi_structure()
    demo_api_design()
    demo_streaming_api()

    print("=" * 60)
    print("Demo 24-1 完成!")
    print()
    print("FastAPI 服务化要点：")
    print("  - Pydantic 模型定义请求/响应")
    print("  - 异步处理提高性能")
    print("  - 流式输出增强用户体验")
    print("  - 健康检查支持监控")
    print()
    print("运行命令：")
    print("  uvicorn main:app --reload")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)