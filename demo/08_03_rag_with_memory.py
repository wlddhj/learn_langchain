"""
第8章 Demo 3：实战 —— RAG + 对话记忆的智能问答

综合运用 RAG 检索 + 对话记忆，构建能理解上下文追问的知识库问答。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
# embeddings = OpenAIEmbeddings(model="text-embedding-v2", api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, check_embedding_ctx_length=False, tiktoken_enabled=False,)

import dashscope


class DashScopeEmbeddings(Embeddings):
    """通义千问 Embeddings 实现"""

    def __init__(self, api_key=None, model="text-embedding-v1"):
        self.api_key = api_key or os.environ["QWEN_API_KEY"]
        self.model = model

        # 配置 dashscope
        import dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        from dashscope import TextEmbedding

        # 确保所有文本都是字符串
        texts = [str(text) for text in texts]

        # 分批处理，每批最多25个文本
        batch_size = 25
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 调用 API - 每个文本单独调用
            batch_embeddings = []
            for text in batch:
                response = TextEmbedding.call(
                    model=self.model,
                    input=text  # 直接传入单个字符串
                )

                # 检查响应
                if response.status_code == 200:
                    embedding = response.output['embeddings'][0]['embedding']
                    batch_embeddings.append(embedding)
                else:
                    print(f"Embedding API 错误: {response.code} - {response.message}")
                    # 返回零向量作为备选
                    zero_dim = 1536
                    batch_embeddings.append([0.0] * zero_dim)

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


# 使用自定义 embeddings
embeddings = DashScopeEmbeddings(api_key=QWEN_API_KEY)

# ============================================================
# 知识库
# ============================================================

def build_knowledge_base():
    """构建 AI 框架知识库"""
    docs = [
        Document(page_content="LangChain 是构建 LLM 应用的框架，支持 RAG、Agent、Chain 模式。"
                 "核心组件：Model I/O、Retrieval、Chains、Agents、Memory。"
                 "最新版本推荐使用 LCEL（管道语法）和 LangGraph。"),
        Document(page_content="LCEL (LangChain Expression Language) 使用 | 管道符连接组件。"
                 "支持 invoke/batch/stream 统一接口。"
                 "常用工具：RunnablePassthrough、RunnableLambda、RunnableParallel。"),
        Document(page_content="LangGraph 是基于图结构的 Agent 框架。"
                 "核心概念：State（状态）、Node（节点）、Edge（边）。"
                 "支持条件路由、循环、持久化、人机交互（HITL）。"),
        Document(page_content="RAG 检索增强生成的完整流程："
                 "1. 文档加载（TextLoader、PyPDFLoader）"
                 "2. 文本分割（RecursiveCharacterTextSplitter）"
                 "3. 向量化（Embeddings）"
                 "4. 存入向量库（FAISS、Chroma）"
                 "5. 检索相关文档 → 注入 prompt → LLM 生成回答"),
        Document(page_content="AI Agent 通过工具调用与外部世界交互。"
                 "ReAct 模式：推理→行动→观察→循环。"
                 "create_react_agent 可快速创建 Agent。"
                 "AgentExecutor 管理 Agent 的执行循环、错误处理和迭代次数。"),
        Document(page_content="多 Agent 系统的三种模式："
                 "1. Supervisor 模式：中心调度器分配任务"
                 "2. Swarm 模式：Agent 间直接交接"
                 "3. Hierarchical 模式：多层级管理"
                 "推荐使用 LangGraph 构建多 Agent 系统。"),
    ]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# ============================================================
# 会话存储
# ============================================================

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ============================================================
# 构建 RAG + Memory Chain
# ============================================================

def build_chain():
    retriever = build_knowledge_base()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是 AI 技术学习助手。基于以下参考资料回答问题。
如果资料中没有相关信息，请根据你的知识回答，并说明"以上回答不在知识库中"。

参考资料：
{context}"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 构建基础 RAG chain
    basic_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 添加消息历史
    chain_with_history = RunnableWithMessageHistory(
        basic_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_history, retriever


# ============================================================
# Demo
# ============================================================

def main():
    print("=" * 50)
    print(f"RAG + 对话记忆 智能问答 [{QWEN_MODEL}]")
    print("=" * 50)
    print()

    chain, retriever = build_chain()
    config = {"configurable": {"session_id": "rag-demo"}}

    # 模拟多轮对话
    conversations = [
        "LangChain 是什么？",
        "它有哪些核心组件？",                    # 追问 LangChain
        "LCEL 怎么用？",                        # 继续追问
        "能给我一个完整的 RAG 流程吗？",         # 换话题
        "刚才我问了哪些问题？",                  # 测试记忆
    ]

    for user_msg in conversations:
        # 检索相关文档
        retrieved = retriever.invoke(user_msg)

        response = chain.invoke({"input": user_msg}, config=config)

        print(f"用户: {user_msg}")
        print(f"  检索: {len(retrieved)} 条文档")
        print(f"  AI: {response}")
        print()

    # 查看会话历史
    history = get_session_history("rag-demo")
    print(f"会话历史: {len(history.messages)} 条消息")
    print()

    # 对比：新会话（无记忆）
    print("--- 对比：新会话（无历史记忆）---")
    new_config = {"configurable": {"session_id": "new-session"}}
    response = chain.invoke(
        {"input": "刚才我问了哪些问题？"},
        config=new_config,
    )
    print(f"用户: 刚才我问了哪些问题？")
    print(f"AI: {response}")
    print()
    print("结论: 新会话没有历史记忆，无法知道之前的问题")


if __name__ == "__main__":
    main()

    print()
    print("=" * 50)
    print("Demo 8-3 完成!")
    print()
    print("RAG + Memory 要点:")
    print("  1. 检索器获取相关文档作为上下文")
    print("  2. MessagesPlaceholder 注入对话历史")
    print("  3. RunnableWithMessageHistory 自动管理历史")
    print("  4. session_id 隔离不同会话")
    print("  5. Agent 能理解上下文追问（如'它'、'刚才'）")
    print("=" * 50)
