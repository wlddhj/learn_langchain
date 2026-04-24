"""
第30章 Demo 3：完整知识问答系统

演示带会话管理、反馈收集的完整问答系统。
需要 QWEN_API_KEY。
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key=None, model="text-embedding-v1"):
        self.api_key = api_key or os.environ["QWEN_API_KEY"]
        self.model = model
        import dashscope
        dashscope.api_key = self.api_key

    def embed_documents(self, texts):
        from dashscope import TextEmbedding
        texts = [str(text) for text in texts]
        all_embeddings = []
        for text in texts:
            response = TextEmbedding.call(model=self.model, input=text)
            if response.status_code == 200:
                all_embeddings.append(response.output['embeddings'][0]['embedding'])
            else:
                all_embeddings.append([0.0] * 1536)
        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([str(text)])[0]


embeddings = DashScopeEmbeddings(api_key=QWEN_API_KEY)


class KnowledgeQASystem:
    """完整知识问答系统"""

    def __init__(self):
        # 构建知识库
        self.docs = [
            Document(page_content="产品价格：基础版 99元/月，专业版 299元/月，企业版 999元/月。", metadata={"source": "pricing.txt"}),
            Document(page_content="退货政策：7天内可退货，30天内可换货。需保持产品完好。", metadata={"source": "return.txt"}),
            Document(page_content="客服电话：400-123-4567，工作时间：9:00-18:00。", metadata={"source": "contact.txt"}),
            Document(page_content="发货时间：下单后24小时内发货，一般3-5天送达。", metadata={"source": "shipping.txt"}),
            Document(page_content="支付方式：支持支付宝、微信支付、银行卡。", metadata={"source": "payment.txt"}),
        ]

        self.vectorstore = FAISS.from_documents(self.docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.sessions = {}
        self.qa_records = {}
        self.feedbacks = {}

    def create_session(self, user_id: str) -> str:
        """创建会话"""
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            "user_id": user_id,
            "history": [],
            "created_at": datetime.now(),
        }
        return session_id

    def ask(self, question: str, session_id: str = None) -> dict:
        """问答"""
        # 检索
        docs = self.retriever.invoke(question)
        context = "\n\n".join([f"[{doc.metadata['source']}] {doc.page_content}" for doc in docs])

        # 生成答案
        prompt = ChatPromptTemplate.from_template("""
基于以下知识库回答问题：

{context}

问题：{question}

要求：
1. 使用知识库信息回答
2. 引用具体来源
3. 如果知识库没有相关信息，请说明
""")

        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question,
        })

        # 记录问答
        qa_id = str(uuid.uuid4())[:8]
        self.qa_records[qa_id] = {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata for doc in docs],
            "timestamp": datetime.now(),
        }

        # 更新会话历史
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                "question": question,
                "answer": answer,
            })

        return {
            "qa_id": qa_id,
            "question": question,
            "answer": answer,
            "sources": [doc.metadata["source"] for doc in docs],
        }

    def collect_feedback(self, qa_id: str, rating: int, comment: str = None) -> str:
        """收集反馈"""
        feedback_id = str(uuid.uuid4())[:8]
        self.feedbacks[feedback_id] = {
            "qa_id": qa_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now(),
        }
        return feedback_id

    def get_stats(self) -> dict:
        """统计"""
        avg_rating = 0
        if self.feedbacks:
            ratings = [f["rating"] for f in self.feedbacks.values()]
            avg_rating = sum(ratings) / len(ratings)

        return {
            "total_qa": len(self.qa_records),
            "total_feedback": len(self.feedbacks),
            "avg_rating": avg_rating,
        }


def demo_full_system():
    """完整系统演示"""
    print("=" * 60)
    print(f"Demo 30-3: 完整知识问答系统 [{QWEN_MODEL}]")
    print("=" * 60)
    print()

    qa_system = KnowledgeQASystem()

    # 创建会话
    session_id = qa_system.create_session("user_001")
    print(f"创建会话: {session_id}")
    print()

    # 多轮问答
    questions = [
        "产品价格是多少？",
        "可以退货吗？",
        "客服电话是多少？",
        "支持哪些支付方式？",
    ]

    print("问答测试：")
    print("-" * 60)

    for q in questions:
        result = qa_system.ask(q, session_id)

        print(f"问题: {q}")
        print(f"答案: {result['answer'][:100]}...")
        print(f"来源: {result['sources']}")
        print()

        # 模拟反馈
        qa_system.collect_feedback(result["qa_id"], rating=4, comment="满意")

    # 统计
    stats = qa_system.get_stats()
    print("系统统计：")
    print("-" * 60)
    print(f"问答总数: {stats['total_qa']}")
    print(f"反馈总数: {stats['total_feedback']}")
    print(f"平均评分: {stats['avg_rating']}")


if __name__ == "__main__":
    demo_full_system()

    print("=" * 60)
    print("Demo 30-3 完成!")
    print()
    print("知识问答系统完整功能：")
    print("  - 文档加载与向量嵌入")
    print("  - 语义检索")
    print("  - LLM 答案生成")
    print("  - 来源引用")
    print("  - 会话管理")
    print("  - 反馈收集")
    print("  - 统计分析")
    print()
    print("扩展建议：")
    print("  - 添加多轮对话支持")
    print("  - 实现文档动态更新")
    print("  - 添加缓存机制")
    print("  - API 服务化")
    print("=" * 60)