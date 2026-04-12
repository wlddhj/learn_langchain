"""
第6章 Demo 3：实战 —— 文件知识库问答系统

从本地文件构建知识库，支持多轮检索问答。
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

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

def build_knowledge_base():
    """构建 Python 学习指南知识库"""
    knowledge = [
        # Python 基础
        ("Python 基础语法", """\
Python 是一种解释型、面向对象的高级编程语言。它以简洁优雅的语法著称。
Python 支持多种编程范式：面向对象、函数式、过程式编程。
Python 的基本数据类型包括：int（整数）、float（浮点数）、str（字符串）、
bool（布尔值）、list（列表）、tuple（元组）、dict（字典）、set（集合）。
列表推导式是 Python 的一大特色，可以用简洁的语法创建列表：
[x**2 for x in range(10) if x % 2 == 0]。"""),

        ("Python 函数与装饰器", """\
Python 中用 def 关键字定义函数。函数是一等公民，可以作为参数传递、
作为返回值、赋值给变量。
装饰器是 Python 的高级特性，本质是一个接受函数并返回新函数的高阶函数。
常用装饰器包括 @staticmethod、@classmethod、@property。
functools.lru_cache 装饰器可以缓存函数结果，提高递归性能。
生成器使用 yield 关键字，可以惰性地生成序列，节省内存。"""),

        ("Python 异步编程", """\
Python 的 asyncio 模块提供了异步编程支持。async/await 语法让异步代码
看起来像同步代码一样清晰。
async def 定义的函数是协程，需要用 await 调用。
asyncio.gather() 可以并发执行多个协程。
asyncio.create_task() 将协程包装为 Task 对象。
aiohttp 是异步 HTTP 客户端/服务器库，适合高并发网络请求。"""),

        ("Python 错误处理", """\
Python 使用 try/except/else/finally 进行错误处理。
可以捕获特定异常类型，也可以用 Exception 捕获所有异常。
raise 语句用于主动抛出异常。
自定义异常类继承 Exception。
常见的内置异常：ValueError、TypeError、KeyError、IndexError、FileNotFoundError。
建议使用 finally 块来释放资源（如关闭文件、数据库连接）。"""),

        # Web 开发
        ("FastAPI 框架", """\
FastAPI 是一个现代、高性能的 Python Web 框架。
它基于 Starlette 和 Pydantic 构建，自动生成 OpenAPI 文档。
使用类型注解定义请求参数和响应模型，自动进行数据校验。
支持异步处理，性能媲美 Node.js 和 Go。
uvicorn 是推荐的 ASGI 服务器。
常用装饰器：@app.get、@app.post、@app.put、@app.delete。"""),

        ("Django 框架", """\
Django 是一个全功能的 Python Web 框架，遵循 MTV（Model-Template-View）架构。
内置 ORM、Admin 后台、认证系统、中间件、缓存等组件。
Django ORM 支持多种数据库：PostgreSQL、MySQL、SQLite、Oracle。
Django REST Framework (DRF) 用于构建 RESTful API。
django-celery 用于异步任务处理。"""),

        # 数据科学
        ("Pandas 数据分析", """\
Pandas 是 Python 最流行的数据分析库。核心数据结构是 Series 和 DataFrame。
DataFrame 是二维表格，支持列操作、行操作、分组聚合、数据透视。
pd.read_csv() 读取 CSV 文件，df.to_csv() 保存为 CSV。
df.describe() 生成描述性统计，df.info() 查看数据概览。
df.groupby() 分组聚合，df.merge() 合并数据框，df.pivot_table() 数据透视。
df.apply() 对行或列应用自定义函数。"""),

        ("NumPy 科学计算", """\
NumPy 是 Python 科学计算的基础库，提供高性能的多维数组 ndarray。
ndarray 支持 Broadcasting（广播）机制，可以对不同形状的数组进行运算。
np.array() 创建数组，np.zeros() 创建全零数组，np.arange() 创建等差数列。
np.dot() 矩阵乘法，np.mean() 平均值，np.std() 标准差。
NumPy 的数组操作比 Python 原生列表快 10-100 倍。"""),
    ]

    docs = []
    for title, content in knowledge:
        docs.append(Document(page_content=content, metadata={"title": title}))
    return docs


def main():
    print("=" * 50)
    print(f"Python 学习知识库问答 [{QWEN_MODEL}]")
    print("=" * 50)
    print()

    # 1. 构建知识库
    print("正在构建知识库...")
    docs = build_knowledge_base()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"文档: {len(docs)} → 切分: {len(chunks)} chunks")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("知识库构建完成!")
    print()

    # 2. RAG prompt
    prompt = ChatPromptTemplate.from_template("""你是 Python 学习助手。基于以下参考资料回答问题。
如果参考资料中没有相关信息，请说"知识库中没有相关内容，我来根据我的知识回答"。

参考资料：
{context}

问题：{question}

回答：""")

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('title', '未知')}] {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 3. 测试问答
    questions = [
        "Python 中什么是装饰器？举个例子。",
        "FastAPI 和 Django 有什么区别？",
        "如何用 Pandas 读取 CSV 文件并查看前5行？",
        "什么是 Python 的 GIL？",  # 知识库中没有
    ]

    for q in questions:
        print(f"Q: {q}")

        # 先看检索到了什么
        retrieved = retriever.invoke(q)
        print(f"  检索到 {len(retrieved)} 条相关文档:")
        for doc in retrieved:
            print(f"    - [{doc.metadata.get('title', '?')}] {doc.page_content[:50]}...")

        # RAG 回答
        answer = rag_chain.invoke(q)
        print(f"  A: {answer}")
        print()


if __name__ == "__main__":
    main()

    print("=" * 50)
    print("Demo 6-3 完成!")
    print()
    print("知识库问答系统要点:")
    print("  1. 准备结构化知识文档（标题+内容）")
    print("  2. 切分为合适大小的 chunks")
    print("  3. 向量化存入 FAISS")
    print("  4. 检索相关文档 → 注入 prompt → LLM 回答")
    print("=" * 50)
