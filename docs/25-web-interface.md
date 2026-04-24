# 第25章：Web 界面集成

## 25.1 为什么需要 Web 界面

| 场景 | 价值 |
|------|------|
| **产品演示** | 直观展示功能 |
| **用户交互** | 提升用户体验 |
| **内部工具** | 快速部署工具 |
| **原型开发** | 快速迭代验证 |

## 25.2 界面方案对比

| 方案 | 特点 | 适用场景 |
|------|------|---------|
| **Streamlit** | 简单快速，适合数据应用 | 内部工具、演示 |
| **Gradio** | 专为 ML/AI 设计 | 模型演示、交互测试 |
| **FastAPI + React** | 灵活强大 | 生产级产品 |
| **Flask + 前端** | 传统方案 | 通用 Web 应用 |

## 25.3 Streamlit 快速开发

### 基础聊天界面

```python
# streamlit_app.py
import streamlit as st
from langchain_openai import ChatOpenAI

st.title("🤖 AI 聯天助手")

# 初始化模型
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model="gpt-4o-mini")

# 初始化历史消息
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("输入消息"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 回复
    with st.chat_message("assistant"):
        response = st.session_state.llm.invoke(prompt)
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

# 运行: streamlit run streamlit_app.py
```

### 流式输出

```python
import streamlit as st
from langchain_openai import ChatOpenAI

st.title("🤖 AI 聯天助手（流式）")

llm = ChatOpenAI(model="gpt-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入消息"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 流式输出
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

### Agent 界面

```python
import streamlit as st
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"搜索结果: {query}"

@tool
def calculate(expression: str) -> str:
    """计算工具"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

st.title("🤖 Agent 助手")

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search, calculate],
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入任务"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # 运行 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = agent.invoke({"messages": [("user", prompt)]})

            # 显示工具调用过程
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        st.info(f"🔧 调用工具: {tc['name']}")

            # 显示最终回答
            final_msg = result["messages"][-1]
            st.markdown(final_msg.content)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": final_msg.content})
```

### RAG 界面

```python
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.title("📚 知识库问答")

# 上传文档
uploaded_files = st.file_uploader("上传文档", type=["txt", "pdf", "md"], accept_multiple_files=True)

if uploaded_files:
    # 处理文档
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = []
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        docs.append(Document(page_content=content, metadata={"source": file.name}))

    # 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 创建向量库
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)

    st.success(f"已处理 {len(chunks)} 个文档片段")

    # RAG Chain
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
    基于以下内容回答问题：
    {context}

    问题：{question}
    """)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

    # 问答
    question = st.text_input("输入问题")
    if question:
        with st.spinner("检索中..."):
            answer = chain.invoke(question)
            st.markdown(f"**回答:** {answer}")
```

### 配置面板

```python
import streamlit as st

st.sidebar.title("⚙️ 配置")

# 模型选择
model = st.sidebar.selectbox(
    "模型",
    ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
)

# Temperature
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
)

# Max Tokens
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=100,
    max_value=4000,
    value=1000,
    step=100,
)

# 使用配置创建模型
llm = ChatOpenAI(
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
)

# 清除历史按钮
if st.sidebar.button("清除对话历史"):
    st.session_state.messages = []
    st.rerun()
```

## 25.4 Gradio 界面

### 基础聊天

```python
# gradio_app.py
import gradio as gr
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def chat(message, history):
    """聊天函数"""
    # Gradio history 格式转换为 messages
    messages = []
    for user_msg, assistant_msg in history:
        messages.append(("user", user_msg))
        messages.append(("assistant", assistant_msg))
    messages.append(("user", message))

    response = llm.invoke(messages)
    return response.content

# 创建界面
demo = gr.ChatInterface(
    chat,
    title="AI 聯天助手",
    description="基于 LangChain 的聯天应用",
)

# 运行
demo.launch()
```

### 流式输出

```python
import gradio as gr
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def stream_chat(message, history):
    """流式聊天"""
    messages = []
    for user_msg, assistant_msg in history:
        messages.append(("user", user_msg))
        messages.append(("assistant", assistant_msg))
    messages.append(("user", message))

    partial_message = ""
    for chunk in llm.stream(messages):
        if chunk.content:
            partial_message += chunk.content
            yield partial_message

demo = gr.ChatInterface(
    stream_chat,
    title="AI 聯天助手（流式）",
)

demo.launch()
```

### 多功能界面

```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

# 翻译功能
def translate(text, target_language):
    prompt = ChatPromptTemplate.from_template(
        "将以下文本翻译成{language}: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"language": target_language, "text": text})

# 摘要功能
def summarize(text):
    prompt = ChatPromptTemplate.from_template(
        "用一句话总结以下内容: {text}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})

# 创建界面
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI 工具集")

    with gr.Tab("聊天"):
        chat_interface = gr.ChatInterface(
            lambda m, h: llm.invoke(m).content,
        )

    with gr.Tab("翻译"):
        text_input = gr.Textbox(label="输入文本", lines=5)
        language = gr.Dropdown(["英文", "日文", "韩文", "法文"], label="目标语言")
        translate_output = gr.Textbox(label="翻译结果", lines=5)
        translate_btn = gr.Button("翻译")
        translate_btn.click(translate, [text_input, language], translate_output)

    with gr.Tab("摘要"):
        summary_input = gr.Textbox(label="输入文本", lines=10)
        summary_output = gr.Textbox(label="摘要", lines=3)
        summary_btn = gr.Button("生成摘要")
        summary_btn.click(summarize, [summary_input], summary_output)

demo.launch()
```

### Agent 界面

```python
import gradio as gr
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    return f"搜索: {query}"

@tool
def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "错误"

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search, calculate],
)

def agent_chat(message, history):
    result = agent.invoke({"messages": [("user", message)]})

    # 返回执行过程
    process = ""
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                process += f"🔧 调用工具: {tc['name']}\n"

    return process + "\n" + result["messages"][-1].content

demo = gr.ChatInterface(
    agent_chat,
    title="🤖 Agent 助手",
    description="可以搜索和计算",
)

demo.launch()
```

## 25.5 部署选项

### Streamlit Cloud

```bash
# 1. 创建 GitHub 仓库
# 2. 推送代码
# 3. 在 https://share.streamlit.io 部署
# 4. 配置环境变量（Secrets）
```

### Gradio Hugging Face Spaces

```bash
# 1. 创建 Hugging Face 账号
# 2. 创建 Space (https://huggingface.co/new-space)
# 3. 选择 Gradio SDK
# 4. 上传代码
# 5. 配置环境变量（Settings > Secrets）
```

### 本地部署

```bash
# Streamlit
streamlit run app.py --server.port 8080

# Gradio
python app.py  # demo.launch(server_name="0.0.0.0", server_port=8080)
```

## 25.6 界面设计最佳实践

### 用户体验

```python
# 1. 显示加载状态
with st.spinner("处理中..."):
    result = llm.invoke(prompt)

# 2. 显示进度
progress_bar = st.progress(0)
for i, chunk in enumerate(llm.stream(prompt)):
    progress_bar.progress(i / 10)

# 3. 错误提示
try:
    result = llm.invoke(prompt)
except Exception as e:
    st.error(f"发生错误: {e}")

# 4. 响应式设计
st.set_page_config(
    page_title="AI 助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

### 安全配置

```python
# Streamlit secrets
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-xxx"

# 代码中使用
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Gradio 认证
demo.launch(
    auth=("username", "password"),
    # 或自定义认证函数
    auth=lambda u, p: check_credentials(u, p),
)
```

### 性能优化

```python
# 1. 缓存模型初始化
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")

llm = get_llm()  # 只初始化一次

# 2. 缓存不变数据
@st.cache_data
def load_documents():
    return load_and_process_docs()

docs = load_documents()

# 3. 异步处理
import asyncio

async def async_process():
    result = await llm.ainvoke(prompt)
    return result

result = asyncio.run(async_process())
```

## 25.7 完整示例

### Streamlit 完整应用

```python
# app.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 配置
st.set_page_config(page_title="AI 助手", page_icon="🤖", layout="wide")

# 侧边栏配置
with st.sidebar:
    st.title("⚙️ 配置")
    model = st.selectbox("模型", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0)
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000)

    if st.button("清除历史"):
        st.session_state.messages = []
        st.rerun()

# 缓存模型
@st.cache_resource
def get_llm(model, temperature, max_tokens):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

llm = get_llm(model, temperature, max_tokens)

# 主界面
st.title("🤖 AI 聯天助手")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("输入消息"):
    # 用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 回复（流式）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in llm.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

## 25.8 界面方案选择

| 需求 | 推荐 |
|------|------|
| **快速演示** | Streamlit |
| **模型展示** | Gradio |
| **生产产品** | FastAPI + React |
| **内部工具** | Streamlit/Gradio |
| **复杂交互** | FastAPI + 前端 |

## 25.9 本章小结

- Streamlit：快速开发，适合数据应用和内部工具
- Gradio：专为 ML/AI 设计，适合模型演示
- 流式输出提升用户体验
- 配置面板让用户调整模型参数
- 缓存优化避免重复初始化
- 部署：Streamlit Cloud、Hugging Face Spaces、本地
- 安全：使用 secrets 管理 API Key，添加认证
- 快速演示用 Streamlit/Gradio，生产产品用 FastAPI + 前端