# 第1章：开发环境搭建

## 1.1 Python 环境准备

推荐使用 Python 3.10+，确保版本兼容性。

```bash
# 检查 Python 版本
python --version

# 推荐使用 uv 管理虚拟环境（比 venv 更快）
pip install uv
```

## 1.2 创建项目与虚拟环境

```bash
# 创建项目目录
mkdir langchain-agent-learning
cd langchain-agent-learning

# 使用 uv 创建虚拟环境
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

## 1.3 安装 LangChain

LangChain 采用模块化设计，按需安装：

```bash
# 核心包（必装）
uv add langchain

# 与具体 LLM 提供商集成
uv add langchain-openai       # OpenAI / Azure OpenAI
uv add langchain-anthropic    # Claude
uv add langchain-community    # 社区集成（第三方服务）

# LangGraph（构建 Agent 的核心框架，后期使用）
uv add langgraph

# 向量数据库（RAG 章节使用）
uv add langchain-chroma              # Chroma 向量数据库（推荐）
uv add faiss-cpu                     # Facebook 的向量检索库

# 文档处理
uv add pypdf                  # PDF 解析
uv add beautifulsoup4         # HTML 解析

# 开发工具
uv add python-dotenv          # 环境变量管理
uv add jupyter                # Notebook 环境（推荐用于学习）
```

## 1.4 API Key 配置

创建 `.env` 文件管理密钥：

```bash
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

> **安全提示**：确保 `.env` 已添加到 `.gitignore`，切勿将 API Key 提交到代码仓库。

```bash
# .gitignore
.env
.venv/
__pycache__/
```

在代码中加载环境变量：

```python
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量
```

## 1.5 验证安装

```python
# verify_setup.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

response = llm.invoke("你好，请用一句话介绍你自己。")
print(response.content)
```

运行成功则环境配置完成。

## 1.6 推荐开发工具

| 工具 | 用途 |
|------|------|
| VS Code / Cursor | 代码编辑器 |
| Jupyter Notebook | 交互式学习（强烈推荐） |
| LangSmith | LangChain 可观测性平台（调试利器） |

### 配置国产模型（推荐国内开发者）

LangChain 通过 OpenAI 兼容接口支持国内主流模型，只需修改 `api_key` 和 `base_url`：

```python
# 智谱 GLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key="your-glm-api-key",
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

# 通义千问 Qwen
llm = ChatOpenAI(
    model="qwen-plus",
    api_key="your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com",
)
```

在 `.env` 中配置：

```bash
# 选择其中一个即可
GLM_API_KEY=your-glm-api-key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4-flash

QWEN_API_KEY=your-qwen-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
```

> **提示**：代码中只需改 `model`、`api_key`、`base_url` 三个参数，其余 LangChain 用法完全相同。

### LangSmith 配置（可选但推荐）

LangSmith 用于追踪 LangChain 应用的每一步执行过程，对调试和学习非常有帮助。

```bash
# .env 中添加
LANGCHAIN_API_KEY=lsv2_xxxxxxxx
LANGCHAIN_TRACING_V2=true
```

## 1.7 本章小结

- 使用 `uv` 管理依赖和虚拟环境
- LangChain 按模块安装：`langchain` + `langchain-openai` 等集成包
- API Key 通过 `.env` 管理，代码中用 `python-dotenv` 加载
- LangSmith 是调试和学习的最佳伴侣
