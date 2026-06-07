# LangChain AI Agent 学习项目

从零基础到独立构建生产级 AI Agent 应用，系统掌握 LangChain 核心概念与实践。

## 项目结构

```
learn_langchain/
├── demo/              # 可运行的示例代码
├── docs/              # 33 章完整学习文档
├── data/              # 示例数据
├── pyproject.toml     # 项目配置
├── .env.example       # 环境变量模板
└── .env               # 环境变量（需自行创建，已加入 .gitignore）
```

## 学习路线

项目共 **33 章**，分为九个部分：

| 部分 | 章节 | 内容 | 建议时长 |
|------|------|------|----------|
| 基础入门 | 第 1-4 章 | 环境搭建、LLM 基础、Prompt 模板、输出解析器 | 3-5 天 |
| 核心机制 | 第 5-8 章 | LCEL Chain、RAG、Retriever、Memory | 5-7 天 |
| Agent 构建 | 第 9-12 章 | Tools、Agent 基础、执行控制、自定义工具 | 7-10 天 |
| 进阶实践 | 第 13-16 章 | LangGraph、多 Agent、综合实战 | 7-10 天 |
| 生产化实践 | 第 17-25 章 | 回调、异步、安全、测试、LangSmith、部署 | 10-15 天 |
| 模型微调 | 第 26 章 | LoRA/QLoRA、数据准备、训练与评估 | 3-5 天 |
| 评估与优化 | 第 27-28 章 | LLM 评估体系、幻觉检测 | 3-5 天 |
| 实战案例 | 第 29-30 章 | 智能客服、知识问答系统 | 5-7 天 |
| 进阶服务化 | 第 31-33 章 | LangServe、语义缓存、用户反馈 | 3-5 天 |

总计约 **8-9 周**。

## 快速开始

### 环境要求

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv)（推荐）或 pip

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd learn_langchain

# 安装依赖（使用 uv）
uv sync

# 或使用 pip
pip install -e .
```

### 配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env
```

编辑 `.env`，填入你的 API Key。项目支持多个 LLM 提供商：

| 提供商 | 环境变量 | 申请地址 |
|--------|----------|----------|
| GLM（智谱 AI） | `GLM_API_KEY` | https://open.bigmodel.cn/ |
| DeepSeek | `DS_API_KEY` | https://platform.deepseek.com/ |
| Qwen（通义千问） | `QWEN_API_KEY` | https://dashscope.console.aliyun.com/ |
| LangSmith（可选） | `LANGCHAIN_API_KEY` | https://smith.langchain.com/ |

至少配置其中一个提供商即可运行大部分示例。

### 运行示例

```bash
# 验证环境
python demo/01_01_verify_setup.py

# LLM 基础
python demo/02_01_chat_basics.py

# RAG 检索增强
python demo/06_02_rag_pipeline.py

# ReAct Agent
python demo/10_01_react_agent.py

# LangGraph
python demo/13_01_stategraph_basics.py
```

## 技术栈

- **LangChain** - LLM 应用开发框架
- **LangGraph** - 多 Agent 工作流编排
- **FAISS** - 向量相似度检索
- **GLM（智谱 AI）** - 默认 LLM 提供商
- **DeepSeek** - 备选 LLM 提供商
- **Qwen（通义千问）** - 备选 LLM 提供商
- **Python-dotenv** - 环境变量管理

## 文档

完整学习文档位于 [docs/](docs/) 目录，包含详细的概念讲解和代码示例说明。

学习大纲详见 [docs/README.md](docs/README.md)。

## License

MIT
