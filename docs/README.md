# LangChain AI Agent 学习大纲

## 目标
从零基础到独立构建 AI Agent 应用，系统掌握 LangChain 核心概念与实践。

## 前置要求
- Python 基础（函数、类、异步编程）
- 了解基本的 API 调用概念
- 了解 LLM 的基本概念（如 ChatGPT）

---

## 第一部分：基础入门

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第1章 | [01-environment-setup.md](01-environment-setup.md) | 开发环境搭建：Python、虚拟环境、LangChain 安装、API Key 配置 |
| 第2章 | [02-llm-basics.md](02-llm-basics.md) | LLM 基础：ChatModel vs LLM、消息类型、调用方式、流式输出 |
| 第3章 | [03-prompt-templates.md](03-prompt-templates.md) | Prompt 模板：ChatPromptTemplate、变量注入、Few-shot 示例 |
| 第4章 | [04-output-parsers.md](04-output-parsers.md) | 输出解析器：StrOutputParser、JsonOutputParser、PydanticOutputParser |

## 第二部分：核心机制

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第5章 | [05-chains.md](05-chains.md) | Chain 链式调用：LCEL (LangChain Expression Language)、管道操作、SequentialChain |
| 第6章 | [06-rag-basics.md](06-rag-basics.md) | RAG 检索增强生成：Document Loaders、Text Splitters、Embeddings、Vector Stores |
| 第7章 | [07-retrievers.md](07-retrievers.md) | Retriever 检索器：相似度检索、MMR、多路召回、Re-ranking |
| 第8章 | [08-memory.md](08-memory.md) | Memory 记忆管理：ConversationBufferMemory、摘要记忆、窗口记忆 |

## 第三部分：Agent 构建

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第9章 | [09-tools.md](09-tools.md) | Tools 工具定义：@tool 装饰器、StructuredTool、工具参数 schema |
| 第10章 | [10-agent-basics.md](10-agent-basics.md) | Agent 基础：Agent 类型、ReAct 模式、create_react_agent |
| 第11章 | [11-agent-executor.md](11-agent-executor.md) | Agent 执行控制与调试：执行循环、错误处理、流式追踪、回调机制 |
| 第12章 | [12-custom-tools.md](12-custom-tools.md) | 自定义工具开发：API 调用工具、数据库查询工具、文件操作工具 |

## 第四部分：进阶实践

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第13章 | [13-langgraph-basics.md](13-langgraph-basics.md) | LangGraph 基础：StateGraph、节点、边、条件路由 |
| 第14章 | [14-langgraph-advanced.md](14-langgraph-advanced.md) | LangGraph 进阶：多 Agent 协作、人机交互 (HITL)、持久化状态 |
| 第15章 | [15-multi-agent.md](15-multi-agent.md) | 多 Agent 系统：Supervisor 模式、Swarm 模式、Agent 间通信 |
| 第16章 | [16-practice-project.md](16-practice-project.md) | 综合实战：构建一个完整的 AI Agent 应用 |

---

## 学习路线建议

```
第1-2章（1-2天）→ 第3-4章（2-3天）→ 第5章（2天）
    ↓
第6-8章（3-5天）→ 第9-12章（5-7天）
    ↓
第13-15章（5-7天）→ 第16章（3-5天）
```

总计约 **3-4 周**，建议边学边动手实践每个章节的代码示例。
