# LangChain AI Agent 学习大纲

## 目标
从零基础到独立构建生产级 AI Agent 应用，系统掌握 LangChain 核心概念与实践。

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
| 第5章 | [05-chains.md](05-chains.md) | Chain 式调用：LCEL (LangChain Expression Language)、管道操作、SequentialChain |
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

## 第五部分：生产化实践

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第17章 | [17-callbacks.md](17-callbacks.md) | 回调系统：BaseCallbackHandler、Token追踪、日志记录、告警机制 |
| 第18章 | [18-async-programming.md](18-async-programming.md) | 异步编程深入：ainvoke/abatch/astream、并发控制、性能优化 |
| 第19章 | [19-error-handling.md](19-error-handling.md) | 异常处理与稳定性：with_retry、with_fallbacks、速率限制、超时控制 |
| 第20章 | [20-cost-optimization.md](20-cost-optimization.md) | Token 成本优化：模型选择、历史压缩、RAG优化、缓存策略 |
| 第21章 | [21-security.md](21-security.md) | 安全防护进阶：Prompt Injection、工具安全、敏感数据、权限控制 |
| 第22章 | [22-testing.md](22-testing.md) | 测试与质量保证：Mock LLM、Agent测试、RAG测试、评估方法 |
| 第23章 | [23-langsmith.md](23-langsmith.md) | LangSmith 深度使用：追踪、调试、评估、监控、成本分析 |
| 第24章 | [24-deployment.md](24-deployment.md) | 部署与生产化：API服务化、容器化、水平扩展、CI/CD |
| 第25章 | [25-web-interface.md](25-web-interface.md) | Web 界面集成：Streamlit、Gradio、流式输出、部署选项 |

## 第六部分：模型微调

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第26章 | [26-model-finetuning.md](26-model-finetuning.md) | 模型微调入门：LoRA/QLoRA原理、数据准备、训练实战、评估部署 |

## 第七部分：评估与优化（新增）

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第27章 | [27-llm-evaluation.md](27-llm-evaluation.md) | LLM 评估体系：自动指标、LLM-as-Judge、RAG评估、Agent评估、人工评估 |
| 第28章 | [28-hallucination-detection.md](28-hallucination-detection.md) | 幻觉检测与处理：幻觉类型、检测方法、预防策略、处理流程 |

## 第八部分：实战案例（新增）

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第29章 | [29-practical-case-customer-service.md](29-practical-case-customer-service.md) | 智能客服系统实战：意图识别、情绪分析、对话管理、Agent集成 |
| 第30章 | [30-practical-case-knowledge-qa.md](30-practical-case-knowledge-qa.md) | 知识问答系统实战：文档处理、检索引擎、答案生成、反馈系统 |

## 第九部分：进阶服务化（新增）

| 章节 | 文档 | 内容概要 |
|------|------|----------|
| 第31章 | [31-langserve.md](31-langserve.md) | LangServe 服务化：快速部署、路由配置、Playground、客户端使用 |
| 第32章 | [32-semantic-cache.md](32-semantic-cache.md) | 语义缓存深入：原理实现、LangChain集成、高级策略、分布式缓存 |
| 第33章 | [33-user-feedback-system.md](33-user-feedback-system.md) | 用户反馈系统：反馈收集、分析系统、优化驱动、报告生成 |

---

## 学习路线建议

```
第一阶段（基础）：第1-4章（3-5天）
    ↓
第二阶段（核心）：第5-8章（5-7天）
    ↓
第三阶段（Agent）：第9-12章（7-10天）
    ↓
第四阶段（进阶）：第13-16章（7-10天）
    ↓
第五阶段（生产化）：第17-25章（10-15天）
    ↓
第六阶段（微调）：第26章（3-5天）
    ↓
第七阶段（评估）：第27-28章（3-5天）
    ↓
第八阶段（实战）：第29-30章（5-7天）
    ↓
第九阶段（服务化）：第31-33章（3-5天）
```

总计约 **8-9 周**，建议边学边动手实践每个章节的代码示例。

## 章节依赖关系

```
第1章 → 第2章 → 第3章 → 第4章 → 第5章
         ↓
    第6章 → 第7章 → 第8章
         ↓
    第9章 → 第10章 → 第11章 → 第12章
         ↓
    第13章 → 第14章 → 第15章 → 第16章
         ↓
    第17-25章（可按需学习，建议顺序：17→18→19→20→21→22→23→24→25）
         ↓
    第26章（模型微调，独立章节）
         ↓
    第27-28章（评估与幻觉检测）
         ↓
    第29-30章（实战案例）
         ↓
    第31-33章（LangServe、缓存、反馈系统）
```
