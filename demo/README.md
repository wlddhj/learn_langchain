# Demo 示例程序

LangChain AI Agent 学习教程的配套示例代码，每个文件可独立运行。

## 快速开始

### 1. 配置 API Key

编辑项目根目录的 `.env` 文件：

```bash
# GLM（第1-3、5-8章 demo 使用）
GLM_API_KEY=your-glm-api-key-here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=glm-4-flash

# DeepSeek（第4章 demo 使用）
DS_API_KEY=your-deepseek-api-key-here
DS_BASE_URL=https://api.deepseek.com
DS_MODEL=deepseek-chat

# Qwen（第4、6-8章 demo 使用）
QWEN_API_KEY=your-qwen-api-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
```

### 2. 安装依赖

```bash
pip install langchain langchain-openai langchain-community langgraph
pip install python-dotenv pydantic faiss-cpu dashscope
```

### 3. 运行

```bash
python demo/01_01_verify_setup.py
```

## 文件索引

### 第1章：环境搭建

| 文件 | 说明 | 需要API |
|------|------|---------|
| `01_01_verify_setup.py` | 环境验证：Python版本、包安装、API Key、快速LLM调用测试 | GLM |
| `01_02_env_config.py` | .env 配置管理：创建模板、加载环境变量、安全提醒 | 否 |
| `01_03_dependency_check.py` | 依赖包检查：按章节检查所有依赖，自动生成安装命令 | 否 |

### 第2章：LLM 基础

| 文件 | 说明 | 需要API |
|------|------|---------|
| `02_01_chat_basics.py` | ChatModel 基础：消息类型、元组语法、多轮对话、temperature、token追踪 | GLM |
| `02_02_invoke_modes.py` | 调用方式对比：invoke/batch/stream/ainvoke 四种方式 | GLM |
| `02_03_structured_output.py` | 结构化输出：Pydantic模型、嵌套结构、情感分析实战 | GLM |

### 第3章：Prompt 模板

| 文件 | 说明 | 需要API |
|------|------|---------|
| `03_01_prompt_templates.py` | PromptTemplate 纯文本模板、ChatPromptTemplate 对话模板 | GLM |
| `03_02_fewshot_and_history.py` | MessagesPlaceholder 对话历史、Few-Shot 少样本提示、模板组合 | GLM |
| `03_03_practical_code_reviewer.py` | 实战：代码审查助手（Python/JS代码审查 + 代码解释） | GLM |

### 第4章：输出解析器

| 文件 | 说明 | 使用模型 |
|------|------|----------|
| `04_01_output_parsers.py` | StrOutputParser、CommaSeparatedListOutputParser、JsonOutputParser | Qwen |
| `04_02_pydantic_and_custom.py` | PydanticOutputParser、with_structured_output、自定义 BooleanOutputParser | Qwen |
| `04_03_practical_data_extractor.py` | 实战：人物/事件/文章多格式信息提取器 | Qwen |

### 第5章：Chain 链式调用

| 文件 | 说明 | 需要API |
|------|------|---------|
| `05_01_lcel_basics.py` | LCEL 管道 `prompt \| llm \| parser`、invoke/batch/stream/async | GLM |
| `05_02_parallel_and_passthrough.py` | RunnablePassthrough 透传、RunnableLambda 自定义函数、RunnableParallel 并行 | GLM |
| `05_03_branching_and_fallback.py` | 条件路由（LLM分类+路由）、Fallback 回退、Retry 重试 | GLM |
| `05_04_practical_translator.py` | 实战：多语言翻译器（并行翻译 + 质量评价 + 反向翻译验证） | GLM |

### 第6章：RAG 检索增强生成

| 文件 | 说明 | 需要API |
|------|------|---------|
| `06_01_document_and_splitter.py` | Document 对象、TextLoader 加载、RecursiveCharacterTextSplitter 分割 | 否 |
| `06_02_rag_pipeline.py` | 完整 RAG 管道：向量化 → FAISS → 检索 → LCEL chain → 问答 | Qwen |
| `06_03_practical_qa.py` | 实战：Python 学习知识库问答系统（8篇知识文档） | Qwen |

### 第7章：Retriever 检索器进阶

| 文件 | 说明 | 需要API |
|------|------|---------|
| `07_01_retrieval_strategies.py` | Similarity vs MMR vs 相似度阈值过滤三种策略对比 | Qwen |
| `07_02_multi_and_ensemble.py` | MultiQueryRetriever 多角度查询 + Ensemble 混合检索 | Qwen |
| `07_03_retrieval_comparison.py` | 实战：4种检索策略在同一数据集上的效果对比 | Qwen |

### 第8章：Memory 记忆管理

| 文件 | 说明 | 需要API |
|------|------|---------|
| `08_01_conversation_memory.py` | RunnableWithMessageHistory 多轮对话记忆、会话隔离、角色设定 | Qwen |
| `08_02_memory_strategies.py` | 滑动窗口记忆、摘要记忆、Token 预算控制 | Qwen |
| `08_03_rag_with_memory.py` | 实战：RAG + 对话记忆，支持上下文追问的知识库问答 | Qwen |

### 第16章：综合实战项目

| 文件 | 说明 | 需要API |
|------|------|---------|
| `16_01_practice_project.py` | 智能研究助手架构：问题分类、自定义工具、LCEL管道、LangGraph工作流 | Qwen |

### 第17章：Callbacks 回调系统

| 文件 | 说明 | 需要API |
|------|------|---------|
| `17_01_callback_basics.py` | BaseCallbackHandler 回调、Token追踪、流式输出回调 | Qwen |
| `17_02_token_tracking.py` | TokenTracker 用量追踪、TimingTracker 时间追踪、成本计算 | Qwen |

### 第18章：异步编程

| 文件 | 说明 | 需要API |
|------|------|---------|
| `18_01_async_basics.py` | ainvoke/abatch/astream 异步调用对比、asyncio.gather 并发 | Qwen |
| `18_02_async_concurrency.py` | Semaphore 并发控制、异步Chain、生产级并发模式 | Qwen |

### 第19章：错误处理

| 文件 | 说明 | 需要API |
|------|------|---------|
| `19_01_error_handling_basics.py` | with_retry 重试策略、with_fallbacks 模型回退 | Qwen |
| `19_02_timeout_rate_limit.py` | asyncio.wait_for 超时控制、SimpleRateLimiter 速率限制 | Qwen |

### 第20章：成本优化

| 文件 | 说明 | 需要API |
|------|------|---------|
| `20_01_cost_basics.py` | 模型定价对比、Prompt优化示例、max_tokens控制 | Qwen |

### 第21章：安全防护

| 文件 | 说明 | 需要API |
|------|------|---------|
| `21_01_prompt_injection.py` | Prompt Injection 检测、可疑关键词、安全Prompt设计 | Qwen |

### 第22章：测试

| 文件 | 说明 | 需要API |
|------|------|---------|
| `22_01_mock_llm.py` | MockLLM 测试、Chain 单元测试、预设响应测试 | 无 |

### 第23章：LangSmith 追踪

| 文件 | 说明 | 需要API |
|------|------|---------|
| `23_01_langsmith_basics.py` | LangSmith 配置、追踪功能、代码示例 | 无 |

### 第24章：FastAPI 服务化

| 文件 | 说明 | 需要API |
|------|------|---------|
| `24_01_fastapi_basics.py` | FastAPI 服务结构、Pydantic模型、流式API | 无 |

### 第25章：Streamlit Web界面

| 文件 | 说明 | 需要API |
|------|------|---------|
| `25_01_streamlit_chat.py` | Streamlit 聊天界面、流式输出、会话状态管理 | Qwen |

### 第26章：模型微调

| 文件 | 说明 | 需要API |
|------|------|---------|
| `26_01_model_finetuning.py` | LoRA原理、配置示例、数据格式、数据验证 | 无 |

### 第27章：LLM 评估

| 文件 | 说明 | 需要API |
|------|------|---------|
| `27_01_llm_evaluation.py` | Token评估、响应时间、关键词检查、LLM-as-Judge概念 | Qwen |

### 第28章：幻觉检测与处理（新增）

| 文件 | 说明 | 需要API |
|------|------|---------|
| `28_01_hallucination_types.py` | 幻觉类型识别、幻觉原因分析、安全 Prompt 设计 | 无 |
| `28_02_detection_methods.py` | 自我检测法、事实核对法、RAG 源引用检测、置信度评估 | Qwen |
| `28_03_prevention_strategies.py` | 不安全 vs 安全 Prompt、分层验证、强约束 RAG、温度对比 | Qwen |

### 第29章：智能客服系统实战（新增）

| 文件 | 说明 | 需要API |
|------|------|---------|
| `29_01_intent_classifier.py` | 意图分类（6种意图）、意图路由、子意图提取 | Qwen |
| `29_02_sentiment_analyzer.py` | 情绪分析（4种情绪）、响应策略、转人工决策 | Qwen |
| `29_03_customer_service.py` | 完整客服系统：意图识别 + 情绪分析 + 响应生成 + 会话管理 | Qwen |

### 第30章：知识问答系统实战（新增）

| 文件 | 说明 | 需要API |
|------|------|---------|
| `30_01_document_processing.py` | Document 对象、文本切分策略、Chunk 大小选择指南 | 无 |
| `30_02_retriever_engine.py` | 向量检索、RAG 答案生成、带来源引用的答案 | Qwen |
| `30_03_complete_qa_system.py` | 完整系统：问答 + 会话管理 + 反馈收集 + 统计分析 | Qwen |

### 第31章：LangServe 服务化（新增）

| 文件 | 说明 | 需要API |
|------|------|---------|
| `31_01_basic_langserve.py` | LangServe 介绍、基础服务代码、API 端点、客户端使用 | 无 |

### 第32章：语义缓存深入（新增）

| 文件 | 说明 | 需要API |
|------|------|---------|
| `32_01_semantic_cache_basic.py` | 传统缓存 vs 语义缓存对比、工作流程、阈值参数、实现示意 | 无 |
| `32_02_semantic_cache_practice.py` | 带监控的语义缓存实现、命中率统计、缓存效果测试 | Qwen |

### 第33章：用户反馈系统（新增）

| 文件 | 说明 | 需要API |
|------|------|---------|
| `33_01_feedback_collection.py` | 反馈类型、收集流程、API 设计示例 | 无 |
| `33_02_feedback_analysis.py` | 评分统计、正确性分析、问题识别、改进循环示意 | 无 |
| `33_03_feedback_integration.py` | 反馈系统与问答集成、完整演示 | Qwen |

## 章节与模型对应关系

| 模型 | 配置变量 | 使用章节 |
|------|----------|----------|
| 智谱 GLM | `GLM_*` | 第1-3章、第5章 |
| Qwen 通义千问 | `QWEN_*` | 第4章、第6-8章 |
| DeepSeek | `DS_*` | 第4章（可选） |

> 所有模型均通过 OpenAI 兼容接口调用，只需配置 `api_key` 和 `base_url` 即可切换。
