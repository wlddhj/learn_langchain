# 第23章：LangSmith 深度使用

## 23.1 LangSmith 是什么

LangSmith 是 LangChain 官方提供的**可观测性平台**，专为 LLM 应用设计：

| 功能 | 说明 |
|------|------|
| **追踪** | 记录每次 LLM 调用的完整信息 |
| **调试** | 查看输入输出、中间步骤 |
| **评估** | 对比不同版本、模型的效果 |
| **监控** | Token 用量、响应时间、成本 |
| **测试** | 创建测试集，自动化评估 |

## 23.2 配置 LangSmith

### 基础配置

```python
import os

# 方式1：环境变量（推荐）
os.environ["LANGCHAIN_API_KEY"] = "lsv2_xxxxxxxx"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-project"  # 项目名称

# 之后所有 LangChain 调用自动追踪
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("你好")  # 自动记录到 LangSmith
```

### 方式2：代码配置

```python
from langchain_core.tracers.langchain import LangChainTracer

tracer = LangChainTracer(
    project_name="my-project",
)

# 调用时传入
response = llm.invoke("你好", config={"callbacks": [tracer]})
```

## 23.3 追踪功能

### 自动追踪内容

每次调用自动记录：

```
Trace 结构:
├── Run (Chain/Agent)
│   ├── inputs: 输入数据
│   ├── outputs: 输出数据
│   ├── start_time: 开始时间
│   ├── end_time: 结束时间
│   ├── Child Runs:
│   │   ├── LLM Call
│   │   │   ├── prompts: 完整 prompt
│   │   │   ├── output: LLM 输出
│   │   │   ├── token_usage: token 统计
│   │   │   ├── model_name: 模型名称
│   │   │   └── latency: 响应时间
│   │   ├── Tool Call
│   │   │   ├── tool_name: 工具名称
│   │   │   ├── tool_input: 工具输入
│   │   │   ├── tool_output: 工具输出
│   │   ├── Retriever Call
│   │   │   ├── query: 查询内容
│   │   │   ├── documents: 检索结果
```

### 查看 Trace

```python
# 执行后，在 LangSmith UI 查看：
# https://smith.langchain.com/o/<org>/projects/p/<project>

# 每个 trace 显示：
# - 完整的调用链
# - 每个组件的输入输出
# - Token 用量
# - 响应时间
# - 错误信息（如有）
```

### 追踪 Agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)

# 执行会自动追踪：
# - 用户输入
# - LLM 的思考过程
# - 工具调用的名称、参数、结果
# - 最终回答

result = agent.invoke({"messages": [("user", "北京天气")]})

# 在 LangSmith 中可以看到：
# 1. Agent 开始
# 2. LLM 调用（决定调用 weather 工具）
# 3. Tool 调用（weather，参数 {"city": "北京"}）
# 4. Tool 结果
# 5. LLM 调用（生成最终回答）
```

## 23.4 调试功能

### 查看完整 Prompt

```python
# LangSmith 会记录发送给 LLM 的完整 prompt

# 示例：查看 RAG 的完整 context
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

result = chain.invoke("什么是 RAG？")

# 在 LangSmith 中点击 LLM Run，可以看到：
# System: 你是一个助手...
# Human: 基于以下内容回答问题：
# [检索到的文档内容]
# 问题：什么是 RAG？
```

### 查看中间结果

```python
# Chain 中的每个步骤都会被追踪

chain = prompt | llm | parser

result = chain.invoke({"topic": "AI"})

# LangSmith 显示：
# Run 1: prompt - 输入 {"topic": "AI"}, 输出 ChatPromptValue
# Run 2: llm - 输入 messages, 输出 AIMessage
# Run 3: parser - 输入 AIMessage, 输出 string
```

### 流式追踪

```python
# 流式输出也会被完整追踪

for chunk in llm.stream("写一首诗"):
    print(chunk.content, end="")

# LangSmith 记录：
# - 所有 token chunks
# - 总 token 数
# - 流式时间
```

## 23.5 评估功能

### 创建测试数据集

```python
from langsmith import Client

client = Client()

# 创建数据集
dataset = client.create_dataset(
    dataset_name="qa_test_set",
    description="问答测试集",
)

# 添加测试用例
client.create_example(
    inputs={"question": "什么是 RAG？"},
    outputs={"expected_answer": "RAG 是检索增强生成"},
    dataset_id=dataset.id,
)

client.create_example(
    inputs={"question": "什么是 LangChain？"},
    outputs={"expected_answer": "LangChain 是 LLM 应用框架"},
    dataset_id=dataset.id,
)
```

### 运行评估

```python
from langsmith.evaluation import evaluate

def target_function(inputs: dict) -> dict:
    """被评估的目标函数"""
    result = chain.invoke(inputs["question"])
    return {"answer": result}

# 定义评估器
def correctness_evaluator(run, example) -> dict:
    """评估正确性"""
    prediction = run.outputs["answer"]
    expected = example.outputs["expected_answer"]

    # 检查关键词
    score = any(kw in prediction for kw in expected.split())
    return {"score": float(score), "value": "correct" if score else "incorrect"}

# 运行评估
results = evaluate(
    target_function,
    data="qa_test_set",  # 数据集名称
    evaluators=[correctness_evaluator],
)

# 查看结果
for result in results:
    print(f"Question: {result.example.inputs['question']}")
    print(f"Answer: {result.run.outputs['answer']}")
    print(f"Score: {result.feedback.score}")
```

### 使用内置评估器

```python
from langsmith.evaluation import LangChainStringEvaluator

# 使用 LLM 评估回答质量
llm_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "helpfulness": "回答是否有助于解决问题？",
            "correctness": "回答是否事实正确？",
            "conciseness": "回答是否简洁明了？",
        },
    },
)

results = evaluate(
    target_function,
    data="qa_test_set",
    evaluators=[llm_evaluator],
)
```

### 对比不同版本

```python
# 评估两个不同的 prompt 版本

def version_a(inputs):
    chain_a = prompt_a | llm | parser
    return {"answer": chain_a.invoke(inputs)}

def version_b(inputs):
    chain_b = prompt_b | llm | parser
    return {"answer": chain_b.invoke(inputs)}

# 分别评估
results_a = evaluate(version_a, data="test_set", evaluators=[evaluator])
results_b = evaluate(version_b, data="test_set", evaluators=[evaluator])

# 在 LangSmith UI 中对比两个实验
# https://smith.langchain.com/o/<org>/projects/p/<project>/compare
```

## 23.6 监控功能

### Token 用量统计

```python
# LangSmith 自动统计每次调用的 token 用量

# 在 UI 中可以查看：
# - 总 token 数
# - 输入 token 数
# - 输出 token 数
# - 按模型分类
# - 按时间分类
```

### 成本分析

```python
# LangSmith 可以估算成本（需配置价格）

# 查看成本：
# - 按项目
# - 按时间段
# - 按模型
# - 按用户
```

### 响应时间分析

```python
# LangSmith 记录每次调用的响应时间

# 分析维度：
# - 平均响应时间
# - 最大响应时间
# - P50/P90/P99 响应时间
# - 响应时间分布图
```

### 错误追踪

```python
# LangSmith 自动记录错误

# 查看错误：
# - 错误类型分布
# - 错误频率
# - 错误详情
# - 错误关联的 trace
```

## 23.7 高级功能

### 标注和注释

```python
# 在 LangSmith UI 中可以对 trace 进行标注

# 标注类型：
# - 评分（1-5星）
# - 标签（如 "good", "bad", "needs_review"）
# - 注释（文字描述）
# - 反馈（用户反馈）
```

### 分支和版本

```python
# LangSmith 支持实验版本管理

# 创建新实验
experiment = client.create_experiment(
    dataset_id=dataset.id,
    experiment_name="prompt_v2",
)

# 对比不同实验结果
```

### 导出和分析

```python
# 导出 trace 数据进行分析

traces = client.list_runs(project_name="my-project")

for trace in traces:
    print(f"ID: {trace.id}")
    print(f"Inputs: {trace.inputs}")
    print(f"Outputs: {trace.outputs}")
    print(f"Tokens: {trace.total_tokens}")
    print(f"Duration: {trace.total_time_ms}ms")
```

## 23.8 最佳实践

### 项目组织

```python
# 按应用/环境分离项目

os.environ["LANGCHAIN_PROJECT"] = "my-app-dev"     # 开发环境
os.environ["LANGCHAIN_PROJECT"] = "my-app-prod"    # 生产环境
os.environ["LANGCHAIN_PROJECT"] = "my-app-test"    # 测试环境
```

### 关键 Run 标注

```python
# 对重要 Run 添加标签

from langchain_core.tracers.langchain import LangChainTracer

tracer = LangChainTracer(project_name="my-project")

# 添加标签
response = llm.invoke(
    "重要查询",
    config={
        "callbacks": [tracer],
        "tags": ["important", "production"],
        "metadata": {"user_id": "user-123", "session_id": "session-456"},
    },
)

# 在 UI 中可以按标签筛选
```

### 定期评估

```python
# 定期运行评估，监控质量

import schedule

def run_daily_evaluation():
    results = evaluate(target_function, data="daily_test_set", evaluators=[evaluator])
    # 发送报告
    send_report(results)

schedule.every().day.at("09:00").do(run_daily_evaluation)
```

### 与回调结合

```python
# LangSmith + 自定义回调

class CombinedCallback(BaseCallbackHandler):
    """LangSmith + 自定义回调"""

    def on_llm_end(self, response, **kwargs):
        # 自定义处理
        log_to_internal_system(response)

# LangSmith 通过环境变量自动启用
# 自定义回调处理额外逻辑
response = llm.invoke("你好", config={"callbacks": [CombinedCallback()]})
```

## 23.9 LangSmith vs 自定义回调

| 维度 | LangSmith | 自定义回调 |
|------|----------|-----------|
| 配置 | 环境变量即可 | 需编写代码 |
| 数据存储 | 云端自动 | 自己管理 |
| 可视化 | 完整 UI | 无 |
| 评估功能 | 内置 | 手动实现 |
| 成本分析 | 自动估算 | 手动计算 |
| 定制性 | 中等 | 高 |
| 适用场景 | 开发调试、生产监控 | 定制需求、实时告警 |

**推荐**：LangSmith + 自定义回调组合使用

## 23.10 LangSmith 配置清单

| 配置项 | 说明 |
|--------|------|
| LANGCHAIN_API_KEY | API Key |
| LANGCHAIN_TRACING_V2 | 启用追踪 |
| LANGCHAIN_PROJECT | 项目名称 |
| tags | Run 标签 |
| metadata | Run 元数据 |

## 23.11 本章小结

- LangSmith 是 LangChain 官方的可观测性平台
- 配置：设置 `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true`
- 自动追踪：LLM调用、工具调用、检索、完整 Chain
- 调试：查看完整 Prompt、中间结果、错误详情
- 评估：创建测试集、运行评估、对比版本
- 监控：Token 用量、成本分析、响应时间、错误追踪
- 高级功能：标注、版本管理、数据导出
- 最佳实践：项目分离、标签标注、定期评估
- LangSmith + 自定义回调组合使用效果最好