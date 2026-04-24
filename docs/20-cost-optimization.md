# 第20章：Token 成本优化

## 20.1 为什么需要成本优化

LLM 调用按 token 计费，成本可能快速累积：

| 模型 | 输入价格 | 输出价格 | 1M tokens 成本 |
|------|---------|---------|---------------|
| gpt-4o | $2.50 | $10.00 | 较高 |
| gpt-4o-mini | $0.15 | $0.60 | 较低 |
| claude-sonnet-4 | $3.00 | $15.00 | 较高 |
| glm-4-flash | ¥0.1 | ¥0.1 | 很低 |

**成本失控的常见原因**：
- 大量历史消息累积
- 过长的文档作为 context
- 不必要的多次 LLM 调用
- Agent 循环次数过多
- 使用昂贵模型处理简单任务

## 20.2 Token 计算原理

### Token vs 字符

```
英文: 1 token ≈ 4 字符 ≈ 0.75 单词
中文: 1 token ≈ 1-2 字符（取决于分词）
代码: 1 token ≈ 3-5 字符
```

### Token 计算示例

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

response = llm.invoke("你好")

# 获取 token 用量
usage = response.response_metadata.get('token_usage', {})
print(f"输入 tokens: {usage.get('prompt_tokens')}")
print(f"输出 tokens: {usage.get('completion_tokens')}")
print(f"总 tokens: {usage.get('total_tokens')}")
```

### 预估 Token 数量

```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """预估文本的 token 数量"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 示例
text = "这是一段中文文本，用来测试 token 计算。"
print(f"预估 tokens: {estimate_tokens(text)}")
```

## 20.3 成本优化策略总览

| 策略 | 效果 | 适用场景 |
|------|------|---------|
| **选择合适模型** | 最大 | 简单任务用小模型 |
| **压缩历史消息** | 显著 | 多轮对话 |
| **文档截断/压缩** | 显著 | RAG 场景 |
| **减少调用次数** | 显著 | Agent 循环 |
| **流式输出** | 中等 | 提升体验，不改成本 |
| **缓存结果** | 显著 | 重复查询 |

## 20.4 选择合适的模型

### 模型选择决策树

```
任务复杂度评估
    ↓
简单任务（翻译、分类、提取）
    → 用 gpt-4o-mini / glm-4-flash
中等任务（总结、分析）
    → 用 gpt-4o-mini 或 claude-sonnet
复杂任务（推理、代码、多步骤）
    → 用 gpt-4o 或 claude-sonnet-4
极复杂任务（研究、创作）
    → 用 gpt-4o 或 claude-opus
```

### 动态模型选择

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

def get_model_by_complexity(task_description: str):
    """根据任务复杂度选择模型"""

    # 简单判断逻辑（实际可以用 LLM 分类）
    simple_keywords = ["翻译", "分类", "提取", "格式化"]
    complex_keywords = ["分析", "推理", "创作", "设计"]

    if any(kw in task_description for kw in simple_keywords):
        return ChatOpenAI(model="gpt-4o-mini")

    elif any(kw in task_description for kw in complex_keywords):
        return ChatOpenAI(model="gpt-4o")

    else:
        return ChatOpenAI(model="gpt-4o-mini")  # 默认用小模型


# 使用
simple_task = "将以下文本翻译成英文"
model = get_model_by_complexity(simple_task)
# model = gpt-4o-mini，节省约 10x 成本
```

### 多模型分工

```python
from langchain_openai import ChatOpenAI

# 分层架构：小模型预处理 + 大模型处理核心
fast_model = ChatOpenAI(model="gpt-4o-mini")    # 快速且便宜
smart_model = ChatOpenAI(model="gpt-4o")        # 强大但贵

def efficient_pipeline(user_input: str):
    """高效的处理流水线"""

    # Step 1: 小模型分类和预处理
    classification = fast_model.invoke(
        f"判断以下问题的类型（简单/复杂）: {user_input}"
    )

    if "简单" in classification.content:
        # Step 2a: 小模型处理简单问题
        return fast_model.invoke(user_input)

    else:
        # Step 2b: 大模型处理复杂问题
        return smart_model.invoke(user_input)
```

## 20.5 历史消息管理

### 问题：消息无限累积

```python
# ❌ 错误：每轮都追加消息，无限增长
messages = []

for i in range(100):
    messages.append(("user", f"问题{i}"))
    response = llm.invoke(messages)
    messages.append(("ai", response.content))

# 100轮后，messages 可能超过 10000 tokens！
```

### 策略1：窗口截断

```python
def truncate_messages(messages, max_messages=10):
    """只保留最近 N 条消息"""
    if len(messages) <= max_messages:
        return messages

    # 保留系统消息 + 最近的消息
    system_messages = [m for m in messages if m[0] == "system"]
    recent_messages = messages[-max_messages:]

    return system_messages + recent_messages

# 使用
messages = truncate_messages(messages, max_messages=10)
response = llm.invoke(messages)
```

### 策略2：Token 限制截断

```python
def truncate_by_tokens(messages, max_tokens=4000):
    """按 token 数截断"""
    total = 0
    result = []

    # 从最新消息开始，向前累加
    for msg in reversed(messages):
        estimated = estimate_tokens(msg[1]) if isinstance(msg, tuple) else estimate_tokens(msg.content)
        if total + estimated > max_tokens:
            break
        result.insert(0, msg)
        total += estimated

    return result
```

### 策略3：摘要压缩

```python
async def summarize_history(messages, llm):
    """将早期对话压缩为摘要"""
    if len(messages) <= 6:
        return messages

    # 需要压缩的早期消息
    to_summarize = messages[:-6]
    recent = messages[-6:]

    # 生成摘要
    history_text = "\n".join([
        f"{m[0]}: {m[1]}" for m in to_summarize
    ])

    summary = await llm.ainvoke(
        f"用一句话总结以下对话的关键信息：\n{history_text}"
    )

    # 返回摘要 + 最近消息
    return [("system", f"历史摘要: {summary.content}")] + recent
```

### 策略对比

| 策略 | Token节省 | 信息保留 | 适用场景 |
|------|----------|---------|---------|
| 窗口截断 | 50-70% | 丢失早期 | 一般对话 |
| Token截断 | 精确控制 | 丢失早期 | 短对话 |
| 摘要压缩 | 60-80% | 关键信息保留 | 长对话 |

## 20.6 RAG 成本优化

### 问题：检索文档过长

```python
# ❌ 错误：检索过多文档
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
docs = retriever.invoke("问题")
context = "\n".join([d.page_content for d in docs])
# context 可能超过 10000 tokens！
```

### 策略1：限制检索数量

```python
# ✅ 正确：只检索必要数量
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 筄略2：文档截断

```python
def truncate_docs(docs, max_chars_per_doc=500):
    """截断每个文档"""
    truncated = []
    for doc in docs:
        if len(doc.page_content) > max_chars_per_doc:
            doc.page_content = doc.page_content[:max_chars_per_doc]
        truncated.append(doc)
    return truncated

docs = truncate_docs(docs, max_chars_per_doc=500)
```

### 策略3：相关性过滤

```python
# 只保留高相关性文档
results_with_scores = vectorstore.similarity_search_with_score("问题", k=10)

# 过滤低相关性文档
threshold = 0.7
filtered_docs = [
    doc for doc, score in results_with_scores
    if score > threshold
]

# 只用高相关性文档
context = "\n".join([d.page_content for d in filtered_docs[:3]])
```

### 策略4：ContextualCompression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 用 LLM 压缩文档，只提取相关部分
compressor = LLMChainExtractor.from_llm(
    ChatOpenAI(model="gpt-4o-mini")  # 用小模型压缩
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# 检索结果被压缩，token 大幅减少
docs = compression_retriever.invoke("问题")
```

## 20.7 Agent 循环优化

### 问题：Agent 无限循环

```python
# ❌ 错误：无限制的循环
agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [("user", "复杂问题")]})
# 可能循环 10+ 次，每次都是 LLM 调用
```

### 策略1：设置最大步数

```python
# ✅ 正确：限制循环次数
result = agent.invoke(
    {"messages": [("user", "问题")]},
    config={"recursion_limit": 5},  # 最多 5 步
)
```

### 策略2：优化工具描述

```python
# ❌ 差：模糊描述导致多次尝试
@tool
def search(query: str) -> str:
    """搜索"""  # 太模糊
    pass

# ✅ 好：清晰描述减少选择错误
@tool
def search(query: str) -> str:
    """搜索网络信息。query 应包含完整关键词。
    当需要查找实时信息、新闻、公开资料时使用。
    返回前 5 条相关结果摘要。"""
    pass
```

### 策略3：工具返回更多信息

```python
# ❌ 差：返回太少，需要再次调用
@tool
def get_weather(city: str) -> str:
    return f"{city}: 25°C"  # 只有温度

# ✅ 好：一次返回所有需要的信息
@tool
def get_weather(city: str) -> str:
    """返回完整天气信息"""
    return f"""
城市: {city}
温度: 25°C
天气: 晴天
湿度: 60%
风速: 3级
空气质量: 良好
穿衣建议: 适合轻薄衣物
"""  # 避免 Agent 再次调用其他工具
```

### 策略4：使用小模型

```python
# Agent 多次调用，用小模型节省成本
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),  # 小模型
    tools=[...],
)
```

## 20.8 缓存策略

### 简单内存缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_invoke(prompt: str):
    """缓存相同问题的回答"""
    return llm.invoke(prompt).content

# 相同问题只调用一次
result1 = cached_invoke("什么是 RAG？")
result2 = cached_invoke("什么是 RAG？")  # 不调用 LLM，直接返回缓存
```

### 基于 embedding 的语义缓存

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

class SemanticCache:
    """语义缓存：相似问题返回缓存答案"""

    def __init__(self, threshold=0.95):
        self.cache = {}  # {embedding: response}
        self.embeddings = OpenAIEmbeddings()
        self.threshold = threshold

    def get(self, query: str):
        """查询缓存"""
        query_embedding = self.embeddings.embed_query(query)

        for cached_emb, response in self.cache.items():
            similarity = np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            )
            if similarity > self.threshold:
                return response  # 找到相似问题的缓存

        return None

    def set(self, query: str, response: str):
        """设置缓存"""
        embedding = self.embeddings.embed_query(query)
        self.cache[embedding] = response


# 使用
cache = SemanticCache()

def cached_query(query: str):
    cached = cache.get(query)
    if cached:
        return cached  # 立即返回，不调用 LLM

    response = llm.invoke(query).content
    cache.set(query, response)
    return response
```

## 20.9 批量处理优化

### 并发批量调用

```python
import asyncio

async def efficient_batch(questions: list[str]):
    """并发批量处理"""
    tasks = [llm.ainvoke(q) for q in questions]
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]

# 比 sync batch 快 3-5x，节省等待时间
asyncio.run(efficient_batch(questions))
```

### 分批处理长列表

```python
async def process_large_batch(items: list, batch_size: int = 10):
    """分批处理，避免同时请求过多"""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await llm.abatch(batch)
        results.extend(batch_results)

        # 避免 rate limit
        await asyncio.sleep(1)

    return results
```

## 20.10 成本监控

### Token 统计回调

```python
class CostTracker:
    """成本追踪"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.total_input = 0
        self.total_output = 0
        self.call_count = 0

        # 价格（美元/1M tokens）
        self.prices = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        }

    def track(self, response):
        """记录一次调用的 token"""
        usage = response.response_metadata.get('token_usage', {})
        self.total_input += usage.get('prompt_tokens', 0)
        self.total_output += usage.get('completion_tokens', 0)
        self.call_count += 1

    def get_cost(self) -> float:
        """计算总成本"""
        price = self.prices.get(self.model, {"input": 0, "output": 0})
        input_cost = self.total_input * price["input"] / 1_000_000
        output_cost = self.total_output * price["output"] / 1_000_000
        return input_cost + output_cost

    def report(self):
        """生成报告"""
        return {
            "model": self.model,
            "calls": self.call_count,
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "cost_usd": self.get_cost(),
        }


# 使用
tracker = CostTracker("gpt-4o-mini")

response = llm.invoke("问题")
tracker.track(response)

print(tracker.report())
```

### 每日成本限制

```python
class DailyLimiter:
    """每日成本限制"""

    def __init__(self, max_cost_usd: float = 10.0):
        self.max_cost = max_cost_usd
        self.tracker = CostTracker()

    def can_call(self) -> bool:
        """检查是否还能调用"""
        return self.tracker.get_cost() < self.max_cost

    def invoke(self, prompt: str):
        """带限制的调用"""
        if not self.can_call():
            raise RuntimeError(f"超出每日成本限制 ${self.max_cost}")

        response = llm.invoke(prompt)
        self.tracker.track(response)
        return response
```

## 20.11 成本优化检查清单

| 检查项 | 说明 |
|--------|------|
| **模型选择** | 简单任务用小模型 |
| **历史截断** | 窗口/摘要压缩历史消息 |
| **文档压缩** | RAG 截断文档、过滤低相关性 |
| **Agent限制** | 设置 recursion_limit |
| **工具优化** | 清晰描述 + 返回完整信息 |
| **缓存** | 相似问题缓存答案 |
| **批量并发** | asyncio.gather 并发处理 |
| **成本监控** | 追踪 token 用量和成本 |

## 20.12 本章小结

- 选择合适模型是最大的成本优化（10x+ 差距）
- 多轮对话必须管理历史消息：窗口截断或摘要压缩
- RAG 场景：限制检索数量、截断文档、过滤低相关性
- Agent 设置 recursion_limit，优化工具描述减少循环
- 使用缓存避免重复调用：简单缓存或语义缓存
- 批量处理用 asyncio.gather 并发
- 生产环境必须追踪 token 用量和成本
- 成本优化与用户体验需要平衡