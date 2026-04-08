# 第5章：Chain 链式调用与 LCEL

## 5.1 什么是 LCEL

LCEL (LangChain Expression Language) 是 LangChain 的核心编程范式，使用 `|` 管道操作符将组件串联起来：

```python
chain = prompt | llm | output_parser
```

这种设计灵感来自 Unix 管道，让数据从左到右流动。

### LCEL 的优势

| 特性 | 说明 |
|------|------|
| 简洁 | 一行代码定义完整链路 |
| 流式支持 | 自动获得流式输出能力 |
| 异步支持 | 同一套代码支持 sync/async |
| 批量支持 | 自动支持 batch 调用 |
| 可观测性 | 自动集成 LangSmith 追踪 |

## 5.2 基础 Chain

### 最简 Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("用一句话解释{concept}")
parser = StrOutputParser()

# LCEL 管道
chain = prompt | llm | parser

# 调用
result = chain.invoke({"concept": "机器学习"})
print(result)  # 纯字符串
```

### 数据流解析

```
{"concept": "机器学习"}
    ↓
[ChatPromptTemplate] → 生成消息列表
    ↓
[ChatModel] → 生成 AIMessage
    ↓
[StrOutputParser] → 提取字符串
    ↓
"机器学习是让计算机从数据中自动学习规律和模式的技术。"
```

## 5.3 Runnable 接口

所有 LCEL 组件都实现 `Runnable` 接口，提供统一的方法：

```python
# 同步调用
result = chain.invoke({"concept": "机器学习"})

# 异步调用
result = await chain.ainvoke({"concept": "机器学习"})

# 批量调用
results = chain.batch([
    {"concept": "机器学习"},
    {"concept": "深度学习"},
    {"concept": "强化学习"},
])

# 流式输出
for chunk in chain.stream({"concept": "机器学习"}):
    print(chunk, end="", flush=True)

# 异步流式
async for chunk in chain.astream({"concept": "机器学习"}):
    print(chunk, end="", flush=True)
```

> 一旦用 LCEL 定义了 chain，就自动获得了 invoke/batch/stream 的能力，无需额外代码。

## 5.4 RunnablePassthrough

数据透传，用于在不修改数据的情况下传递参数：

```python
from langchain_core.runnables import RunnablePassthrough

# RunnablePassthrough: 直接透传输入
# RunnablePassthrough.assign(): 在输入上追加新字段

chain = RunnablePassthrough.assign(
    # 对输入的 "question" 字段不做处理，直接透传
    # 同时可以新增计算字段
)

# 实际示例：同时保留原始输入和 LLM 输出
chain = {
    "question": RunnablePassthrough(),         # 透传原始问题
    "answer": prompt | llm | StrOutputParser(),  # LLM 生成的答案
}

result = chain.invoke("什么是Python？")
print(result)
# {"question": "什么是Python？", "answer": "Python是一种..."}
```

### 常见模式：同时返回检索文档和回答

```python
from langchain_core.runnables import RunnablePassthrough

# RAG 场景：保留检索到的文档 + LLM 回答
chain = {
    "context": retriever | format_docs,   # 检索文档
    "question": RunnablePassthrough(),     # 透传问题
} | prompt | llm | StrOutputParser()
```

## 5.5 RunnableLambda

将任意 Python 函数包装为 Runnable：

```python
from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> int:
    """计算字数"""
    return len(text.split())

# 包装为 Runnable
count_runnable = RunnableLambda(word_count)

# 可以插入 chain 中
chain = prompt | llm | StrOutputParser() | RunnableLambda(word_count)

result = chain.invoke({"concept": "机器学习"})
print(result)  # 输出的字数
```

### 异步函数包装

```python
async def async_word_count(text: str) -> int:
    await asyncio.sleep(0)  # 模拟异步操作
    return len(text.split())

chain = prompt | llm | StrOutputParser() | RunnableLambda(async_word_count)
```

## 5.6 分支与路由

### 条件分支 (RunnableBranch)

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    # (条件函数, 执行的 chain)
    (lambda x: "数学" in x["question"], math_chain),
    (lambda x: "历史" in x["question"], history_chain),
    (lambda x: "编程" in x["question"], code_chain),
    # 默认分支（条件都不满足时执行）
    default_chain,
)

result = branch.invoke({"question": "请解释数学中的导数"})
```

### 动态路由（LLM 路由）

```python
from langchain_core.runnables import RunnableLambda

def route_chain(input_dict):
    """根据分类结果路由到不同的 chain"""
    question = input_dict["question"]
    category = input_dict["category"]

    chains = {
        "math": math_chain,
        "history": history_chain,
        "code": code_chain,
    }
    return chains.get(category, default_chain).invoke(question)

# 先分类，再路由
classification_prompt = ChatPromptTemplate.from_template(
    "请将以下问题分类为 math/history/code 中的一个词：\n{question}"
)

full_chain = (
    {
        "category": classification_prompt | llm | StrOutputParser(),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(route_chain)
)
```

## 5.7 并行执行

```python
from langchain_core.runnables import RunnableParallel

# 同时执行多个 chain，结果合并为字典
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain,
)

result = parallel.invoke("这部电影讲述了一个关于勇气和友情的故事...")
print(result["summary"])     # 摘要
print(result["keywords"])    # 关键词
print(result["sentiment"])   # 情感分析
```

> **注意**：字典语法 `{key: runnable}` 是 `RunnableParallel` 的简写。

## 5.8 Chain 的 fallback 机制

当一个 chain 失败时，自动尝试备选方案：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 主 chain 使用 GPT-4，失败时回退到 Claude
fallback_chain = (
    primary_prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
).with_fallbacks([
    secondary_prompt
    | ChatAnthropic(model="claude-sonnet-4-20250514")
    | StrOutputParser(),
])
```

## 5.9 重试机制

```python
# 自动重试（带指数退避）
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,      # 最多重试 3 次
    retry_if_exception_type=(Exception,),  # 遇到什么异常重试
)
```

## 5.10 完整实战示例：多语言翻译器

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

translate_prompt = ChatPromptTemplate.from_template(
    "将以下文本翻译成{language}，只输出翻译结果：\n\n{text}"
)

review_prompt = ChatPromptTemplate.from_template(
    "请评价以下翻译的质量，给出1-10分：\n\n原文: {text}\n译文: {translation}"
)

# 翻译 chain
translate_chain = translate_prompt | llm | StrOutputParser()

# 评价 chain
review_chain = review_prompt | llm | StrOutputParser()

# 完整 chain：翻译 + 评价并行
full_chain = (
    {
        "translation": translate_chain,
        "text": lambda x: x["text"],
        "language": lambda x: x["language"],
    }
    | RunnableParallel(
        translation=lambda x: x["translation"],
        review=lambda x: review_chain.invoke({
            "text": x["text"],
            "translation": x["translation"],
        }),
    )
)

result = full_chain.invoke({
    "text": "人工智能正在改变世界",
    "language": "English",
})
print(result["translation"])
print(result["review"])
```

## 5.11 本章小结

- **LCEL** 是 LangChain 的核心范式，用 `|` 管道连接组件
- 所有组件实现 `Runnable` 接口，统一支持 `invoke/batch/stream`
- `RunnablePassthrough`：透传数据
- `RunnableLambda`：包装自定义函数
- `RunnableParallel`：并行执行多个 chain
- `RunnableBranch`：条件路由
- `.with_fallbacks()`：失败回退
- `.with_retry()`：自动重试
