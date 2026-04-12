# 第3章：Prompt 模板

## 3.1 为什么需要 Prompt 模板

硬编码 prompt 的问题：
- 难以复用和维护
- 变量拼接容易出错
- 无法方便地进行版本管理

Prompt 模板的优势：
- 变量占位，动态生成 prompt
- 支持组合和继承
- 类型安全（LangChain 会校验变量）

## 3.2 PromptTemplate（纯文本模板）

```python
from langchain_core.prompts import PromptTemplate

# 创建模板
template = PromptTemplate.from_template("给我讲一个关于{topic}的笑话")

# 使用模板
prompt = template.invoke({"topic": "程序员"})
print(prompt.to_string())
# 给我讲一个关于程序员的笑话

# 也可以用 format
text = template.format(topic="程序员")
```

### 带多个变量的模板

```python
template = PromptTemplate.from_template(
    "请将以下{source_language}文本翻译成{target_language}：\n\n{text}"
)

prompt = template.invoke({
    "source_language": "中文",
    "target_language": "英文",
    "text": "今天天气真好",
})
```

### 自动识别变量

```python
# from_template 会自动识别 {variable} 格式的变量
template = PromptTemplate.from_template("{greeting}, {name}!")
print(template.input_variables)  # ['greeting', 'name']
```

## 3.3 ChatPromptTemplate（对话模板，推荐）

这是最常用的模板类型，专为 ChatModel 设计：

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式一：从消息元组构建（最推荐）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用{style}的风格回答问题。"),
    ("human", "{question}"),
])

# 使用模板
chain = prompt | llm  # LCEL 语法，第5章详解
response = chain.invoke({
    "role": "资深Python开发者",
    "style": "简洁专业",
    "question": "什么是装饰器？",
})
```

### 消息类型对照

```python
prompt = ChatPromptTemplate.from_messages([
    # (类型, 内容模板)
    ("system", "你是{role}"),                    # 系统消息
    ("human", "{question}"),                      # 用户消息
    ("ai", "好的，让我想想..."),                   # AI 历史
    ("human", "{follow_up}"),                     # 后续提问
])
```

## 3.4 MessagesPlaceholder（历史消息占位）

在构建带记忆的对话时，需要动态插入历史消息：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 历史消息占位
    ("human", "{question}"),
])

# 使用时传入历史消息
from langchain_core.messages import HumanMessage, AIMessage

response = prompt | llm
result = response.invoke({
    "chat_history": [
        HumanMessage(content="我叫小明"),
        AIMessage(content="你好小明！有什么可以帮你的？"),
    ],
    "question": "我叫什么名字？",  # AI 能记住历史上下文
})
```

## 3.5 Few-Shot 提示（少样本示例）

通过示例引导 LLM 理解期望的输出格式：

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# 定义示例
examples = [
    {"input": "开心", "output": "😄 开心 (happy) - 形容心情愉悦"},
    {"input": "难过", "output": "😢 难过 (sad) - 形容心情低落"},
    {"input": "生气", "output": "😠 生气 (angry) - 形容心情愤怒"},
]

# 示例模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Few-Shot 模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 组合成完整 prompt
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感词典，按照示例格式解释情感词汇。"),
    few_shot_prompt,
    ("human", "{word}"),
])

# 使用
chain = final_prompt | llm
result = chain.invoke({"word": "紧张"})
```

## 3.6 模板组合

模板可以像乐高一样组合：

```python
# 基础系统 prompt
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{domain}专家。"),
])

# 追加用户消息
full_prompt = system_prompt + ("human", "{question}")

# 等价于
full_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{domain}专家。"),
    ("human", "{question}"),
])
```

## 3.7 PipelinePromptTemplate（管道组合）

> **注意**：`PipelinePromptTemplate` 在新版 LangChain 中已废弃。推荐使用 LCEL 管道或 `+` 操作符组合模板。

将多个子模板组合成一个大模板：

```python
from langchain_core.prompts import PipelinePromptTemplate, PromptTemplate

# 完整 prompt 的框架
full_template = PromptTemplate.from_template("""
{instruction}

{context}

问题: {question}
""")

# 子模板：指令
instruction_template = PromptTemplate.from_template(
    "你是一个{role}，请回答以下问题。"
)

# 子模板：上下文
context_template = PromptTemplate.from_template(
    "参考信息：\n{reference}"
)

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=[
        ("instruction", instruction_template),
        ("context", context_template),
    ]
)
```

**推荐替代方案**：使用 LCEL 管道或直接拼接字符串更灵活：

```python
# 方式一：直接在 chain 中组合（推荐）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。请回答以下问题。"),
    ("human", "参考信息：{reference}\n\n问题：{question}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "role": "Python 专家",
    "reference": "装饰器是 Python 的高级特性...",
    "question": "什么是装饰器？",
})
```

## 3.8 实用技巧

### 多行文本格式化

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个代码审查助手。请按以下格式回复：

## 问题分析
<分析代码中的问题>

## 修改建议
<给出具体的修改建议>

## 改进后的代码
<给出改进后的完整代码>
"""),
    ("human", "请审查以下代码：\n```python\n{code}\n```"),
])
```

### 条件性内容

```python
# 根据条件动态构建 prompt
def create_prompt(include_examples: bool):
    messages = [("system", "你是一个翻译助手。")]
    if include_examples:
        messages.append(("human", "示例：苹果 → apple"))
        messages.append(("ai", "明白了，我会按照这个格式翻译。"))
    messages.append(("human", "{text}"))
    return ChatPromptTemplate.from_messages(messages)
```

## 3.9 本章小结

- `PromptTemplate`：纯文本模板，适用于 LLM 接口
- `ChatPromptTemplate`：对话模板，适用于 ChatModel，**最常用**
- `MessagesPlaceholder`：动态插入消息列表，用于对话历史
- `FewShotChatMessagePromptTemplate`：少样本示例引导输出格式
- 模板支持组合（`+` 操作符）和管道（`PipelinePromptTemplate`）
