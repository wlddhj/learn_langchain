# 第4章：输出解析器

## 4.1 为什么需要输出解析器

LLM 返回的是纯文本字符串，但应用通常需要：
- 结构化的数据（JSON、列表等）
- 特定格式的输出
- 类型安全的 Python 对象

输出解析器负责将 LLM 的文本输出转换为程序可用的结构化数据。

## 4.2 StrOutputParser（最简单）

将 AIMessage 转换为纯字符串：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("human", "给我讲一个关于{topic}的笑话"),
])

# StrOutputParser 提取 AIMessage.content
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "程序员"})
print(type(result))  # <class 'str'>
print(result)        # 纯字符串，不是 AIMessage
```

> **提示**：几乎所有 chain 的末尾都会用到 `StrOutputParser`。

## 4.3 CommaSeparatedListOutputParser

解析逗号分隔的列表：

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

# 获取格式指令（可以注入到 prompt 中）
format_instructions = parser.get_format_instructions()
print(format_instructions)
# "Your response should be a list of comma separated values, eg: `foo, bar, baz`"

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个命名专家。{format_instructions}"),
    ("human", "给我5个{language}项目的名字"),
])

chain = prompt | llm | parser

result = chain.invoke({
    "language": "Python",
    "format_instructions": parser.get_format_instructions(),
})
print(result)  # ['PyFlow', 'DataForge', 'CodeNest', 'AlgoPy', 'ByteCraft']
print(type(result))  # <class 'list'>
```

## 4.4 JsonOutputParser

解析 JSON 输出：

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义期望的数据结构
class Recipe(BaseModel):
    name: str = Field(description="菜名")
    ingredients: list[str] = Field(description="食材列表")
    steps: list[str] = Field(description="烹饪步骤")
    cook_time: str = Field(description="烹饪时间")

parser = JsonOutputParser(pydantic_object=Recipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个厨艺专家。\n{format_instructions}"),
    ("human", "教我做{dish}"),
])

chain = prompt | llm | parser

result = chain.invoke({
    "dish": "番茄炒蛋",
    "format_instructions": parser.get_format_instructions(),
})

print(result)
# {
#   "name": "番茄炒蛋",
#   "ingredients": ["番茄", "鸡蛋", "盐", "糖", "食用油"],
#   "steps": ["1. 番茄切块...", "2. 鸡蛋打散..."],
#   "cook_time": "10分钟"
# }
print(type(result))  # <class 'dict'>
```

## 4.5 PydanticOutputParser

返回 Pydantic 模型实例（比 JsonOutputParser 多了类型校验）：

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

class BookRecommendation(BaseModel):
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    genre: str = Field(description="类型")
    rating: float = Field(description="评分 1.0-5.0")
    reason: str = Field(description="推荐理由")

parser = PydanticOutputParser(pydantic_object=BookRecommendation)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个图书推荐专家。\n{format_instructions}"),
    ("human", "推荐一本关于{topic}的书"),
])

chain = prompt | llm | parser

result = chain.invoke({
    "topic": "人工智能入门",
    "format_instructions": parser.get_format_instructions(),
})

print(result.title)    # 直接访问属性
print(result.author)   # Pydantic 模型实例
print(type(result))    # <class 'BookRecommendation'>
```

### JsonOutputParser vs PydanticOutputParser

| 特性 | JsonOutputParser | PydanticOutputParser |
|------|-----------------|---------------------|
| 返回类型 | `dict` | Pydantic 模型实例 |
| 类型校验 | 无 | 有（字段类型、约束） |
| 属性访问 | `result["key"]` | `result.key` |
| 推荐场景 | 简单 JSON | 需要类型安全时 |

## 4.6 结构化输出的现代方案（推荐）

> **最佳实践**：使用 `with_structured_output` 替代手动解析器。

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class SentimentResult(BaseModel):
    """情感分析结果"""
    sentiment: str = Field(description="情感倾向: positive/negative/neutral")
    confidence: float = Field(description="置信度 0-1")
    keywords: list[str] = Field(description="关键词列表")

llm = ChatOpenAI(model="gpt-4o-mini")

# 直接让 LLM 返回结构化数据，无需解析器
structured_llm = llm.with_structured_output(SentimentResult)

result = structured_llm.invoke("这个产品太棒了，用起来非常流畅！")
print(result.sentiment)    # positive
print(result.confidence)   # 0.95
print(result.keywords)     # ['棒', '流畅']
```

**为什么推荐 `with_structured_output`？**
- 更可靠：利用 LLM 的 function calling / tool use 能力
- 更简洁：无需编写 format_instructions
- 更准确：解析失败率更低

### 底层实现策略

`with_structured_output` 有三种实现策略，LangChain 会根据模型提供商自动选择最优方案：

| 策略 | 原理 | 严格程度 |
|------|------|----------|
| `openai-tools` | 通过 tool calling + `tool_choice` 约束输出 | 最高，保证 schema 合规 |
| `json_schema` | 通过 `response_format={"type":"json_schema"}` 约束 | 高，原生 JSON Schema 校验 |
| `json_mode` | 通过 `response_format={"type":"json_object"}` 约束 | 中，只保证 JSON 格式，不保证 schema |

可以通过 `method` 参数显式指定：

```python
# 显式指定使用 json_mode
structured_llm = llm.with_structured_output(SentimentResult, method="json_mode")
```

### 各提供商支持情况

本项目通过 `ChatOpenAI` + `base_url` 接入多家国产 LLM，各家对 `with_structured_output` 的支持程度不同：

| 提供商 | `openai-tools` | `json_schema` | `json_mode` | 备注 |
|--------|:-:|:-:|:-:|------|
| **OpenAI** (GPT-4o) | ✅ | ✅ | ✅ | 原生支持，最稳定 |
| **Anthropic** (Claude) | ✅ | ✅ | ✅ | 原生支持 |
| **Google Gemini** | ✅ | ✅ | ✅ | 原生支持 |
| **Qwen** (通义千问) | ⚠️ 部分 | ⚠️ 部分模型 | ✅ | 不支持 `stream=True` + `tools` 同时使用 |
| **GLM** (智谱 AI) | ⚠️ 有限 | ⚠️ 有限 | ✅ | OpenAI 兼容接口对 `json_schema` 支持不完整 |
| **DeepSeek** | ⚠️ 部分 | ⚠️ 有限 | ✅ | 建议使用 `json_mode` |

> **注意**：由于 Qwen、GLM、DeepSeek 均通过 OpenAI 兼容接口接入，默认策略 `openai-tools` 可能不完全兼容。如果遇到报错，请切换为 `method="json_mode"`。

### 推荐做法

针对本项目的国产模型使用场景，按兼容性从高到低排列：

**方案一：`with_structured_output` + `json_mode`（推荐）**

兼容性好，大多数提供商支持：

```python
structured_llm = llm.with_structured_output(SentimentResult, method="json_mode")
```

**方案二：`PydanticOutputParser`（通用兜底）**

不依赖任何原生 API 能力，纯 Prompt 工程，适用于所有 LLM：

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=SentimentResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是情感分析助手。\n{format_instructions}"),
    ("human", "{text}"),
])

chain = prompt | llm | parser
result = chain.invoke({
    "text": "这个产品太棒了！",
    "format_instructions": parser.get_format_instructions(),
})
```

### 三种方式对比

| 方式 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| `with_structured_output()` | 调用简洁，无需手写 format_instructions | 依赖提供商 API 支持 | OpenAI / Anthropic 等原生支持的平台 |
| `with_structured_output(method="json_mode")` | 兼容性好，大多数提供商支持 | 只保证 JSON 格式，schema 可能不完全匹配 | Qwen / GLM / DeepSeek 等兼容接口 |
| `PydanticOutputParser` | 通用性最强，任何 LLM 都可用 | 需要手动注入 format_instructions | 所有提供商的兜底方案 |

## 4.7 自定义输出解析器

当内置解析器不满足需求时：

```python
from langchain_core.output_parsers import BaseOutputParser

class BooleanOutputParser(BaseOutputParser[bool]):
    """将 LLM 输出解析为布尔值"""

    true_values = {"是", "yes", "true", "对", "正确", "1"}
    false_values = {"否", "no", "false", "错", "错误", "0"}

    def parse(self, text: str) -> bool:
        cleaned = text.strip().lower()
        if cleaned in self.true_values:
            return True
        if cleaned in self.false_values:
            return False
        raise ValueError(f"无法将 '{text}' 解析为布尔值")

    @property
    def _type(self) -> str:
        return "boolean_output_parser"

# 使用
parser = BooleanOutputParser()
print(parser.parse("是"))    # True
print(parser.parse("no"))    # False
```

## 4.8 输出修复（Output Fixing）

当 LLM 输出不符合预期格式时，自动修复：

```python
from langchain_core.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)

# OutputFixingParser：解析失败时，让 LLM 自己修复输出
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatOpenAI(model="gpt-4o-mini"),
)

# 即使 LLM 输出格式有瑕疵，fixing_parser 也会尝试修复
```

## 4.9 本章小结

- `StrOutputParser`：提取纯文本，几乎所有 chain 都需要
- `JsonOutputParser` / `PydanticOutputParser`：解析结构化 JSON 输出
- **推荐使用 `with_structured_output`** 代替手动解析器
- 自定义解析器继承 `BaseOutputParser` 实现特定需求
- `OutputFixingParser` 可以在解析失败时自动让 LLM 修复输出
