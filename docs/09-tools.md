# 第9章：Tools 工具定义

## 9.1 什么是 Tool

Tool（工具）让 Agent 能够与外部世界交互。LLM 本身只能生成文本，但通过工具它可以：
- 搜索网页
- 查询数据库
- 调用 API
- 执行代码
- 读写文件

### LLM 如何选择工具

当工具绑定到 LLM 后，工具的名称、描述和参数会作为额外信息发送给模型。LLM 的决策过程：

```
用户提问 → LLM 分析问题 → 判断是否需要工具
                              ↓ 是              ↓ 否
                    选择最合适的工具      直接生成文本回答
                    生成参数 JSON
```

关键洞察：**LLM 不执行工具，只决定调用哪个工具和传什么参数**。实际执行由应用代码完成。

### 工具调用消息流

```
1. HumanMessage:  用户提问 "北京天气怎么样？"
2. AIMessage:     LLM 决定调用工具 → tool_calls=[{name: "search_weather", args: {city: "北京"}}]
3. ToolMessage:   工具执行结果 → "晴天，25°C"
4. AIMessage:     LLM 根据工具结果生成最终回答 → "北京今天是晴天，气温25°C"
```

## 9.2 工具的核心要素

每个工具需要定义：

| 要素 | 说明 | 为什么重要 |
|------|------|-----------|
| name | 工具名称 | LLM 用它来选择调用哪个工具 |
| description | 工具描述 | LLM 根据描述决定何时使用 |
| parameters | 参数 schema | 告诉 LLM 需要传什么参数 |
| function | 实际执行函数 | 工具的具体实现 |

## 9.3 @tool 装饰器 vs StructuredTool

| 特性 | `@tool` 装饰器 | `StructuredTool` |
|------|---------------|-----------------|
| 易用性 | 最简单，推荐大多数场景 | 需要更多代码 |
| 参数定义 | 自动从类型注解生成 | 手动定义 Pydantic schema |
| 描述来源 | 从 docstring 提取 | 手动指定 description |
| 灵活性 | 标准 | 支持异步、自定义验证等 |
| 适用场景 | 90% 的情况 | 需要精细控制时 |

## 9.4 使用 @tool 装饰器（最简单）

```python
from langchain_core.tools import tool

@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气信息"""
    # 实际应用中调用天气 API
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "深圳": "小雨，28°C",
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

# 工具属性
print(search_weather.name)          # "search_weather"
print(search_weather.description)   # "查询指定城市的天气信息"
print(search_weather.args)          # {'city': {'title': 'City', 'type': 'string'}}
```

### 类型注解的重要性

```python
from typing import Optional

@tool
def search_products(
    keyword: str,                            # 必填参数
    category: Optional[str] = None,          # 可选参数
    max_price: float = 1000.0,               # 带默认值
    sort_by: str = "relevance",              # 带默认值
) -> str:
    """搜索商品

    Args:
        keyword: 搜索关键词
        category: 商品分类（可选）
        max_price: 最高价格，默认1000
        sort_by: 排序方式，默认relevance
    """
    return f"搜索 '{keyword}' 的结果，分类: {category}，最高价: {max_price}"

# LangChain 会自动从类型注解和 docstring 生成参数 schema
print(search_products.args_schema.model_json_schema())
```

## 9.5 使用 StructuredTool（更灵活）

当需要更精细的控制时使用：

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(description="第一个数字")
    b: float = Field(description="第二个数字")
    operation: str = Field(description="运算类型: add/subtract/multiply/divide")

def calculator_func(a: float, b: float, operation: str) -> str:
    if operation == "add":
        return f"{a} + {b} = {a + b}"
    elif operation == "subtract":
        return f"{a} - {b} = {a - b}"
    elif operation == "multiply":
        return f"{a} * {b} = {a * b}"
    elif operation == "divide":
        return f"{a} / {b} = {a / b}"
    else:
        return f"不支持的运算: {operation}"

calculator = StructuredTool.from_function(
    func=calculator_func,
    name="calculator",
    description="执行基本数学运算",
    args_schema=CalculatorInput,
)
```

### 异步函数

```python
import httpx

async def fetch_url(url: str) -> str:
    """异步获取网页内容"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text[:500]

fetch_tool = StructuredTool.from_function(
    func=fetch_url,              # 同步函数（这里不需要）
    coroutine=fetch_url,         # 异步函数
    name="fetch_url",
    description="获取指定URL的网页内容",
)
```

## 9.6 直接将工具绑定到 LLM

不需要 Agent，手动测试工具调用：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 将工具绑定到 LLM
llm_with_tools = llm.bind_tools([search_weather, calculator])

# LLM 会根据问题选择合适的工具
response = llm_with_tools.invoke("北京天气怎么样？")

# 检查是否有工具调用
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"工具名: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
        # tool_call['name'] = "search_weather"
        # tool_call['args'] = {"city": "北京"}
```

### 手动执行工具调用

```python
from langchain_core.messages import HumanMessage, ToolMessage

# 1. 用户提问
messages = [HumanMessage(content="北京天气怎么样？")]

# 2. LLM 决定调用工具
response = llm_with_tools.invoke(messages)
messages.append(response)

# 3. 执行工具
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # 找到对应的工具并执行
    tools_map = {
        "search_weather": search_weather,
        "calculator": calculator,
    }
    result = tools_map[tool_name].invoke(tool_args)

    # 将工具结果添加到消息列表
    messages.append(ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
    ))

# 4. 把工具结果发给 LLM 生成最终回答
final_response = llm_with_tools.invoke(messages)
print(final_response.content)  # "北京今天是晴天，气温25°C"
```

## 9.7 Schema 自动生成原理

LangChain 使用 Pydantic 自动将 Python 类型注解转为 JSON Schema，这个 Schema 会发给 LLM：

```python
# Python 类型 → JSON Schema
str       → {"type": "string"}
int       → {"type": "integer"}
float     → {"type": "number"}
bool      → {"type": "boolean"}
Optional  → 不在 required 列表中
Field()   → 添加 description、default、constraints
Literal   → {"enum": [...]}
```

LLM 收到的是 JSON Schema 格式的参数描述，因此**类型注解和 docstring 直接影响 LLM 的工具选择准确性**。

## 9.8 工具的参数 Schema

LangChain 使用 Pydantic 自动生成 JSON Schema：

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, ge=1, le=20, description="结果数量")
    language: Literal["zh", "en"] = Field(default="zh", description="语言")

@tool(args_schema=SearchInput)
def search_web(query: str, num_results: int = 5, language: str = "zh") -> str:
    """搜索网页信息"""
    return f"搜索 '{query}'，返回 {num_results} 条 {language} 结果"

# 查看生成的 schema
print(search_web.args_schema.model_json_schema())
```

## 9.9 内置工具

LangChain 提供了一些现成的工具：

```python
# Wikipedia 搜索
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Arxiv 论文搜索
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

# Python REPL（执行代码，注意安全！）
from langchain_community.tools import PythonREPLTool
python_repl = PythonREPLTool()

# DuckDuckGo 搜索
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
```

## 9.10 工具设计的最佳实践

### 1. 描述要清晰具体

```python
# 好的描述
@tool
def search_order(order_id: str) -> str:
    """根据订单ID查询订单详情，包括商品列表、价格、配送状态。
    当用户询问订单状态、物流信息时使用此工具。
    order_id 格式：ORD-XXXXXX"""
    pass

# 差的描述
@tool
def search_order(order_id: str) -> str:
    """查询订单"""  # 太模糊
    pass
```

### 2. 返回结构化信息

```python
@tool
def get_stock_price(symbol: str) -> str:
    """查询股票价格"""
    # 返回格式化的文本，而不是原始数据
    price = fetch_price(symbol)
    return f"股票 {symbol} 当前价格: ¥{price:.2f}"
```

### 3. 处理错误情况

```python
@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询"""
    try:
        result = execute_sql(sql)
        if not result:
            return "查询结果为空"
        return str(result)
    except Exception as e:
        return f"查询失败: {str(e)}"  # 返回错误信息，不要抛异常
```

### 4. 工具数量控制

- 建议 3-10 个工具
- 太多：LLM 会混淆选择
- 太少：Agent 能力受限

### 常见错误

| 错误 | 问题 | 正确做法 |
|------|------|---------|
| 描述太模糊 | LLM 不知道何时使用 | 说明用途 + 使用场景 |
| 抛出异常 | Agent 崩溃 | 返回描述性错误文本 |
| 返回原始数据 | LLM 难以理解 | 返回格式化的文本 |
| 工具太多(>10) | LLM 选择困难 | 3-7 个，按需分组 |
| 工具职责重叠 | LLM 选择混乱 | 每个工具职责清晰 |
| 无类型注解 | Schema 不完整 | 所有参数加类型注解 |

## 9.11 本章小结

- `@tool` 装饰器是最简单的工具定义方式
- `StructuredTool` 提供更精细的控制
- 工具的 `description` 和参数描述至关重要，直接影响 LLM 的选择
- `bind_tools()` 将工具绑定到 LLM，支持手动测试
- 工具设计原则：描述清晰、错误处理、返回有意义的文本
