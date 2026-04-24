# 第22章：测试与质量保证

## 22.1 LLM 应用测试的难点

| 难点 | 原因 | 影响 |
|------|------|------|
| **输出不确定性** | LLM 输出每次可能不同 | 难以精确匹配 |
| **依赖外部服务** | LLM API 可能不稳定 | 测试失败 |
| **执行路径不确定** | Agent 决策路径可变 | 覆盖率低 |
| **评估标准模糊** | "好回答"难以量化 | 断言困难 |

## 22.2 测试策略总览

```
┌────────────────────────────────────────────┐
│              测试金字塔                      │
│                                            │
│           E2E测试（少量）                    │
│         ─────────────────                   │
│          集成测试（适量）                    │
│       ─────────────────────                 │
│          单元测试（大量）                    │
│    ──────────────────────────────           │
│          Mock测试（核心逻辑）                │
└────────────────────────────────────────────┘
```

## 22.3 Mock LLM 测试

### 为什么 Mock

- 避免 API 调用成本
- 测试稳定可靠
- 验证逻辑流程

### 简单 Mock LLM

```python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

class MockChatModel(BaseChatModel):
    """简单的 Mock LLM"""

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}

    def _generate(self, messages, **kwargs):
        # 根据输入返回预设响应
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

        # 查找预设响应
        for key, response in self.responses.items():
            if key in content:
                return AIMessage(content=response)

        # 默认响应
        return AIMessage(content="Mock response")

    @property
    def _llm_type(self):
        return "mock"

# 使用
mock_llm = MockChatModel({
    "天气": "北京今天晴天，25°C",
    "时间": "现在是上午10点",
})

response = mock_llm.invoke("北京天气怎么样？")
assert "晴天" in response.content
```

### 预设响应序列

```python
class SequenceMockModel(BaseChatModel):
    """按序列返回响应的 Mock"""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.index = 0

    def _generate(self, messages, **kwargs):
        if self.index < len(self.responses):
            response = AIMessage(content=self.responses[self.index])
            self.index += 1
            return response
        return AIMessage(content="No more responses")

    @property
    def _llm_type(self):
        return "sequence_mock"


# 使用：测试多步流程
mock_llm = SequenceMockModel([
    "决定调用工具: search_weather",
    "北京天气: 晴天 25°C",
])

# 第一次调用
r1 = mock_llm.invoke("北京天气？")
assert "search_weather" in r1.content

# 第二次调用
r2 = mock_llm.invoke("工具结果")
assert "晴天" in r2.content
```

## 22.4 Chain 单元测试

### 测试 Prompt 模板

```python
from langchain_core.prompts import ChatPromptTemplate

def test_prompt_template():
    """测试 Prompt 模板"""
    template = ChatPromptTemplate.from_template(
        "将以下文本翻译成{language}: {text}"
    )

    # 测试变量填充
    result = template.invoke({
        "language": "英文",
        "text": "你好",
    })

    assert "英文" in result.to_string()
    assert "你好" in result.to_string()

    # 测试缺失变量抛异常
    try:
        template.invoke({"language": "英文"})  # 缺少 text
        assert False  # 应该抛异常
    except KeyError:
        assert True


test_prompt_template()
```

### 测试输出解析器

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel

def test_output_parser():
    """测试输出解析器"""
    # StrOutputParser
    parser = StrOutputParser()
    message = AIMessage(content="测试内容")
    result = parser.invoke(message)
    assert result == "测试内容"

    # JsonOutputParser
    class Person(BaseModel):
        name: str
        age: int

    json_parser = JsonOutputParser(pydantic_object=Person)
    json_text = '{"name": "张三", "age": 25}'
    result = json_parser.invoke(json_text)
    assert result["name"] == "张三"
    assert result["age"] == 25


test_output_parser()
```

### 测试 LCEL Chain

```python
def test_lcel_chain():
    """测试 LCEL Chain"""
    # 使用 Mock LLM
    mock_llm = MockChatModel({"机器学习": "机器学习是AI的分支"})

    template = ChatPromptTemplate.from_template("解释{topic}")
    parser = StrOutputParser()

    chain = template | mock_llm | parser

    # 测试调用
    result = chain.invoke({"topic": "机器学习"})
    assert "AI" in result

    # 测试批量
    results = chain.batch([
        {"topic": "机器学习"},
        {"topic": "深度学习"},
    ])
    assert len(results) == 2


test_lcel_chain()
```

## 22.5 Agent 测试

### Mock 工具测试

```python
from langchain_core.tools import tool

def test_mock_tool():
    """测试 Mock 工具"""

    @tool
    def mock_weather(city: str) -> str:
        """模拟天气查询"""
        return f"{city}: 晴天 25°C"

    # 测试工具调用
    result = mock_weather.invoke({"city": "北京"})
    assert "晴天" in result

    # 测试工具属性
    assert mock_weather.name == "mock_weather"
    assert "city" in str(mock_weather.args)


test_mock_tool()
```

### Agent 流程测试

```python
def test_agent_flow():
    """测试 Agent 执行流程"""

    # Mock LLM：预设执行步骤
    mock_llm = SequenceMockModel([
        # Step 1: 决定调用工具
        AIMessage(
            content="",
            tool_calls=[{"name": "weather", "args": {"city": "北京"}, "id": "call1"}]
        ),
        # Step 2: 根据工具结果回答
        AIMessage(content="北京今天晴天，气温25°C"),
    ])

    # Mock 工具
    @tool
    def weather(city: str) -> str:
        return f"{city}: 晴天 25°C"

    # 手动模拟执行流程
    messages = [("user", "北京天气？")]

    # Step 1
    r1 = mock_llm.invoke(messages)
    assert r1.tool_calls[0]["name"] == "weather"

    # Step 2: 执行工具
    messages.append(r1)
    tool_result = weather.invoke(r1.tool_calls[0]["args"])
    messages.append(ToolMessage(content=tool_result, tool_call_id="call1"))

    # Step 3
    r2 = mock_llm.invoke(messages)
    assert "晴天" in r2.content


test_agent_flow()
```

## 22.6 RAG 测试

### 测试文档切分

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_text_splitter():
    """测试文本切分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    text = "这是一段很长的文本..." * 50
    chunks = splitter.split_text(text)

    # 验证切分
    assert len(chunks) > 1

    # 验证 chunk 大小
    for chunk in chunks:
        assert len(chunk) <= 100

    # 验证 overlap
    if len(chunks) > 1:
        overlap = chunks[0][-20:]
        assert overlap in chunks[1]


test_text_splitter()
```

### Mock 向量检索

```python
class MockRetriever:
    """Mock 检索器"""

    def __init__(self, documents: list[str]):
        self.documents = documents

    def invoke(self, query: str):
        # 简单返回所有文档
        from langchain_core.documents import Document
        return [Document(page_content=d) for d in self.documents]

def test_rag_chain():
    """测试 RAG Chain"""
    mock_retriever = MockRetriever([
        "LangChain 是一个框架",
        "RAG 是检索增强生成",
    ])

    mock_llm = MockChatModel({"RAG": "RAG 是检索增强生成技术"})

    def format_docs(docs):
        return "\n".join(d.page_content for d in docs)

    chain = (
        {"context": mock_retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template("基于{context}回答{question}")
        | mock_llm
        | StrOutputParser()
    )

    result = chain.invoke("什么是 RAG？")
    assert "检索" in result


test_rag_chain()
```

## 22.7 集成测试

### 真实 LLM 调用测试

```python
import pytest
from langchain_openai import ChatOpenAI

# 仅在特定环境运行
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="需要 API Key")
def test_real_llm():
    """真实 LLM 测试（慢、贵）"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke("你好")
    assert response.content
    assert len(response.content) > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="需要 API Key")
def test_real_agent():
    """真实 Agent 测试"""
    from langgraph.prebuilt import create_react_agent

    @tool
    def search(query: str) -> str:
        return f"搜索结果: {query}"

    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[search],
    )

    result = agent.invoke({"messages": [("user", "搜索 LangChain")]})

    assert result["messages"]
    assert result["messages"][-1].content
```

### 流程完整性测试

```python
def test_full_pipeline():
    """完整流程测试"""
    # 使用真实组件，但用小模型和限制配置

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        request_timeout=10,
        max_tokens=100,
    )

    # 测试完整 chain
    chain = prompt | llm | parser

    result = chain.invoke({"topic": "AI"}, config={"recursion_limit": 3})

    # 验证输出
    assert result
    assert isinstance(result, str)
```

## 22.8 回答质量评估

### 精确匹配 vs 模糊匹配

```python
def test_exact_match():
    """精确匹配测试（不推荐用于 LLM）"""
    response = llm.invoke("1+1等于多少？")
    assert "2" in response.content  # 可以精确匹配


def test_fuzzy_match():
    """模糊匹配测试（推荐）"""
    response = llm.invoke("什么是机器学习？")

    # 检查关键词
    keywords = ["学习", "数据", "AI", "算法"]
    assert any(kw in response.content for kw in keywords)

    # 检查长度合理
    assert 50 < len(response.content) < 500
```

### 使用 LLM 评估 LLM

```python
async def llm_evaluate(question: str, answer: str) -> dict:
    """用 LLM 评估回答质量"""

    evaluator = ChatOpenAI(model="gpt-4o-mini")

    eval_prompt = f"""评估以下回答的质量：

问题: {question}
回答: {answer}

请从以下维度评分（1-5分）：
1. 相关性：回答是否与问题相关
2. 准确性：回答是否事实正确
3. 完整性：回答是否完整
4. 清晰度：回答是否易于理解

输出 JSON 格式: {"relevance": x, "accuracy": x, "completeness": x, "clarity": x}"""

    response = await evaluator.ainvoke(eval_prompt)

    # 解析评分
    import json
    scores = json.loads(response.content)
    return scores


# 使用
scores = await llm_evaluate("什么是 RAG？", "RAG 是检索增强生成...")
assert scores["relevance"] >= 3
```

### A/B 测试

```python
def ab_test(prompt_a: str, prompt_b: str, test_cases: list[str]):
    """A/B 测试不同 prompt"""

    results_a = []
    results_b = []

    for test_case in test_cases:
        # 使用 prompt_a
        chain_a = ChatPromptTemplate.from_template(prompt_a) | llm | parser
        result_a = chain_a.invoke({"input": test_case})
        results_a.append(result_a)

        # 使用 prompt_b
        chain_b = ChatPromptTemplate.from_template(prompt_b) | llm | parser
        result_b = chain_b.invoke({"input": test_case})
        results_b.append(result_b)

    # 比较结果
    for a, b in zip(results_a, results_b):
        # 记录长度、关键词等指标
        print(f"A: {len(a)} chars, B: {len(b)} chars")

    return results_a, results_b
```

## 22.9 测试工具

### pytest 配置

```python
# pytest.ini 或 conftest.py

import pytest
from langchain_openai import ChatOpenAI

# 全局测试配置
@pytest.fixture(scope="session")
def mock_llm():
    """Mock LLM fixture"""
    return MockChatModel({"default": "测试响应"})

@pytest.fixture(scope="session")
def real_llm():
    """真实 LLM fixture（需要 API Key）"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("需要 OPENAI_API_KEY")
    return ChatOpenAI(model="gpt-4o-mini")

# 测试用例
def test_with_mock(mock_llm):
    response = mock_llm.invoke("测试")
    assert response.content

def test_with_real(real_llm):
    response = real_llm.invoke("你好")
    assert response.content
```

### 测试数据管理

```python
# test_data.py

TEST_CASES = {
    "translation": [
        {"input": "你好", "expected_keywords": ["hello", "hi"]},
        {"input": "谢谢", "expected_keywords": ["thank", "thanks"]},
    ],
    "qa": [
        {"question": "什么是 AI？", "expected_keywords": ["智能", "计算机"]},
        {"question": "什么是 RAG？", "expected_keywords": ["检索", "生成"]},
    ],
}

def test_translation_cases(mock_llm):
    for case in TEST_CASES["translation"]:
        result = mock_llm.invoke(case["input"])
        assert any(kw in result.content.lower() for kw in case["expected_keywords"])
```

## 22.10 测试最佳实践

### 实践1：分层测试

```python
# 单元测试：快速、大量
def test_parser_unit():
    parser = StrOutputParser()
    assert parser.invoke(AIMessage(content="test")) == "test"

# 集成测试：适量
@pytest.mark.integration
def test_chain_integration():
    chain = prompt | mock_llm | parser
    result = chain.invoke({"topic": "test"})
    assert result

# E2E 测试：少量
@pytest.mark.e2e
@pytest.mark.slow
def test_full_flow_e2e():
    # 使用真实组件
    result = full_pipeline.invoke({"input": "test"})
    assert result
```

### 实践2：隔离不稳定因素

```python
def test_agent_logic():
    """测试 Agent 逻辑，不测试 LLM 决策"""
    # Mock LLM 返回固定的工具调用
    mock_llm = SequenceMockModel([
        AIMessage(tool_calls=[{"name": "weather", "args": {"city": "北京"}}]),
        AIMessage(content="北京晴天"),
    ])

    # 测试：流程是否正确执行
    # 不测试：LLM 是否能正确决定调用工具
    agent = create_react_agent(mock_llm, [mock_weather_tool])

    result = agent.invoke({"messages": [("user", "天气")]})
    assert "晴天" in result["messages"][-1].content
```

### 实践3：使用确定性测试

```python
def test_deterministic():
    """确定性测试"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 多次调用应该返回相似结果
    results = llm.batch(["1+1等于几？"] * 3)

    # 检查一致性
    for r in results:
        assert "2" in r.content
```

### 实践4：失败重试

```python
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_flaky_llm():
    """可能失败的测试，自动重试"""
    result = llm.invoke("复杂问题")
    assert len(result.content) > 50
```

## 22.11 测试清单

| 测试类型 | 内容 | 运行频率 |
|---------|------|---------|
| Mock LLM 测试 | 验证逻辑流程 | 每次 commit |
| 单元测试 | Prompt、Parser、Chain | 每次 commit |
| Agent 流程测试 | 工具调用流程 | 每天 |
| 集成测试 | 真实组件组合 | 每天 |
| E2E 测试 | 完整用户流程 | 每周 |
| A/B 测试 | Prompt 效果对比 | 需要时 |
| 回答质量评估 | LLM 输出质量 | 定期 |

## 22.12 本章小结

- LLM 测试难点：输出不确定、依赖外部服务、评估标准模糊
- Mock LLM：避免 API 成本，测试稳定可靠
- 单元测试：Prompt、Parser、Chain 的逻辑验证
- Agent 测试：Mock 工具 + Mock LLM 验证流程
- RAG 测试：Mock 检索器验证链路
- 模糊匹配比精确匹配更适合 LLM 输出
- 可用 LLM 评估 LLM 输出质量
- 分层测试：单元（大量）→ 集成（适量）→ E2E（少量）
- 隔离不稳定因素，测试逻辑而非 LLM 决策能力