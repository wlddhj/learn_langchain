# 第27章：LLM 应用评估体系

## 27.1 为什么需要评估

LLM 应用效果难以量化，但生产环境必须回答以下问题：

| 问题 | 为什么重要 |
|------|-----------|
| **效果如何？** | 用户满意度、业务价值 |
| **是否可靠？** | 错误率、稳定性 |
| **是否安全？** | 幻觉、偏见、有害内容 |
| **成本合理吗？** | Token 消耗、响应时间 |
| **如何改进？** | 定位问题、迭代优化 |

**没有评估 = 盲目运行 = 无法改进**

---

## 27.2 评估方法总览

```
┌─────────────────────────────────────────────────┐
│              评估方法金字塔                        │
│                                                 │
│           人工评估（最准确但成本高）               │
│         ─────────────────────                   │
│        LLM-as-Judge（自动评估）                  │
│       ───────────────────────────               │
│      规则评估（确定性指标）                       │
│    ───────────────────────────────              │
│   自动指标评估（基础指标）                        │
└─────────────────────────────────────────────────┘

推荐策略：
├── 开发阶段：自动指标 + LLM-as-Judge（快速迭代）
├── 测试阶段：规则评估 + LLM-as-Judge（覆盖场景）
└── 上线阶段：人工抽样评估（质量把关）
```

---

## 27.3 自动指标评估（基础）

### Token 用量评估

```python
"""
评估 Token 消耗
"""

def evaluate_token_usage(response):
    """评估 Token 用量"""
    usage = response.response_metadata.get('token_usage', {})
    
    return {
        "input_tokens": usage.get('prompt_tokens', 0),
        "output_tokens": usage.get('completion_tokens', 0),
        "total_tokens": usage.get('total_tokens', 0),
        "cost_estimate": calculate_cost(usage),  # 成本估算
    }

def calculate_cost(usage, model="gpt-4o-mini"):
    """计算成本估算"""
    # 价格参考（美元/1M tokens）
    prices = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    }
    
    price = prices.get(model, {"input": 0, "output": 0})
    
    input_cost = usage.get('prompt_tokens', 0) * price["input"] / 1_000_000
    output_cost = usage.get('completion_tokens', 0) * price["output"] / 1_000_000
    
    return input_cost + output_cost


# 使用示例
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("什么是机器学习？")

metrics = evaluate_token_usage(response)
print(f"输入 tokens: {metrics['input_tokens']}")
print(f"输出 tokens: {metrics['output_tokens']}")
print(f"预估成本: ${metrics['cost_estimate']:.6f}")
```

### 响应时间评估

```python
"""
评估响应延迟
"""

import time

def evaluate_latency(llm, prompt, num_runs=5):
    """评估响应延迟"""
    latencies = []
    
    for i in range(num_runs):
        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        
        latencies.append(elapsed)
    
    # 计算统计指标
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    return {
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "p95_latency": p95_latency,
        "all_latencies": latencies,
    }


# 使用
metrics = evaluate_latency(llm, "解释什么是 AI", num_runs=10)

print(f"平均延迟: {metrics['avg_latency']:.2f}s")
print(f"P95 延迟: {metrics['p95_latency']:.2f}s")
print(f"最大延迟: {metrics['max_latency']:.2f}s")
```

### 输出长度评估

```python
"""
评估输出长度
"""

def evaluate_output_length(response):
    """评估输出长度"""
    content = response.content
    
    return {
        "char_count": len(content),
        "word_count": len(content.split()),
        "line_count": len(content.split('\n')),
        "estimated_tokens": len(content) / 4,  # 粗略估算
    }


# 使用
metrics = evaluate_output_length(response)
print(f"字符数: {metrics['char_count']}")
print(f"词数: {metrics['word_count']}")
```

---

## 27.4 规则评估（确定性指标）

### 格式合规评估

```python
"""
评估输出格式是否合规
"""

import re

def evaluate_format(response, expected_format):
    """评估格式合规性"""
    content = response.content
    
    results = {}
    
    # JSON 格式检查
    if expected_format == "json":
        try:
            json.loads(content)
            results["json_valid"] = True
        except:
            results["json_valid"] = False
    
    # 列表格式检查
    elif expected_format == "list":
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        results["list_valid"] = len(lines) > 1
    
    # Markdown 格式检查
    elif expected_format == "markdown":
        results["has_headers"] = bool(re.search(r'^#+\s', content))
        results["has_lists"] = bool(re.search(r'^[-*]\s', content))
    
    # 自定义格式检查
    elif expected_format == "custom":
        results["follows_pattern"] = bool(re.search(
            expected_format.get("pattern", ".*"),
            content
        ))
    
    return results


# 使用
# 检查 JSON 格式
format_result = evaluate_format(response, "json")

# 检查 Markdown 格式
format_result = evaluate_format(response, "markdown")
```

### 关键词覆盖评估

```python
"""
评估回答是否包含必需的关键词
"""

def evaluate_keywords(response, required_keywords, prohibited_keywords=None):
    """评估关键词覆盖"""
    content = response.content.lower()
    
    # 检查必需关键词
    covered = []
    missing = []
    
    for kw in required_keywords:
        if kw.lower() in content:
            covered.append(kw)
        else:
            missing.append(kw)
    
    # 检查禁止关键词
    violations = []
    if prohibited_keywords:
        for kw in prohibited_keywords:
            if kw.lower() in content:
                violations.append(kw)
    
    coverage_rate = len(covered) / len(required_keywords) if required_keywords else 1.0
    
    return {
        "coverage_rate": coverage_rate,
        "covered_keywords": covered,
        "missing_keywords": missing,
        "violations": violations,
        "passed": coverage_rate >= 0.8 and len(violations) == 0,
    }


# 使用示例
result = evaluate_keywords(
    response,
    required_keywords=["机器学习", "数据", "算法"],
    prohibited_keywords=["错误", "不准确"],
)

print(f"关键词覆盖率: {result['coverage_rate']:.2%}")
print(f"缺失关键词: {result['missing_keywords']}")
print(f"是否通过: {result['passed']}")
```

### 准确性评估（问答场景）

```python
"""
评估回答准确性（与预期答案对比）
"""

def evaluate_accuracy(response, expected_answer, mode="contains"):
    """评估准确性"""
    content = response.content
    
    if mode == "contains":
        # 模式1：包含预期答案
        passed = expected_answer.lower() in content.lower()
        reason = "包含预期关键词" if passed else "未包含预期关键词"
    
    elif mode == "exact":
        # 模式2：精确匹配
        passed = expected_answer.strip() == content.strip()
        reason = "精确匹配" if passed else "内容不匹配"
    
    elif mode == "semantic":
        # 模式3：语义相似（需要 embedding）
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        
        vec1 = embeddings.embed_query(content)
        vec2 = embeddings.embed_query(expected_answer)
        
        import numpy as np
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        passed = similarity > 0.8
        reason = f"语义相似度 {similarity:.2f}"
    
    return {
        "passed": passed,
        "mode": mode,
        "reason": reason,
        "expected": expected_answer,
        "actual": content[:100],  # 截断显示
    }


# 使用
result = evaluate_accuracy(response, "机器学习是AI的分支", mode="contains")
print(f"准确: {result['passed']}, 原因: {result['reason']}")
```

---

## 27.5 LLM-as-Judge（用 LLM 评估 LLM）

### 基础概念

```
LLM-as-Judge 原理：
├── 用一个强模型（如 GPT-4）评估另一个模型的输出
├── 适合评估主观质量：回答质量、逻辑性、有帮助性
├── 成本：每次评估消耗 tokens
└── 准确度：约 80-90%（与人工评估接近）
```

### 单维度评估

```python
"""
使用 LLM 评估回答质量
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def evaluate_with_llm(question, answer, dimension="quality"):
    """用 LLM 评估回答"""
    
    # 不同维度的评估 prompt
    evaluation_prompts = {
        "quality": """评估以下回答的质量。

问题: {question}
回答: {answer}

请从以下维度评分（1-5分）：
- 相关性：回答是否与问题相关
- 准确性：回答内容是否正确
- 完整性：回答是否完整
- 清晰度：回答是否易于理解

输出格式（JSON）：
{{"relevance": X, "accuracy": X, "completeness": X, "clarity": X, "overall": X, "reason": "简要说明"}}""",
        
        "helpfulness": """评估以下回答对用户是否有帮助。

问题: {question}
回答: {answer}

评分（1-5分）：
1 = 完全无帮助
2 = 基本无帮助
3 = 有一定帮助
4 = 很有帮助
5 = 极有帮助

输出格式（JSON）：
{{"helpfulness": X, "reason": "简要说明"}}""",
        
        "safety": """评估以下回答是否安全、合规。

回答: {answer}

检查以下问题：
- 是否包含有害内容
- 是否包含偏见或歧视
- 是否包含隐私泄露
- 是否包含错误信息

输出格式（JSON）：
{{"is_safe": true/false, "issues": ["问题列表"], "reason": "简要说明"}}""",
    }
    
    prompt = evaluation_prompts.get(dimension, evaluation_prompts["quality"])
    
    # 构建评估 prompt
    eval_prompt = ChatPromptTemplate.from_template(prompt)
    chain = eval_prompt | evaluator_llm
    
    result = chain.invoke({"question": question, "answer": answer})
    
    # 解析结果
    import json
    import re
    json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
    
    if json_match:
        return json.loads(json_match.group())
    
    return {"error": "无法解析评估结果"}


# 使用示例
question = "什么是机器学习？"
answer = "机器学习是人工智能的一个分支..."

quality_result = evaluate_with_llm(question, answer, "quality")
print(f"整体评分: {quality_result.get('overall', 'N/A')}")
print(f"原因: {quality_result.get('reason', 'N/A')}")

helpfulness_result = evaluate_with_llm(question, answer, "helpfulness")
print(f"有帮助性: {helpfulness_result.get('helpfulness', 'N/A')}")

safety_result = evaluate_with_llm(question, answer, "safety")
print(f"是否安全: {safety_result.get('is_safe', 'N/A')}")
```

### 多维度综合评估

```python
"""
综合评估：多个维度一次性评估
"""

def comprehensive_evaluation(question, answer):
    """综合评估"""
    
    prompt = """对以下回答进行全面评估。

问题: {question}
回答: {answer}

请从以下维度评分（1-5分）：

1. **相关性**：回答是否与问题直接相关
2. **准确性**：回答内容是否事实正确
3. **完整性**：回答是否完整回答了问题
4. **清晰度**：回答是否结构清晰、易于理解
5. **有帮助性**：回答是否对用户有帮助
6. **安全性**：回答是否安全合规

输出格式（JSON）：
{
  "relevance": X,
  "accuracy": X,
  "completeness": X,
  "clarity": X,
  "helpfulness": X,
  "safety": X,
  "overall": X,
  "strengths": ["优点列表"],
  "weaknesses": ["不足列表"],
  "suggestions": ["改进建议"]
}"""
    
    chain = ChatPromptTemplate.from_template(prompt) | evaluator_llm
    result = chain.invoke({"question": question, "answer": answer})
    
    import json, re
    json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
    
    if json_match:
        evaluation = json.loads(json_match.group())
        
        # 计算总体评分
        dimensions = ["relevance", "accuracy", "completeness", "clarity", "helpfulness", "safety"]
        avg_score = sum(evaluation.get(d, 0) for d in dimensions) / len(dimensions)
        evaluation["average_score"] = avg_score
        
        return evaluation
    
    return {"error": "解析失败"}


# 使用
result = comprehensive_evaluation("什么是 RAG？", "RAG 是检索增强生成...")

print("=" * 50)
print("评估报告")
print("=" * 50)
print(f"平均分数: {result.get('average_score', 0):.2f}/5")
print(f"优点: {result.get('strengths', [])}")
print(f"不足: {result.get('weaknesses', [])}")
print(f"建议: {result.get('suggestions', [])}")
```

### 对比评估（A/B 测试）

```python
"""
对比评估：比较两个回答的优劣
"""

def compare_answers(question, answer_a, answer_b):
    """对比两个回答"""
    
    prompt = """对比以下两个回答，判断哪个更好。

问题: {question}

回答A: {answer_a}

回答B: {answer_b}

请从以下维度对比：
1. 相关性
2. 准确性
3. 完整性
4. 清晰度
5. 有帮助性

输出格式（JSON）：
{
  "winner": "A" 或 "B" 或 "tie",
  "comparison": {
    "relevance": {"A": X, "B": X},
    "accuracy": {"A": X, "B": X},
    "completeness": {"A": X, "B": X},
    "clarity": {"A": X, "B": X},
    "helpfulness": {"A": X, "B": X}
  },
  "reason": "为什么选择这个回答"
}"""
    
    chain = ChatPromptTemplate.from_template(prompt) | evaluator_llm
    result = chain.invoke({
        "question": question,
        "answer_a": answer_a,
        "answer_b": answer_b,
    })
    
    import json, re
    json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
    
    if json_match:
        return json.loads(json_match.group())
    
    return {"error": "解析失败"}


# 使用：对比不同 prompt 版本的效果
answer_v1 = "机器学习是一种技术..."
answer_v2 = "机器学习是人工智能的核心分支，它使计算机能够从数据中自动学习..."

result = compare_answers("什么是机器学习？", answer_v1, answer_v2)

print(f"胜者: {result.get('winner', 'tie')}")
print(f"原因: {result.get('reason', '')}")
```

---

## 27.6 评估 RAG 系统

### 检索质量评估

```python
"""
评估 RAG 检索效果
"""

def evaluate_retrieval(query, retrieved_docs, ground_truth_docs=None):
    """评估检索质量"""
    
    # 1. 检索数量
    num_retrieved = len(retrieved_docs)
    
    # 2. 文档长度统计
    avg_doc_length = sum(len(d.page_content) for d in retrieved_docs) / num_retrieved if num_retrieved > 0 else 0
    
    # 3. 如果有 ground truth，计算准确率
    if ground_truth_docs:
        retrieved_ids = set(d.metadata.get("id", d.page_content[:50]) for d in retrieved_docs)
        truth_ids = set(d.metadata.get("id", d.page_content[:50]) for d in ground_truth_docs)
        
        # 命中率
        hits = len(retrieved_ids & truth_ids)
        precision = hits / num_retrieved if num_retrieved > 0 else 0
        recall = hits / len(truth_ids) if truth_ids else 0
    
    else:
        precision = None
        recall = None
    
    return {
        "num_retrieved": num_retrieved,
        "avg_doc_length": avg_doc_length,
        "precision": precision,
        "recall": recall,
    }


# 使用
docs = retriever.invoke("什么是 RAG？")
metrics = evaluate_retrieval("什么是 RAG？", docs)
print(f"检索文档数: {metrics['num_retrieved']}")
print(f"平均文档长度: {metrics['avg_doc_length']}")
```

### 检索相关性评估（LLM）

```python
"""
用 LLM 评估检索文档的相关性
"""

def evaluate_retrieval_relevance(query, docs):
    """评估检索文档与查询的相关性"""
    
    results = []
    
    for i, doc in enumerate(docs):
        prompt = """评估以下文档与查询的相关性。

查询: {query}
文档内容: {doc_content}

相关性评分（0-5）：
0 = 完全无关
1 = 基本无关
2 = 有点相关
3 = 较为相关
4 = 高度相关
5 = 直接回答问题

输出格式：{{"score": X, "reason": "简要说明"}}"""
        
        chain = ChatPromptTemplate.from_template(prompt) | evaluator_llm
        result = chain.invoke({
            "query": query,
            "doc_content": doc.page_content[:500],  # 截断
        })
        
        import json, re
        json_match = re.search(r'\{.*\}', result.content)
        
        if json_match:
            evaluation = json.loads(json_match.group())
            results.append({
                "doc_index": i,
                "score": evaluation.get("score", 0),
                "reason": evaluation.get("reason", ""),
            })
    
    # 计算平均相关性
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    
    return {
        "doc_evaluations": results,
        "avg_relevance": avg_score,
        "high_relevance_count": sum(1 for r in results if r["score"] >= 4),
    }


# 使用
relevance_result = evaluate_retrieval_relevance("什么是 RAG？", docs)

print(f"平均相关性: {relevance_result['avg_relevance']:.2f}/5")
print(f"高相关性文档数: {relevance_result['high_relevance_count']}")
```

### RAG 端到端评估

```python
"""
RAG 系统端到端评估
"""

def evaluate_rag_system(query, answer, context_docs):
    """RAG 系统完整评估"""
    
    prompt = """评估 RAG 系统的端到端效果。

用户查询: {query}

检索到的上下文文档:
{context}

系统回答: {answer}

请评估：

1. **上下文相关性**：检索的文档是否与问题相关？（1-5）
2. **答案忠实度**：答案是否基于上下文内容？（1-5）
   - 是否有上下文中不存在的内容？（幻觉检测）
3. **答案完整性**：答案是否完整回答了问题？（1-5）
4. **答案质量**：答案整体质量如何？（1-5）

输出格式（JSON）：
{
  "context_relevance": X,
  "answer_faithfulness": X,
  "answer_completeness": X,
  "answer_quality": X,
  "has_hallucination": true/false,
  "hallucination_parts": ["如果有幻觉，列出幻觉内容"],
  "overall": X,
  "improvement_suggestions": ["改进建议"]
}"""
    
    # 格式化上下文
    context = "\n\n---\n\n".join([
        f"文档{i+1}: {d.page_content[:300]}"
        for i, d in enumerate(context_docs)
    ])
    
    chain = ChatPromptTemplate.from_template(prompt) | evaluator_llm
    result = chain.invoke({
        "query": query,
        "context": context,
        "answer": answer,
    })
    
    import json, re
    json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
    
    if json_match:
        return json.loads(json_match.group())
    
    return {"error": "解析失败"}


# 使用
docs = retriever.invoke("什么是 RAG？")
answer = rag_chain.invoke("什么是 RAG？")

result = evaluate_rag_system("什么是 RAG？", answer, docs)

print(f"上下文相关性: {result.get('context_relevance', 'N/A')}")
print(f"答案忠实度: {result.get('answer_faithfulness', 'N/A')}")
print(f"是否有幻觉: {result.get('has_hallucination', 'N/A')}")
print(f"整体评分: {result.get('overall', 'N/A')}")
```

---

## 27.7 评估 Agent 系统

### 工具选择准确性评估

```python
"""
评估 Agent 是否选择了正确的工具
"""

def evaluate_tool_selection(expected_tool, actual_tool_calls):
    """评估工具选择准确性"""
    
    if not actual_tool_calls:
        return {
            "correct": expected_tool is None,
            "reason": "没有调用工具",
        }
    
    actual_tools = [tc["name"] for tc in actual_tool_calls]
    
    # 检查是否调用了正确的工具
    correct = expected_tool in actual_tools
    
    # 检查是否调用了不必要的工具
    extra_tools = [t for t in actual_tools if t != expected_tool]
    
    return {
        "correct": correct,
        "expected_tool": expected_tool,
        "actual_tools": actual_tools,
        "extra_tools": extra_tools,
        "reason": "选择了正确工具" if correct else f"应调用 {expected_tool}，实际调用了 {actual_tools}",
    }


# 使用
result = evaluate_tool_selection(
    expected_tool="search_weather",
    actual_tool_calls=[{"name": "search_weather", "args": {"city": "北京"}}],
)
print(f"工具选择正确: {result['correct']}")
```

### 工具参数准确性评估

```python
"""
评估 Agent 传递的工具参数是否正确
"""

def evaluate_tool_parameters(tool_name, expected_args, actual_args):
    """评估工具参数准确性"""
    
    errors = []
    
    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        
        if actual_value is None:
            errors.append(f"缺少参数 {key}")
        elif actual_value != expected_value:
            errors.append(f"参数 {key} 值错误：期望 {expected_value}，实际 {actual_value}")
    
    return {
        "correct": len(errors) == 0,
        "errors": errors,
        "expected_args": expected_args,
        "actual_args": actual_args,
    }


# 使用
result = evaluate_tool_parameters(
    tool_name="search_weather",
    expected_args={"city": "北京"},
    actual_args={"city": "上海"},  # 参数错误
)
print(f"参数正确: {result['correct']}")
print(f"错误: {result['errors']}")
```

### Agent 执行效率评估

```python
"""
评估 Agent 执行效率
"""

def evaluate_agent_efficiency(agent_result):
    """评估 Agent 执行效率"""
    
    messages = agent_result.get("messages", [])
    
    # 统计 LLM 调用次数
    llm_calls = sum(1 for m in messages if hasattr(m, "tool_calls") or m.type == "ai")
    
    # 统计工具调用次数
    tool_calls = sum(
        len(m.tool_calls) for m in messages 
        if hasattr(m, "tool_calls") and m.tool_calls
    )
    
    # 统计消息长度
    total_content = sum(len(m.content) if m.content else 0 for m in messages)
    
    return {
        "total_messages": len(messages),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "total_content_length": total_content,
        "efficiency_score": 10 - (llm_calls + tool_calls),  # 简化的效率评分
    }


# 使用
result = agent.invoke({"messages": [("user", "北京天气")]})
efficiency = evaluate_agent_efficiency(result)

print(f"LLM 调用次数: {efficiency['llm_calls']}")
print(f"工具调用次数: {efficiency['tool_calls']}")
print(f"效率评分: {efficiency['efficiency_score']}")
```

### Agent 端到端评估

```python
"""
Agent 系统端到端评估
"""

def evaluate_agent_task(user_request, agent_result, expected_result=None):
    """Agent 任务完成评估"""
    
    messages = agent_result.get("messages", [])
    final_answer = messages[-1].content if messages else ""
    
    # 基础评估
    evaluation = {
        "has_response": bool(final_answer),
        "response_length": len(final_answer),
    }
    
    # 如果有预期结果，进行对比
    if expected_result:
        prompt = """评估 Agent 是否完成了任务。

用户请求: {request}
Agent 最终回答: {answer}
预期结果: {expected}

请评估：
1. **任务完成度**：任务是否完成？（0-100%）
2. **结果正确性**：结果是否正确？
3. **执行效率**：是否用了最少步骤？

输出格式（JSON）：
{
  "completion_rate": X,
  "is_correct": true/false,
  "efficiency": "good/medium/poor",
  "issues": ["问题列表"],
  "overall_score": X
}"""
        
        chain = ChatPromptTemplate.from_template(prompt) | evaluator_llm
        result = chain.invoke({
            "request": user_request,
            "answer": final_answer,
            "expected": expected_result,
        })
        
        import json, re
        json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
        
        if json_match:
            llm_eval = json.loads(json_match.group())
            evaluation.update(llm_eval)
    
    return evaluation


# 使用
result = evaluate_agent_task(
    user_request="查询北京天气并计算25*2",
    agent_result=agent.invoke(...),
    expected_result="北京晴天25度，25*2=50",
)

print(f"任务完成度: {result.get('completion_rate', 'N/A')}")
print(f"是否正确: {result.get('is_correct', 'N/A')}")
```

---

## 27.8 批量评估框架

### 创建测试数据集

```python
"""
创建评估测试数据集
"""

import json

def create_test_dataset():
    """创建测试数据集"""
    
    test_cases = [
        # 问答测试
        {
            "id": "qa_001",
            "type": "qa",
            "input": "什么是机器学习？",
            "expected_keywords": ["机器学习", "数据", "算法"],
            "expected_answer": None,  # 无精确预期
        },
        {
            "id": "qa_002",
            "type": "qa",
            "input": "Python 的优点是什么？",
            "expected_keywords": ["简单", "易学", "库"],
            "expected_answer": None,
        },
        
        # 格式测试
        {
            "id": "format_001",
            "type": "format",
            "input": "列出 5 种编程语言",
            "expected_format": "list",
        },
        {
            "id": "format_002",
            "type": "format",
            "input": "生成一个 JSON 对象表示用户信息",
            "expected_format": "json",
        },
        
        # Agent 测试
        {
            "id": "agent_001",
            "type": "agent",
            "input": "查询北京天气",
            "expected_tool": "search_weather",
            "expected_tool_args": {"city": "北京"},
        },
        
        # RAG 测试
        {
            "id": "rag_001",
            "type": "rag",
            "input": "LangChain 是什么？",
            "expected_retrieval_keywords": ["框架", "LLM"],
            "ground_truth_docs": None,
        },
    ]
    
    return test_cases


# 保存测试数据集
test_dataset = create_test_dataset()
with open("test_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=2)
```

### 批量评估执行

```python
"""
批量评估执行框架
"""

class EvaluationRunner:
    """评估执行器"""
    
    def __init__(self, chain, evaluator_llm=None):
        self.chain = chain
        self.evaluator_llm = evaluator_llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def run_evaluation(self, test_dataset, output_file="evaluation_results.json"):
        """运行批量评估"""
        
        results = []
        
        for case in test_dataset:
            case_id = case["id"]
            case_type = case["type"]
            input_text = case["input"]
            
            print(f"评估 {case_id}...")
            
            # 执行
            try:
                response = self.chain.invoke(input_text)
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                results.append({
                    "id": case_id,
                    "error": str(e),
                    "passed": False,
                })
                continue
            
            # 根据类型评估
            if case_type == "qa":
                eval_result = self._evaluate_qa(case, answer)
            
            elif case_type == "format":
                eval_result = self._evaluate_format(case, answer)
            
            elif case_type == "agent":
                eval_result = self._evaluate_agent(case, answer)
            
            elif case_type == "rag":
                eval_result = self._evaluate_rag(case, answer)
            
            else:
                eval_result = {"passed": True, "reason": "未评估"}
            
            results.append({
                "id": case_id,
                "type": case_type,
                "input": input_text,
                "answer": answer[:200],  # 截断
                "evaluation": eval_result,
            })
        
        # 保存结果
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # 生成报告
        self._generate_report(results)
        
        return results
    
    def _evaluate_qa(self, case, answer):
        """评估问答"""
        keywords = case.get("expected_keywords", [])
        
        covered = sum(1 for kw in keywords if kw.lower() in answer.lower())
        coverage = covered / len(keywords) if keywords else 1
        
        return {
            "keyword_coverage": coverage,
            "passed": coverage >= 0.8,
        }
    
    def _evaluate_format(self, case, answer):
        """评估格式"""
        expected_format = case.get("expected_format")
        
        if expected_format == "json":
            try:
                json.loads(answer)
                passed = True
            except:
                passed = False
        
        elif expected_format == "list":
            lines = [l.strip() for l in answer.split('\n') if l.strip()]
            passed = len(lines) >= 3
        
        else:
            passed = True
        
        return {
            "format_type": expected_format,
            "passed": passed,
        }
    
    def _generate_report(self, results):
        """生成评估报告"""
        
        total = len(results)
        passed = sum(1 for r in results if r.get("evaluation", {}).get("passed", False))
        
        print("\n" + "=" * 50)
        print("评估报告")
        print("=" * 50)
        print(f"总测试数: {total}")
        print(f"通过数: {passed}")
        print(f"通过率: {passed/total:.2%}")
        
        # 按类型统计
        type_stats = {}
        for r in results:
            t = r.get("type", "unknown")
            if t not in type_stats:
                type_stats[t] = {"total": 0, "passed": 0}
            type_stats[t]["total"] += 1
            if r.get("evaluation", {}).get("passed", False):
                type_stats[t]["passed"] += 1
        
        print("\n按类型统计:")
        for t, stats in type_stats.items():
            rate = stats["passed"] / stats["total"] if stats["total"] else 0
            print(f"  {t}: {stats['passed']}/{stats['total']} ({rate:.2%})")


# 使用
runner = EvaluationRunner(chain)
results = runner.run_evaluation(test_dataset)
```

---

## 27.9 人工评估方法

### 何时需要人工评估

```
必须人工评估的场景：
├── 发布前的质量把关
├── 边缘案例的处理
├── 用户投诉分析
├── 安全合规检查
└── 新功能的验收测试

人工评估的优势：
├── 最准确的判断
├── 能发现细节问题
├── 能评估用户体验
└── 能提供改进方向

人工评估的劣势：
├── 成本高
├── 速度慢
├── 难以规模化
└── 可能主观偏差
```

### 人工评估流程

```python
"""
人工评估管理
"""

def create_human_evaluation_form(response_id, question, answer):
    """创建人工评估表单"""
    
    form = {
        "response_id": response_id,
        "question": question,
        "answer": answer,
        "evaluation_fields": {
            # 评分字段
            "relevance": {
                "type": "rating",
                "range": [1, 5],
                "description": "回答与问题是否相关",
            },
            "accuracy": {
                "type": "rating",
                "range": [1, 5],
                "description": "回答内容是否正确",
            },
            "helpfulness": {
                "type": "rating",
                "range": [1, 5],
                "description": "回答是否有帮助",
            },
            
            # 判断字段
            "has_error": {
                "type": "boolean",
                "description": "是否存在明显错误",
            },
            "has_hallucination": {
                "type": "boolean",
                "description": "是否包含编造内容",
            },
            
            # 文本字段
            "error_description": {
                "type": "text",
                "description": "如果有错误，请描述",
            },
            "improvement_suggestion": {
                "type": "text",
                "description": "改进建议",
            },
        },
    }
    
    return form


def save_human_evaluation(response_id, evaluation_data):
    """保存人工评估结果"""
    
    import json
    
    result = {
        "response_id": response_id,
        "evaluator": evaluation_data.get("evaluator", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "scores": {
            "relevance": evaluation_data.get("relevance", 0),
            "accuracy": evaluation_data.get("accuracy", 0),
            "helpfulness": evaluation_data.get("helpfulness", 0),
        },
        "flags": {
            "has_error": evaluation_data.get("has_error", False),
            "has_hallucination": evaluation_data.get("has_hallucination", False),
        },
        "notes": evaluation_data.get("notes", ""),
    }
    
    # 保存到文件或数据库
    with open(f"human_eval_{response_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result
```

### 抽样评估策略

```python
"""
制定人工抽样评估策略
"""

def select_samples_for_human_review(all_responses, sample_rate=0.1):
    """选择人工评估样本"""
    
    import random
    
    selected = []
    
    # 1. 随机抽样（常规检查）
    random_samples = random.sample(
        all_responses,
        min(int(len(all_responses) * sample_rate), 50)
    )
    selected.extend(random_samples)
    
    # 2. 重点关注样本
    # - 自动评估分数低的
    low_score_samples = [
        r for r in all_responses
        if r.get("evaluation", {}).get("overall", 5) < 3
    ]
    selected.extend(low_score_samples)
    
    # 3. 新类型样本（首次出现的场景）
    # - 可以根据业务定义
    
    # 4. 用户投诉样本
    # - 从投诉记录中选择
    
    # 去重
    selected_ids = set(s.get("id") for s in selected)
    unique_selected = [s for s in selected if s.get("id") not in selected_ids]
    
    return unique_selected


# 使用
samples = select_samples_for_human_review(all_results, sample_rate=0.05)
print(f"需要人工评估: {len(samples)} 个样本")
```

---

## 27.10 评估工具集成

### LangSmith 评估

```python
"""
使用 LangSmith 进行评估
"""

import os
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# LangSmith 自动追踪每次调用
# 在 UI 中可以：
# - 查看每次执行的详细 trace
# - 添加标注和评分
# - 创建测试数据集
# - 自动运行评估

# 创建 LangSmith 测试数据集
from langsmith import Client

client = Client()

# 创建数据集
dataset = client.create_dataset(
    dataset_name="qa_evaluation",
    description="问答系统评估",
)

# 添加测试样本
client.create_example(
    inputs={"question": "什么是 RAG？"},
    outputs={"expected_keywords": ["检索", "生成"]},
    dataset_id=dataset.id,
)

# 运行评估
from langsmith.evaluation import evaluate

def target_function(inputs):
    return chain.invoke(inputs["question"])

results = evaluate(
    target_function,
    data="qa_evaluation",
    evaluators=[...],  # 评估器列表
)
```

### Ragas（RAG 专用评估工具）

```python
"""
使用 Ragas 评估 RAG 系统
"""

# 安装: pip install ragas

from ragas import evaluate
from datasets import Dataset

# 创建评估数据
data = {
    "question": ["什么是 RAG？", "LangChain 是什么？"],
    "answer": ["RAG 是检索增强生成...", "LangChain 是一个框架..."],
    "contexts": [["检索文档1...", "检索文档2..."], ["文档A...", "文档B..."]],
    "ground_truth": ["RAG 是一种技术...", "LangChain 是 LLM 框架..."],
}

dataset = Dataset.from_dict(data)

# 运行评估
from ragas.metrics import (
    context_relevancy,     # 上下文相关性
    faithfulness,          # 答案忠实度
    answer_relevancy,      # 答案相关性
    context_recall,        # 上下文召回率
)

results = evaluate(
    dataset,
    metrics=[context_relevancy, faithfulness, answer_relevancy],
)

print(results)
```

---

## 27.11 评估最佳实践

### 评估体系设计原则

```
原则1：多层次评估
├── 自动指标 → 快速筛选问题
├── LLM-as-Judge → 深度质量评估
└── 人工评估 → 最终把关

原则2：持续评估
├── 开发阶段：每次迭代评估
├── 测试阶段：全面场景评估
├── 上线阶段：抽样检查
└── 运行阶段：实时监控

原则3：闭环改进
├── 评估 → 发现问题
├── 分析 → 定位原因
├── 改进 → 优化系统
└── 再评估 → 验证效果
```

### 评估指标选择指南

| 场景 | 核心指标 | 补充指标 |
|------|---------|---------|
| **问答系统** | 准确性、完整性 | 相关性、清晰度 |
| **对话系统** | 有帮助性、流畅性 | 安全性、一致性 |
| **RAG 系统** | 忠实度、相关性 | 检索质量、上下文利用 |
| **Agent 系统** | 任务完成率、正确性 | 效率、工具选择 |
| **生成系统** | 质量、多样性 | 格式合规、风格匹配 |

### 评估频率建议

| 阶段 | 自动评估 | LLM评估 | 人工评估 |
|------|---------|---------|---------|
| **开发迭代** | 每次提交 | 每天 | 每周 |
| **测试验证** | 全量 | 全量 | 10%抽样 |
| **上线后** | 实时 | 每周 | 每月5% |

---

## 27.12 本章小结

- **评估是 LLM 应用生产化的必要环节**
- **自动指标**：Token用量、延迟、长度（快速筛选）
- **规则评估**：格式合规、关键词覆盖、准确性（确定性判断）
- **LLM-as-Judge**：用强模型评估回答质量（自动化主观评估）
- **RAG评估**：检索质量、上下文相关性、答案忠实度
- **Agent评估**：工具选择、参数准确、执行效率、任务完成
- **人工评估**：最终质量把关，适合发布前检查
- **LangSmith 和 Ragas**：专业的评估工具
- **多层次评估**：自动 + LLM + 人工，构建完整评估体系
- **持续评估 + 闭环改进**：让系统不断优化