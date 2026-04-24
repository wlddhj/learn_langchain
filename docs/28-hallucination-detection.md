# 第28章：幻觉检测与处理

## 28.1 什么是 LLM 幻觉

LLM 幻觉是指模型生成看似合理但实际不正确、不存在或与事实不符的内容。

### 幻觉类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **事实幻觉** | 编造不存在的事实 | "李白出生于 2020 年" |
| **来源幻觉** | 错误引用来源 | "根据《2024年报告》..."（报告不存在） |
| **逻辑幻觉** | 推理链条错误 | "因为 A，所以 B"（实际上 A 不导致 B） |
| **数量幻觉** | 错误的数量信息 | "全球人口约 100 亿"（实际约 80 亿） |
| **时间幻觉** | 时间信息错误 | "iPhone 15 发布于 2010 年" |
| **细节幻觉** | 细节描述错误 | 虚构人物细节、事件细节 |

### 幻觉产生原因

```
1. 数据问题
   - 训练数据不完整或过时
   - 数据偏见或噪声

2. 模型问题
   - 模型知识边界模糊
   - 模型过度自信
   - 生成策略（温度过高）

3. Prompt 问题
   - 问题超出模型知识范围
   - 诱导性问题
   - 缺乏上下文约束

4. RAG 问题
   - 检索内容不相关
   - 检索内容不足
   - 模型忽略检索内容
```

## 28.2 幻觉检测方法

### 自我检测法

让模型自我检查生成内容的准确性：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 生成回答
generate_prompt = ChatPromptTemplate.from_template("""
请回答以下问题：
{question}

请提供详细答案，包括相关的事实、日期、数据等。
""")

# 自我检测
verify_prompt = ChatPromptTemplate.from_template("""
请检查以下回答中是否存在幻觉（不正确或虚构的信息）：

问题：{question}
回答：{answer}

请按以下格式输出检测结果：
1. 列出回答中可能存在问题的具体陈述
2. 对每个问题陈述，指出其可能不正确的原因
3. 给出整体可信度评分（高/中/低）

如果有确定的事实错误，请提供正确信息。
""")

# 使用 Chain
def generate_with_verification(question: str):
    # 生成答案
    answer = (generate_prompt | llm).invoke({"question": question}).content
    
    # 自我检测
    verification = (verify_prompt | llm).invoke({
        "question": question,
        "answer": answer
    }).content
    
    return {
        "answer": answer,
        "verification": verification,
    }

# 测试
result = generate_with_verification("李白出生于哪一年？")
print("回答:", result["answer"])
print("\n检测结果:", result["verification"])
```

### 事实核对法

使用外部知识源验证关键事实：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, List

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class FactCheckRequest(BaseModel):
    claims: list[str]  # 待验证的事实陈述
    confidence: float  # 整体置信度

class FactCheckResult(BaseModel):
    claim: str
    is_verifiable: bool
    verified: bool
    correct_info: str | None
    source: str | None
    notes: str

# 提取关键事实
extract_prompt = ChatPromptTemplate.from_template("""
从以下回答中提取需要验证的关键事实陈述（如日期、数量、人名、事件等）：

回答：{answer}

{format_instructions}
""")

parser = PydanticOutputParser(pydantic_object=FactCheckRequest)

def extract_claims(answer: str) -> FactCheckRequest:
    chain = extract_prompt | llm | parser
    return chain.invoke({
        "answer": answer,
        "format_instructions": parser.get_format_instructions(),
    })

# 使用 Wikipedia 或其他知识源验证
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def verify_claims(claims: list[str]) -> list[FactCheckResult]:
    results = []
    for claim in claims:
        try:
            # 从 Wikipedia 搜索相关信息
            search_query = claim.split()[0:5]  # 提取关键词
            wiki_result = wikipedia.run(" ".join(search_query))
            
            # 使用 LLM 判断是否验证
            verify_prompt = ChatPromptTemplate.from_template("""
            基于以下参考信息，验证这个陈述是否正确：
            
            陈述：{claim}
            参考信息：{reference}
            
            请判断：
            1. 这个陈述是否可验证？
            2. 如果可验证，是否正确？
            3. 如果不正确，正确信息是什么？
            """)
            
            verification = (verify_prompt | llm).invoke({
                "claim": claim,
                "reference": wiki_result,
            }).content
            
            results.append(FactCheckResult(
                claim=claim,
                is_verifiable=True,
                verified="正确" in verification,
                correct_info=verification if "不正确" in verification else None,
                source="Wikipedia",
                notes=verification,
            ))
        except Exception as e:
            results.append(FactCheckResult(
                claim=claim,
                is_verifiable=False,
                verified=False,
                correct_info=None,
                source=None,
                notes=f"无法验证: {str(e)}",
            ))
    
    return results

# 完整流程
def fact_check_pipeline(answer: str):
    # 提取关键事实
    claims = extract_claims(answer)
    
    # 验证每个事实
    results = verify_claims(claims.claims)
    
    return {
        "original_answer": answer,
        "claims_count": len(claims.claims),
        "verification_results": results,
        "confidence": claims.confidence,
    }
```

### RAG 源引用检测

确保回答基于检索内容，避免无根据的扩展：

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# 创建向量库（假设已有数据）
vectorstore = Chroma(
    persist_directory="./knowledge_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG 回答
rag_prompt = ChatPromptTemplate.from_template("""
基于以下参考资料回答问题。如果参考资料中没有相关信息，请明确说明"参考资料中未找到相关信息"，不要编造内容。

参考资料：
{context}

问题：{question}

请：
1. 首先判断参考资料是否足以回答问题
2. 如果不足以回答，请说明缺少什么信息
3. 如果可以回答，请引用具体来源（标注来源编号）
4. 不要添加参考资料中不存在的信息

回答格式：
- 来源充足性：[充足/部分充足/不足]
- 回答内容：[基于引用的回答]
- 引用来源：[引用的具体段落编号]
""")

def format_docs_with_source(docs):
    """格式化文档并保留来源编号"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[来源 {i}] {doc.page_content}\n来源文件: {doc.metadata.get('source', '未知')}")
    return "\n\n".join(formatted)

def rag_with_source_tracking(question: str):
    # 检索
    docs = retriever.invoke(question)
    context = format_docs_with_source(docs)
    
    # 生成回答
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question,
    })
    
    return {
        "question": question,
        "retrieved_docs": docs,
        "context": context,
        "answer": answer,
    }

# 检测是否使用了检索内容
def check_source_usage(result: dict) -> dict:
    """检查回答是否基于检索内容"""
    answer = result["answer"]
    docs = result["retrieved_docs"]
    
    check_prompt = ChatPromptTemplate.from_template("""
    检查以下回答是否正确引用了检索内容：
    
    回答：{answer}
    
    检索内容：
    {context}
    
    请评估：
    1. 回答中的信息是否都来自检索内容？
    2. 是否有添加检索内容之外的"新"信息？
    3. 引用标注是否准确？
    
    评估结果：
    - 源覆盖率：[高/中/低]（回答内容多少来自检索）
    - 幻觉风险：[高/中/低]（存在编造内容的风险）
    - 具体问题：[列出可能的问题]
    """)
    
    check_chain = check_prompt | llm | StrOutputParser()
    check_result = check_chain.invoke({
        "answer": answer,
        "context": format_docs_with_source(docs),
    })
    
    return {
        **result,
        "source_check": check_result,
    }
```

### 交叉验证法

使用多个模型交叉验证答案：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# 使用多个模型
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0),
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0),
    "claude": ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0),
}

prompt = ChatPromptTemplate.from_template("""
请回答以下问题，提供准确的事实信息：
{question}

注意：
- 只提供你有确信的信息
- 不确定的信息请标注"不确定"
- 提供事实时请尽量精确
""")

def multi_model_verify(question: str):
    """使用多个模型验证"""
    results = {}
    
    for name, model in models.items():
        answer = (prompt | model).invoke({"question": question}).content
        results[name] = answer
    
    # 比较各模型答案
    compare_prompt = ChatPromptTemplate.from_template("""
    比较以下多个模型对同一问题的回答，找出共识和分歧：
    
    问题：{question}
    
    GPT-4o-mini 回答：{gpt_mini}
    GPT-4o 回答：{gpt_4o}
    Claude 回答：{claude}
    
    请分析：
    1. 各模型共识的部分（所有模型一致的内容）
    2. 各模型分歧的部分（不一致的内容）
    3. 哪些内容可能是幻觉（某个模型独有的"事实"）
    4. 最终推荐答案（基于共识）
    """)
    
    comparison = (compare_prompt | models["gpt-4o"]).invoke({
        "question": question,
        "gpt_mini": results["gpt-4o-mini"],
        "gpt_4o": results["gpt-4o"],
        "claude": results["claude"],
    }).content
    
    return {
        "question": question,
        "model_answers": results,
        "comparison": comparison,
    }
```

## 28.3 幻觉检测工具

### 使用 LangSmith 追踪

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "hallucination_detection"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

client = Client()
llm = ChatOpenAI(model="gpt-4o-mini")

# 创建检测链
detect_prompt = ChatPromptTemplate.from_template("""
分析以下回答是否存在幻觉风险：

问题：{question}
回答：{answer}

幻觉风险指标：
1. 知识边界风险：问题是否超出模型知识范围
2. 细节过度风险：是否提供了过多无法验证的细节
3. 时间敏感风险：回答是否涉及可能过时的信息
4. 来源引用风险：是否引用了可能不存在的内容

请给出：
- 风险等级：[高/中/低]
- 具体风险点
- 改进建议
""")

def detect_hallucination(question: str, answer: str):
    chain = detect_prompt | llm
    result = chain.invoke({
        "question": question,
        "answer": answer,
    })
    
    # 使用 LangSmith 记录
    run_id = result.response_metadata.get("run_id")
    
    return {
        "detection_result": result.content,
        "run_id": run_id,
    }

# 后续可从 LangSmith 分析历史运行
def analyze_hallucination_history(project_name: str):
    """分析项目中的幻觉检测历史"""
    runs = client.list_runs(project_name=project_name)
    
    high_risk_count = 0
    for run in runs:
        if "高" in run.outputs.get("detection_result", ""):
            high_risk_count += 1
    
    return {
        "total_runs": len(list(runs)),
        "high_risk_count": high_risk_count,
        "risk_rate": high_risk_count / len(list(runs)) if runs else 0,
    }
```

### 使用第三方检测工具

```python
# 使用 Hunor（幻觉检测库）
# pip install hunor

from hunor import HallucinationDetector

detector = HallucinationDetector(model="gpt-4o-mini")

def detect_with_hunor(question: str, answer: str):
    result = detector.detect(
        question=question,
        answer=answer,
    )
    
    return {
        "hallucination_score": result.score,  # 0-1，越高越可能幻觉
        "detected_issues": result.issues,
        "confidence": result.confidence,
    }

# 使用 FAVA（事实验证）
# pip install fava

from fava import FactVerifier

verifier = FactVerifier()

def verify_with_fava(claim: str):
    result = verifier.verify(claim)
    
    return {
        "claim": claim,
        "verified": result.is_true,
        "evidence": result.evidence,
        "source": result.source,
    }
```

## 28.4 幻觉预防策略

### Prompt 约束法

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 安全 Prompt 设计
safe_prompt = ChatPromptTemplate.from_template("""
请回答以下问题：

{question}

回答要求：
1. **知识边界声明**：如果问题超出你的知识范围，请明确说明"我无法确定"
2. **不确定性标注**：对于不确定的信息，使用"可能"、"据记载"等词
3. **避免过度细节**：不要编造无法验证的具体数字、日期、姓名
4. **来源谨慎**：不要引用可能不存在的具体报告、文章
5. **时间敏感**：如果信息可能已过时，请标注"截至我的知识截止日期"

请按以下格式回答：
- 确认部分：[你确信的信息]
- 不确定部分：[你不确定的信息，标注原因]
- 缺失部分：[你无法回答的部分]
""")

chain = safe_prompt | llm

def safe_answer(question: str):
    return chain.invoke({"question": question}).content

# 测试
print(safe_answer("李白的具体出生日期是哪一天？"))
print(safe_answer("2024年全球GDP是多少？"))
```

### 检索增强法（强约束 RAG）

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 强约束 RAG Prompt
strict_rag_prompt = ChatPromptTemplate.from_template("""
**严格基于以下参考资料回答问题**

参考资料：
{context}

问题：{question}

规则：
1. 只使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，必须回答："参考资料中未找到相关信息，无法回答"
3. 不要添加任何参考资料之外的知识
4. 不要猜测或推断
5. 引用时标注具体来源段落

回答模板：
```
基于来源 [编号] 的信息：
[引用的具体内容]

是否完全回答问题：[是/部分/否]
缺少的信息：[如果有]
```
""")

def strict_rag_answer(question: str, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    
    chain = strict_rag_prompt | llm
    return chain.invoke({
        "context": context,
        "question": question,
    }).content
```

### 分层验证法

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class LayeredAnswer(BaseModel):
    high_confidence: str  # 高置信度内容（可验证的事实）
    medium_confidence: str  # 中置信度内容（常见知识）
    low_confidence: str  # 低置信度内容（推断或不确定）
    unanswerable: str  # 无法回答的部分

layered_prompt = ChatPromptTemplate.from_template("""
请分层回答以下问题，按置信度分类：

问题：{question}

请将回答分为四层：
1. **高置信度**：可通过权威来源验证的事实（如百科全书、官方数据）
2. **中置信度**：广泛认知但难以精确验证的内容
3. **低置信度**：推断、估计、可能不准确的内容
4. **无法回答**：超出知识范围或无可靠信息的内容

每层内容请详细说明，如果某层没有内容请填写"无"。
""")

def layered_answer(question: str) -> LayeredAnswer:
    from langchain_core.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=LayeredAnswer)
    
    prompt = ChatPromptTemplate.from_template("""
    {template}
    
    {format_instructions}
    """)
    
    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
    
    return chain.invoke({
        "template": layered_prompt.template,
        "question": question,
    })
```

### 温度控制法

```python
from langchain_openai import ChatOpenAI

# 事实性问题使用低温度
factual_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创意性问题可使用较高温度
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def answer_by_type(question: str, question_type: str):
    """根据问题类型选择温度"""
    if question_type == "factual":
        # 事实性问题 - 低温度，减少幻觉
        return factual_llm.invoke(question).content
    elif question_type == "creative":
        # 创意性问题 - 可接受更高温度
        return creative_llm.invoke(question).content
    else:
        # 默认使用低温度
        return factual_llm.invoke(question).content

# 问题类型判断
def classify_question(question: str):
    classify_prompt = ChatPromptTemplate.from_template("""
    判断以下问题的类型：
    
    问题：{question}
    
    类型：
    - factual：事实性问题（需要准确事实）
    - creative：创意性问题（可接受创意回答）
    - opinion：观点性问题（可接受主观观点）
    
    只输出类型名称。
    """)
    
    return (classify_prompt | factual_llm).invoke({"question": question}).content
```

## 28.5 幻觉处理流程

### 完整检测处理 Pipeline

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class HallucinationReport(BaseModel):
    risk_level: str  # high/medium/low
    issues: list[str]
    confidence: float  # 0-1
    verified_claims: list[str]
    unverified_claims: list[str]
    recommendations: list[str]

class ProcessedAnswer(BaseModel):
    original_answer: str
    verified_answer: str
    removed_content: list[str]
    added_caveats: list[str]
    hallucination_report: HallucinationReport

class HallucinationPipeline:
    def __init__(self, llm, knowledge_source=None):
        self.llm = llm
        self.knowledge_source = knowledge_source
    
    def detect(self, question: str, answer: str) -> HallucinationReport:
        """检测幻觉"""
        detect_prompt = ChatPromptTemplate.from_template("""
        分析以下回答的幻觉风险：
        
        问题：{question}
        回答：{answer}
        
        请评估：
        1. 风险等级（high/medium/low）
        2. 具体问题列表（哪些陈述可能有问题）
        3. 置信度评分（0-1）
        4. 可验证的事实（有可靠来源的内容）
        5. 不可验证的陈述（可能虚构的内容）
        6. 改进建议
        
        输出 JSON 格式。
        """)
        
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=HallucinationReport)
        
        chain = detect_prompt | self.llm | parser
        return chain.invoke({"question": question, "answer": answer})
    
    def process(self, question: str, answer: str, report: HallucinationReport) -> ProcessedAnswer:
        """处理幻觉内容"""
        if report.risk_level == "low":
            return ProcessedAnswer(
                original_answer=answer,
                verified_answer=answer,
                removed_content=[],
                added_caveats=[],
                hallucination_report=report,
            )
        
        process_prompt = ChatPromptTemplate.from_template("""
        原回答存在幻觉风险，请修正：
        
        原回答：{answer}
        问题点：{issues}
        不可验证内容：{unverified}
        
        请：
        1. 移除可能虚构的内容
        2. 对不确定内容添加警示（如"据记载"、"可能"）
        3. 标注无法验证的部分
        
        输出修正后的回答。
        """)
        
        chain = process_prompt | self.llm
        
        verified_answer = chain.invoke({
            "answer": answer,
            "issues": report.issues,
            "unverified": report.unverified_claims,
        }).content
        
        return ProcessedAnswer(
            original_answer=answer,
            verified_answer=verified_answer,
            removed_content=report.unverified_claims,
            added_caveats=["添加了不确定性警示"],
            hallucination_report=report,
        )
    
    def run(self, question: str, answer: str) -> ProcessedAnswer:
        """完整流程"""
        # 检测
        report = self.detect(question, answer)
        
        # 处理
        processed = self.process(question, answer, report)
        
        return processed

# 使用
pipeline = HallucinationPipeline(llm)

question = "李白的具体出生日期是什么？"
answer = "李白出生于公元 701 年 2 月 28 日，在四川绵州。"

result = pipeline.run(question, answer)
print("原始回答:", result.original_answer)
print("修正回答:", result.verified_answer)
print("风险等级:", result.hallucination_report.risk_level)
print("移除内容:", result.removed_content)
```

### 增量验证法

```python
def incremental_generate_with_verify(question: str, llm):
    """增量生成并验证每个部分"""
    
    # 分解问题
    decompose_prompt = ChatPromptTemplate.from_template("""
    将以下问题分解为几个可独立回答和验证的子问题：
    
    问题：{question}
    
    输出子问题列表。
    """)
    
    sub_questions = (decompose_prompt | llm).invoke({"question": question}).content
    
    # 对每个子问题回答并验证
    results = []
    for sub_q in sub_questions.split("\n"):
        if sub_q.strip():
            # 回答
            answer = llm.invoke(sub_q).content
            
            # 验证
            verify_prompt = ChatPromptTemplate.from_template("""
            验证以下回答是否可靠：
            
            问题：{question}
            回答：{answer}
            
            可靠性：[高/中/低]
            原因：[简短说明]
            """)
            
            verification = (verify_prompt | llm).invoke({
                "question": sub_q,
                "answer": answer,
            }).content
            
            results.append({
                "sub_question": sub_q,
                "answer": answer,
                "verification": verification,
            })
    
    # 合并可靠内容
    reliable_parts = []
    unreliable_parts = []
    
    for r in results:
        if "高" in r["verification"]:
            reliable_parts.append(r["answer"])
        else:
            unreliable_parts.append({
                "question": r["sub_question"],
                "answer": r["answer"],
                "reason": r["verification"],
            })
    
    # 生成最终回答
    final_prompt = ChatPromptTemplate.from_template("""
    基于以下验证过的内容，生成最终回答：
    
    可靠内容：{reliable}
    不可靠内容（需标注）：{unreliable}
    
    请：
    1. 首先呈现可靠内容
    2. 对不可靠内容添加警示标注
    """)
    
    final_answer = (final_prompt | llm).invoke({
        "reliable": "\n".join(reliable_parts),
        "unreliable": str(unreliable_parts),
    }).content
    
    return {
        "question": question,
        "sub_questions": results,
        "final_answer": final_answer,
        "reliable_count": len(reliable_parts),
        "unreliable_count": len(unreliable_parts),
    }
```

## 28.6 实际应用场景

### 智能问答系统

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

class SafeQASystem:
    """安全的问答系统"""
    
    def __init__(self, knowledge_db_path: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        self.vectorstore = Chroma(
            persist_directory=knowledge_db_path,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        self.pipeline = HallucinationPipeline(self.llm)
    
    def answer(self, question: str) -> dict:
        # 1. 检索
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. 生成（带约束）
        prompt = ChatPromptTemplate.from_template("""
        基于以下参考资料回答问题。如果资料中没有相关信息，请明确说明。
        
        参考资料：
        {context}
        
        问题：{question}
        
        要求：
        - 只使用资料中的信息
        - 标注来源
        - 缺少信息时明确说明
        """)
        
        initial_answer = (prompt | self.llm).invoke({
            "context": context,
            "question": question,
        }).content
        
        # 3. 幻觉检测
        report = self.pipeline.detect(question, initial_answer)
        
        # 4. 处理
        processed = self.pipeline.process(question, initial_answer, report)
        
        return {
            "question": question,
            "answer": processed.verified_answer,
            "sources": [doc.metadata for doc in docs],
            "hallucination_report": report,
            "confidence": self._calculate_confidence(report),
        }
    
    def _calculate_confidence(self, report: HallucinationReport) -> float:
        """计算整体置信度"""
        if report.risk_level == "low":
            return 0.9
        elif report.risk_level == "medium":
            return 0.6
        else:
            return 0.3

# 使用
qa_system = SafeQASystem("./knowledge_db")
result = qa_system.answer("公司的退货政策是什么？")
print("回答:", result["answer"])
print("置信度:", result["confidence"])
```

### Agent 工具调用验证

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_database(query: str) -> str:
    """搜索数据库"""
    # 实际实现
    return "搜索结果..."

@tool
def calculate(expression: str) -> str:
    """计算"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 创建 Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    tools=[search_database, calculate],
)

def verify_agent_response(result: dict) -> dict:
    """验证 Agent 响应"""
    messages = result["messages"]
    
    # 检查工具调用是否合理
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                })
    
    # 使用 LLM 评估工具调用合理性
    verify_prompt = ChatPromptTemplate.from_template("""
    评估以下 Agent 工具调用是否合理：
    
    工具调用记录：{tool_calls}
    最终回答：{final_answer}
    用户问题：{question}
    
    请评估：
    1. 工具调用是否必要
    2. 参数是否正确
    3. 最终回答是否基于工具结果
    4. 是否存在幻觉风险
    
    输出评估结果。
    """)
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    verification = (verify_prompt | llm).invoke({
        "tool_calls": tool_calls,
        "final_answer": messages[-1].content,
        "question": messages[0].content,
    }).content
    
    return {
        "original_result": result,
        "tool_calls": tool_calls,
        "verification": verification,
    }
```

## 28.7 幻觉检测最佳实践

### 检测策略选择

| 场景 | 推荐策略 |
|------|---------|
| **事实查询** | RAG 源引用检测 + 事实核对 |
| **知识问答** | 自我检测 + 分层验证 |
| **Agent 任务** | 工具调用验证 + 结果检查 |
| **开放对话** | 温度控制 + Prompt 约束 |
| **高风险场景** | 多模型交叉验证 |

### 处理流程建议

```python
# 推荐的处理流程
def recommended_workflow(question: str, context: str = None):
    """
    推荐的幻觉处理流程：
    1. 预防：使用安全 Prompt、低温度
    2. 检测：自我检测 + 源引用检查
    3. 处理：移除不可靠内容、添加警示
    4. 监控：记录幻觉事件、持续改进
    """
    
    # Step 1: 预防
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Step 2: 生成（带约束）
    if context:
        # RAG 模式
        prompt = ChatPromptTemplate.from_template("""
        严格基于以下内容回答：
        {context}
        
        问题：{question}
        
        规则：只使用给定内容，不添加额外信息。
        """)
        answer = (prompt | llm).invoke({"context": context, "question": question}).content
    else:
        # 普通模式（带约束）
        prompt = ChatPromptTemplate.from_template("""
        回答问题：{question}
        
        要求：
        - 标注不确定内容
        - 不要编造具体事实
        """)
        answer = (prompt | llm).invoke({"question": question}).content
    
    # Step 3: 检测
    pipeline = HallucinationPipeline(llm)
    report = pipeline.detect(question, answer)
    
    # Step 4: 处理
    if report.risk_level != "low":
        processed = pipeline.process(question, answer, report)
        answer = processed.verified_answer
    
    # Step 5: 监控（记录）
    log_hallucination_event(question, answer, report)
    
    return {
        "answer": answer,
        "hallucination_report": report,
    }

def log_hallucination_event(question, answer, report):
    """记录幻觉事件"""
    import json
    from datetime import datetime
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "risk_level": report.risk_level,
        "issues": report.issues,
    }
    
    with open("hallucination_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

## 28.8 幻觉检测评估

### 评估指标

```python
class HallucinationMetrics:
    """幻觉检测评估指标"""
    
    def __init__(self):
        self.total_answers = 0
        self.high_risk_count = 0
        self.medium_risk_count = 0
        self.low_risk_count = 0
        self.corrected_count = 0
    
    def record(self, report: HallucinationReport, was_corrected: bool):
        self.total_answers += 1
        
        if report.risk_level == "high":
            self.high_risk_count += 1
        elif report.risk_level == "medium":
            self.medium_risk_count += 1
        else:
            self.low_risk_count += 1
        
        if was_corrected:
            self.corrected_count += 1
    
    def get_summary(self) -> dict:
        return {
            "total_answers": self.total_answers,
            "high_risk_rate": self.high_risk_count / self.total_answers,
            "medium_risk_rate": self.medium_risk_count / self.total_answers,
            "low_risk_rate": self.low_risk_count / self.total_answers,
            "correction_rate": self.corrected_count / self.total_answers,
        }
```

## 28.9 本章小结

- 幻觉类型：事实、来源、逻辑、数量、时间、细节幻觉
- 幻觉原因：数据、模型、Prompt、RAG 问题
- 检测方法：自我检测、事实核对、RAG 源引用、交叉验证
- 检测工具：LangSmith 追踪、第三方检测库（Hunor、FAVA）
- 预防策略：Prompt 约束、强约束 RAG、分层验证、温度控制
- 处理流程：检测 → 评估 → 修正 → 监控
- 实际应用：安全问答系统、Agent 工具验证
- 最佳实践：预防为主、检测为辅、持续监控