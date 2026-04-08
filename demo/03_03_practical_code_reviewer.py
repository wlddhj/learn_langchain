"""
第3章 Demo 3：实战 —— 代码审查助手

综合运用 ChatPromptTemplate、变量注入、格式化输出，构建一个代码审查工具。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("GLM_API_KEY") or os.environ["GLM_API_KEY"].startswith("your-"):
    print("错误: 未设置 GLM_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")

llm = ChatOpenAI(model=GLM_MODEL, temperature=0, api_key=GLM_API_KEY, base_url=GLM_BASE_URL)


# 定义代码审查 prompt
review_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深的{language}代码审查专家。请按以下格式审查代码：

## 问题分析
逐一列出代码中的问题（如果没有问题，说明代码质量良好）

## 改进建议
给出具体的改进建议，包括性能、可读性、安全性等方面

## 评分
给出 1-10 分的评分和简短理由"""),
    ("human", "请审查以下{language}代码：\n```{language}\n{code}\n```"),
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个编程老师。用通俗易懂的语言解释代码的功能，适合初学者理解。"),
    ("human", "请解释以下{language}代码：\n```{language}\n{code}\n```"),
])


def review_code(language: str, code: str):
    """审查代码"""
    chain = review_prompt | llm | StrOutputParser()
    return chain.invoke({"language": language, "code": code})


def explain_code(language: str, code: str):
    """解释代码"""
    chain = explain_prompt | llm | StrOutputParser()
    return chain.invoke({"language": language, "code": code})


def main():
    print("=" * 50)
    print(f"代码审查助手 (模型: {GLM_MODEL})")
    print("=" * 50)

    # 测试代码 1：有问题的 Python 代码
    bad_python = """\
def calc(a,b):
    c=a+b
    d=c*2
    return d

result = calc("3", 5)
print(result)
"""

    print("=" * 50)
    print("示例 1: 审查有问题的 Python 代码")
    print("=" * 50)
    print(f"代码:\n{bad_python}")
    print()

    result = review_code("python", bad_python)
    print(result)
    print()

    # 测试代码 2：JavaScript 代码
    js_code = """\
async function fetchUserData(userId) {
  const response = await fetch('/api/users/' + userId);
  const data = await response.json();
  return data;
}
"""

    print("=" * 50)
    print("示例 2: 审查 JavaScript 代码")
    print("=" * 50)
    print(f"代码:\n{js_code}")
    print()

    result = review_code("javascript", js_code)
    print(result)
    print()

    # 测试代码 3：解释代码
    complex_code = """\
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

    print("=" * 50)
    print("示例 3: 解释 Python 代码")
    print("=" * 50)
    print(f"代码:\n{complex_code}")
    print()

    result = explain_code("python", complex_code)
    print(result)
    print()


if __name__ == "__main__":
    main()

    print("=" * 50)
    print("Demo 3-3 完成!")
    print()
    print("本 demo 综合运用了:")
    print("  - ChatPromptTemplate 多变量模板")
    print("  - StrOutputParser 提取纯文本")
    print("  - LCEL 管道: prompt | llm | parser")
    print("=" * 50)
