"""
第12章 Demo 1：API 调用工具

演示 HTTP 请求工具、REST API 工具、搜索工具的开发。
可独立运行。
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 1. HTTP 请求工具
# ============================================================

@tool
def http_get(url: str) -> str:
    """发送 GET 请求获取指定 URL 的内容。
    当需要获取网页内容、API 数据时使用。
    url: 完整的 URL 地址，如 'https://httpbin.org/get'"""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            if len(content) > 3000:
                content = content[:3000] + "\n... (内容已截断)"
            return content
    except Exception as e:
        return f"请求失败: {type(e).__name__} - {e}"


@tool
def http_post(url: str, data: str = "{}") -> str:
    """发送 POST 请求到指定 URL。
    当需要提交数据到 API 时使用。
    url: 目标 URL
    data: JSON 格式的请求数据"""
    try:
        import urllib.request
        body = data.encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            if len(content) > 3000:
                content = content[:3000] + "\n... (内容已截断)"
            return content
    except Exception as e:
        return f"请求失败: {type(e).__name__} - {e}"


# ============================================================
# 2. GitHub API 工具
# ============================================================

@tool
def query_github_repo(repo_name: str, info_type: str = "info") -> str:
    """查询 GitHub 仓库信息。
    当需要了解开源项目的基本信息、Issues、README 时使用。
    repo_name: 仓库名称，格式 owner/repo，如 'langchain-ai/langchain'
    info_type: 查询类型 - info(基本信息)/issues(问题列表)"""
    try:
        import urllib.request
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Mozilla/5.0"}
        base_url = f"https://api.github.com/repos/{repo_name}"

        if info_type == "info":
            req = urllib.request.Request(base_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            return (f"仓库: {data.get('full_name', 'N/A')}\n"
                    f"描述: {data.get('description', '无')}\n"
                    f"Stars: {data.get('stargazers_count', 0)}\n"
                    f"Forks: {data.get('forks_count', 0)}\n"
                    f"语言: {data.get('language', '未知')}\n"
                    f"许可证: {data.get('license', {}).get('name', '无') if data.get('license') else '无'}\n"
                    f"最近更新: {data.get('updated_at', 'N/A')}")

        elif info_type == "issues":
            req = urllib.request.Request(
                f"{base_url}/issues?per_page=5&state=open", headers=headers
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                issues = json.loads(resp.read().decode("utf-8"))

            result = f"仓库 {repo_name} 最近的 Issues:\n"
            for issue in issues[:5]:
                labels = ", ".join(l["name"] for l in issue.get("labels", []))
                result += f"  #{issue['number']} {issue['title']}"
                if labels:
                    result += f" [{labels}]"
                result += "\n"
            return result

        else:
            return f"不支持的查询类型: {info_type}，支持: info, issues"

    except Exception as e:
        return f"查询失败: {type(e).__name__} - {e}"


# ============================================================
# 3. 模拟搜索工具
# ============================================================

@tool
def search_web(query: str, num_results: int = 3) -> str:
    """搜索互联网获取信息。
    当需要查找最新资讯、技术文档、解决方案时使用。
    query: 搜索关键词
    num_results: 返回结果数量（1-5）"""
    # 模拟搜索结果（实际应用中可接入 DuckDuckGo、Bing 等）
    mock_results = {
        "Python": [
            {"title": "Python 官方文档", "url": "https://docs.python.org", "snippet": "Python 编程语言的官方文档，包含教程和API参考"},
            {"title": "Real Python", "url": "https://realpython.com", "snippet": "高质量的 Python 教程和实战指南"},
            {"title": "Python Wikipedia", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)", "snippet": "Python 编程语言的维基百科介绍"},
        ],
        "LangChain": [
            {"title": "LangChain 官方文档", "url": "https://python.langchain.com", "snippet": "LangChain 框架的官方文档和 API 参考"},
            {"title": "LangChain GitHub", "url": "https://github.com/langchain-ai/langchain", "snippet": "LangChain 开源项目，包含示例和教程"},
            {"title": "LangChain 教程", "url": "https://js.langchain.com/docs/tutorials", "snippet": "LangChain 入门教程和最佳实践"},
        ],
        "AI Agent": [
            {"title": "什么是 AI Agent", "url": "https://example.com/ai-agent-intro", "snippet": "AI Agent 的概念、原理和应用场景介绍"},
            {"title": "LangGraph Agent 教程", "url": "https://langchain-ai.github.io/langgraph", "snippet": "使用 LangGraph 构建 Agent 的官方教程"},
        ],
    }

    results = []
    query_lower = query.lower()
    for key, items in mock_results.items():
        if key.lower() in query_lower or any(
            key.lower() in query_lower for key in mock_results
        ):
            results.extend(items)

    if not results:
        # 对所有结果做模糊匹配
        for key, items in mock_results.items():
            for item in items:
                if query_lower in item["title"].lower() or query_lower in item["snippet"].lower():
                    results.append(item)

    if not results:
        return f"未找到 '{query}' 的相关结果。请尝试其他关键词。"

    output = f"搜索 '{query}' 的结果:\n\n"
    for i, r in enumerate(results[:num_results], 1):
        output += f"{i}. {r['title']}\n"
        output += f"   链接: {r['url']}\n"
        output += f"   摘要: {r['snippet']}\n\n"

    return output


# ============================================================
# Demo 1: 直接测试工具
# ============================================================

def demo_tool_testing():
    print("=" * 50)
    print("Demo 12-1 (1/3): 直接测试工具")
    print("=" * 50)

    # 测试 GitHub 工具
    print("--- GitHub 仓库查询 ---")
    result = query_github_repo.invoke({"repo_name": "langchain-ai/langchain", "info_type": "info"})
    print(result)
    print()

    # 测试搜索工具
    print("--- 搜索工具 ---")
    result = search_web.invoke({"query": "LangChain", "num_results": 3})
    print(result)

    # 测试 HTTP 工具
    print("--- HTTP GET ---")
    result = http_get.invoke({"url": "https://httpbin.org/get"})
    print(result[:200])
    print()


# ============================================================
# Demo 2: Agent 使用 API 工具
# ============================================================

def demo_api_agent():
    print("=" * 50)
    print(f"Demo 12-1 (2/3): Agent 使用 API 工具 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[query_github_repo, search_web, http_get],
        system_prompt="你是一个技术信息助手。帮助用户查询技术项目和搜索信息。用中文简洁回答。",
    )

    questions = [
        "帮我看看 langchain-ai/langchain 这个项目有多少 Stars？",
        "搜索一下 Python 的学习资源",
    ]

    for q in questions:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print(f"用户: {q}")
        print(f"AI:   {result['messages'][-1].content[:200]}")
        print()


# ============================================================
# Demo 3: 工具错误处理
# ============================================================

def demo_error_handling():
    print("=" * 50)
    print("Demo 12-1 (3/3): 工具错误处理")
    print("=" * 50)

    # 测试各种错误情况
    print("--- 404 URL ---")
    result = http_get.invoke({"url": "https://httpbin.org/status/404"})
    print(f"结果: {result[:100]}")
    print()

    print("--- 无效仓库 ---")
    result = query_github_repo.invoke({"repo_name": "nonexistent/repo12345"})
    print(f"结果: {result[:100]}")
    print()

    print("--- 无搜索结果 ---")
    result = search_web.invoke({"query": "xyzabc123nonexistent"})
    print(f"结果: {result[:100]}")
    print()

    print("结论: 所有工具都有完善的错误处理，不会因为异常导致崩溃")


if __name__ == "__main__":
    demo_tool_testing()
    demo_api_agent()
    demo_error_handling()

    print("=" * 50)
    print("Demo 12-1 完成!")
    print()
    print("API 工具开发要点:")
    print("  1. urllib/httpx    - 发送 HTTP 请求")
    print("  2. try/except      - 捕获网络异常，返回友好错误文本")
    print("  3. 超时设置        - 防止请求卡住")
    print("  4. 内容截断        - 限制返回长度，防止 token 溢出")
    print("  5. User-Agent      - 避免被网站拒绝")
    print("  6. 错误信息可读    - Agent 能理解错误并给出替代方案")
    print("=" * 50)
