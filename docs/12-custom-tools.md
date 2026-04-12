# 第12章：自定义工具开发

## 12.1 工具开发原则

好的工具应该：
- **单一职责**：每个工具做一件事
- **描述清晰**：让 LLM 准确理解何时使用
- **错误容忍**：返回错误文本而非抛异常
- **返回可读**：返回 LLM 能理解的文本

### 工具的安全边界

工具让 LLM 能与外部世界交互，但也带来安全风险：

| 风险类型 | 示例 | 防护措施 |
|---------|------|---------|
| SQL 注入 | `DROP TABLE users` | 只允许 SELECT，检查危险关键词 |
| 路径穿越 | `../../etc/passwd` | 限制工作目录，规范化路径 |
| 命令注入 | `; rm -rf /` | 过滤特殊字符，避免 `eval` |
| 数据泄露 | 查询敏感数据 | 限制可查表和字段 |
| API 滥用 | 频繁调用外部 API | 设置超时和速率限制 |

核心原则：**永远不要信任 LLM 生成的参数**，在工具内部做安全检查。

## 12.2 API 调用工具

### HTTP 请求工具

```python
import httpx
from langchain_core.tools import tool

@tool
def http_get(url: str) -> str:
    """发送 GET 请求获取指定 URL 的内容。
    当需要获取网页、API 数据时使用。
    url 必须是完整的 URL 地址。"""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            # 截断过长的响应
            content = response.text
            if len(content) > 3000:
                content = content[:3000] + "\n... (内容已截断)"
            return content
    except httpx.HTTPError as e:
        return f"请求失败: {e}"
    except Exception as e:
        return f"错误: {type(e).__name__} - {e}"
```

### REST API 工具（带认证）

```python
@tool
def query_github_repo(repo_name: str, info_type: str = "info") -> str:
    """查询 GitHub 仓库信息。
    repo_name: 仓库名称，格式为 owner/repo
    info_type: 查询类型 - info(基本信息)/issues(问题列表)/readme(README内容)"""
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}
        base_url = f"https://api.github.com/repos/{repo_name}"

        with httpx.Client(timeout=10.0, headers=headers) as client:
            if info_type == "info":
                resp = client.get(base_url)
                data = resp.json()
                return f"""仓库: {data['full_name']}
描述: {data.get('description', '无')}
Stars: {data['stargazers_count']}
语言: {data.get('language', '未知')}
许可证: {data.get('license', {}).get('name', '无') if data.get('license') else '无'}"""

            elif info_type == "issues":
                resp = client.get(f"{base_url}/issues?per_page=5&state=open")
                issues = resp.json()
                result = "最近的 Issues:\n"
                for issue in issues[:5]:
                    result += f"  #{issue['number']} {issue['title']}\n"
                return result

            elif info_type == "readme":
                resp = client.get(f"{base_url}/readme", headers={
                    **headers, "Accept": "application/vnd.github.raw"
                })
                content = resp.text
                return content[:2000] if len(content) > 2000 else content

            else:
                return f"不支持的查询类型: {info_type}"
    except Exception as e:
        return f"查询失败: {e}"
```

### 错误信息的设计

工具返回的错误信息不仅要给人看，更要给 LLM 看。好的错误信息帮助 LLM 自我修正：

```python
# 差的错误信息
return "Error"                    # LLM 不知道哪里错了

# 好的错误信息
return "错误：除数不能为零，请提供非零的除数"  # LLM 知道如何修正
return "错误：仅支持 SELECT 查询"              # LLM 知道限制
return "错误：请求超时，请稍后重试"            # LLM 知道可以重试
```

## 12.3 数据库查询工具

### SQLite 工具

```python
import sqlite3
from langchain_core.tools import tool

# 创建示例数据库
def setup_database():
    conn = sqlite3.connect("shop.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT, category TEXT, price REAL, stock INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER, quantity INTEGER,
            total_price REAL, status TEXT
        )
    """)
    conn.commit()
    conn.close()

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询并返回结果。
    仅支持 SELECT 查询，不支持修改操作。
    可用的表: products(商品表), orders(订单表)。
    products 表字段: id, name, category, price, stock
    orders 表字段: id, product_id, quantity, total_price, status"""
    # 安全检查
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "错误：仅支持 SELECT 查询"

    try:
        conn = sqlite3.connect("shop.db")
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "查询结果为空"

        # 格式化为表格
        result = " | ".join(columns) + "\n"
        result += "-" * len(result) + "\n"
        for row in rows[:20]:  # 限制返回行数
            result += " | ".join(str(v) for v in row) + "\n"

        return result
    except Exception as e:
        return f"查询失败: {e}"
```

## 12.4 文件操作工具

```python
import os
from langchain_core.tools import tool

@tool
def list_files(directory: str = ".") -> str:
    """列出指定目录下的文件和文件夹。
    directory: 目录路径，默认为当前目录"""
    try:
        items = os.listdir(directory)
        result = f"目录 '{directory}' 内容:\n"
        for item in sorted(items):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                result += f"  📁 {item}/\n"
            else:
                size = os.path.getsize(path)
                result += f"  📄 {item} ({size} bytes)\n"
        return result
    except FileNotFoundError:
        return f"目录不存在: {directory}"
    except Exception as e:
        return f"错误: {e}"

@tool
def read_file(file_path: str) -> str:
    """读取文件内容。
    file_path: 文件的完整路径"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 5000:
            content = content[:5000] + "\n... (文件内容已截断)"
        return content
    except FileNotFoundError:
        return f"文件不存在: {file_path}"
    except Exception as e:
        return f"读取失败: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """将内容写入文件。
    file_path: 文件路径
    content: 要写入的内容"""
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入 {len(content)} 字符到 {file_path}"
    except Exception as e:
        return f"写入失败: {e}"
```

## 12.5 搜索工具

```python
@tool
def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 搜索引擎搜索信息。
    query: 搜索关键词
    max_results: 最大结果数"""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "未找到相关结果"

        output = ""
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n"
            output += f"   链接: {r['href']}\n"
            output += f"   摘要: {r['body'][:150]}\n\n"

        return output
    except Exception as e:
        return f"搜索失败: {e}"
```

## 12.6 组合工具：数据处理管线

```python
from pydantic import BaseModel, Field
from typing import Optional

class DataAnalysisInput(BaseModel):
    file_path: str = Field(description="CSV 文件路径")
    operation: str = Field(description="操作类型: summary/correlation/filter")
    column: Optional[str] = Field(default=None, description="目标列名")
    condition: Optional[str] = Field(default=None, description="过滤条件")

@tool(args_schema=DataAnalysisInput)
def analyze_csv(
    file_path: str,
    operation: str,
    column: str = None,
    condition: str = None,
) -> str:
    """分析 CSV 数据文件。支持摘要统计、相关性分析、数据过滤。"""
    import pandas as pd

    try:
        df = pd.read_csv(file_path)

        if operation == "summary":
            return df.describe().to_string()

        elif operation == "correlation":
            return df.corr(numeric_only=True).to_string()

        elif operation == "filter":
            if not condition:
                return "错误：filter 操作需要提供 condition 参数"
            filtered = df.query(condition)
            return filtered.head(20).to_string()

        else:
            return f"不支持的操作: {operation}"

    except Exception as e:
        return f"分析失败: {e}"
```

## 12.7 构建完整的工具集

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 工具集合
all_tools = [
    # 信息获取
    duckduckgo_search,
    query_github_repo,
    http_get,

    # 数据处理
    query_database,
    analyze_csv,

    # 文件操作
    list_files,
    read_file,
    write_file,
]

# 创建 Agent
model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(
    model=model,
    tools=all_tools,
    prompt="""你是一个数据分析助手。你可以：
1. 搜索网络获取信息
2. 查询数据库
3. 分析 CSV 文件
4. 读写文件

请根据用户的需求选择合适的工具。回答使用中文。""",
)

# 使用
result = agent.invoke({
    "messages": [("user", "帮我分析当前目录下有哪些数据文件，并查看第一个文件的内容")]
})
```

## 12.8 工具开发清单

| 检查项 | 说明 |
|--------|------|
| 名称有意义 | `search_web` 而不是 `tool1` |
| 描述详细 | 包含用途、参数说明、使用场景 |
| 参数有类型注解 | 使用 Pydantic 定义 |
| 错误处理完善 | try/except 返回友好错误文本 |
| 返回值可读 | 格式化文本，非原始数据 |
| 截断保护 | 限制返回长度防止 token 溢出 |
| 安全检查 | 过滤危险操作（如非 SELECT SQL） |
| 超时设置 | 外部调用设置 timeout |
| 速率限制 | 防止频繁调用外部 API |

## 12.9 本章小结

- API 工具：封装 HTTP 请求，处理认证和错误
- 数据库工具：安全的 SQL 查询，限制只读操作
- 文件工具：读写文件，注意路径安全
- 搜索工具：集成搜索引擎获取实时信息
- 工具开发的关键：清晰的描述 + 完善的错误处理 + 可读的返回值
