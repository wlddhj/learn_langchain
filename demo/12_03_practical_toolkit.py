"""
第12章 Demo 3：实战 —— 完整数据分析工具集

综合运用数据库、文件、计算工具，构建数据分析 Agent。
可独立运行。
"""

import os
import sys
import sqlite3
import tempfile
import shutil
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

WORK_DIR = tempfile.mkdtemp(prefix="toolkit_demo_")
DB_PATH = os.path.join(WORK_DIR, "analytics.db")


# ============================================================
# 初始化数据
# ============================================================

def setup_data():
    """创建完整的示例数据集"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 员工表
    c.execute("""CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT, department TEXT, position TEXT,
        salary REAL, hire_date TEXT, performance_score REAL
    )""")

    employees = [
        (1, "张三", "技术部", "高级工程师", 28000, "2021-03-15", 4.5),
        (2, "李四", "产品部", "产品经理", 25000, "2022-07-01", 4.2),
        (3, "王五", "设计部", "设计总监", 32000, "2019-11-20", 4.8),
        (4, "赵六", "技术部", "工程师", 22000, "2023-01-10", 3.9),
        (5, "孙七", "市场部", "市场经理", 26000, "2020-06-15", 4.1),
        (6, "周八", "技术部", "架构师", 38000, "2018-09-01", 4.7),
        (7, "吴九", "产品部", "初级产品", 18000, "2023-08-20", 3.6),
        (8, "郑十", "市场部", "运营专员", 15000, "2024-01-05", 3.8),
        (9, "钱十一", "设计部", "UI设计师", 20000, "2022-04-12", 4.0),
        (10, "陈十二", "技术部", "测试工程师", 19000, "2023-05-18", 3.7),
    ]
    c.executemany("INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?, ?, ?)", employees)

    # 项目表
    c.execute("""CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY,
        name TEXT, leader_id INTEGER, status TEXT,
        budget REAL, start_date TEXT, end_date TEXT
    )""")

    projects = [
        (1, "用户系统重构", 1, "进行中", 500000, "2024-01-01", "2024-06-30"),
        (2, "AI 助手开发", 6, "进行中", 800000, "2024-02-01", "2024-09-30"),
        (3, "移动端改版", 3, "已完成", 300000, "2023-10-01", "2024-01-31"),
        (4, "数据中台建设", 2, "规划中", 1200000, "2024-04-01", "2024-12-31"),
        (5, "品牌升级", 5, "进行中", 200000, "2024-01-15", "2024-05-31"),
    ]
    c.executemany("INSERT OR REPLACE INTO projects VALUES (?, ?, ?, ?, ?, ?, ?)", projects)

    conn.commit()
    conn.close()

    # 创建分析报告文件
    report = """公司年度分析概览
================
员工总数: 10人
部门分布: 技术部4人、产品部2人、设计部2人、市场部2人
平均薪资: ¥24,300
平均绩效: 4.13分

项目概况:
- 进行中: 3个
- 已完成: 1个
- 规划中: 1个
- 总预算: ¥3,000,000"""

    with open(os.path.join(WORK_DIR, "overview.txt"), "w", encoding="utf-8") as f:
        f.write(report)


# ============================================================
# 工具集定义
# ============================================================

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询，返回格式化的表格数据。
    仅支持 SELECT 查询。可用表:
    - employees(员工表): id, name, department, position, salary, hire_date, performance_score
    - projects(项目表): id, name, leader_id, status, budget, start_date, end_date
    sql: SELECT 语句"""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "错误: 仅支持 SELECT 查询。"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(sql.strip())
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "查询结果为空。"

        result = " | ".join(columns) + "\n"
        result += "-+-".join(["-" * len(c) for c in columns]) + "\n"
        for row in rows[:20]:
            result += " | ".join(str(v) for v in row) + "\n"
        result += f"(共 {len(rows)} 行)"
        return result

    except Exception as e:
        return f"查询失败: {e}"


@tool
def list_files() -> str:
    """列出工作目录中的所有文件"""
    try:
        items = os.listdir(WORK_DIR)
        if not items:
            return "工作目录为空。"
        result = "文件列表:\n"
        for item in sorted(items):
            path = os.path.join(WORK_DIR, item)
            size = os.path.getsize(path)
            result += f"  {item} ({size}B)\n"
        return result
    except Exception as e:
        return f"错误: {e}"


@tool
def read_file(file_path: str) -> str:
    """读取文件内容。file_path: 文件名"""
    full_path = os.path.join(WORK_DIR, file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()[:5000]
    except FileNotFoundError:
        return f"文件不存在: {file_path}"
    except Exception as e:
        return f"读取失败: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """将内容写入文件。file_path: 文件名, content: 内容"""
    full_path = os.path.join(WORK_DIR, file_path)
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {file_path} ({len(content)} 字符)"
    except Exception as e:
        return f"写入失败: {e}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。expression: 数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "不安全的表达式"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 创建分析 Agent
# ============================================================

def create_analyst_agent():
    """创建数据分析 Agent"""
    return create_agent(
        model=llm,
        tools=[query_database, list_files, read_file, write_file, calculate],
        system_prompt=f"""你是一个专业的数据分析助手「数智」。你可以：

1. 使用 query_database 查询数据库（仅 SELECT）
2. 使用 list_files 查看文件列表
3. 使用 read_file 读取文件
4. 使用 write_file 保存分析结果
5. 使用 calculate 做数学计算

数据分析原则:
- 先了解数据全貌，再进行深入分析
- 用 SQL 聚合函数统计数据
- 把分析结论保存到文件
- 回答简洁专业，数据准确
- 用中文回答""",
    )


# ============================================================
# Demo 1: 数据探索
# ============================================================

def demo_data_exploration():
    print("=" * 50)
    print(f"数据分析 Agent (1/3): 数据探索 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_analyst_agent()

    questions = [
        "帮我了解一下数据库里有什么数据？先列出所有文件，再查看 employees 表有多少员工",
        "技术部有多少人？平均薪资是多少？",
    ]

    for q in questions:
        print(f"用户: {q}")
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 10},
        )
        print(f"数智: {result['messages'][-1].content[:250]}")
        print()


# ============================================================
# Demo 2: 深度分析
# ============================================================

def demo_deep_analysis():
    print("=" * 50)
    print(f"数据分析 Agent (2/3): 深度分析 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_analyst_agent()

    questions = [
        "各部门的薪资分布情况如何？哪个部门平均薪资最高？",
        "绩效评分前3的员工分别是谁？他们参与了哪些项目？",
        "所有项目的总预算是多少？进行中的项目占多少比例？",
    ]

    for q in questions:
        print(f"用户: {q}")
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 10},
        )
        print(f"数智: {result['messages'][-1].content[:250]}")
        print()


# ============================================================
# Demo 3: 生成报告
# ============================================================

def demo_generate_report():
    print("=" * 50)
    print(f"数据分析 Agent (3/3): 生成报告 [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_analyst_agent()

    # 综合分析任务
    task = """请帮我做一份简短的分析报告，包括：
1. 员工总数和部门分布
2. 平均薪资和最高/最低薪资
3. 进行中的项目数量和总预算
把报告保存到 analysis_report.txt 文件中"""

    print(f"用户: {task}")
    print("-" * 40)

    for event in agent.stream(
        {"messages": [("user", task)]},
        config={"recursion_limit": 15},
    ):
        for node_name, node_output in event.items():
            msg = node_output["messages"][-1]
            if node_name == "agent" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    args_preview = str(tc["args"])
                    if len(args_preview) > 60:
                        args_preview = args_preview[:60] + "..."
                    print(f"  [调用] {tc['name']}({args_preview})")
            elif node_name == "tools":
                content = msg.content[:80].replace("\n", " ")
                print(f"  [结果] {content}")
            elif node_name == "agent" and msg.content:
                print(f"  [回答] {msg.content[:200]}")

    print()

    # 验证生成的文件
    print("--- 生成的报告内容 ---")
    report_content = read_file.invoke({"file_path": "analysis_report.txt"})
    print(report_content)
    print()


if __name__ == "__main__":
    setup_data()

    demo_data_exploration()
    demo_deep_analysis()
    demo_generate_report()

    # 清理
    try:
        shutil.rmtree(WORK_DIR)
    except Exception:
        pass

    print("=" * 50)
    print("Demo 12-3 完成!")
    print()
    print("数据分析工具集要点:")
    print("  1. 工具组合          - 数据库 + 文件 + 计算协同工作")
    print("  2. System Prompt     - 定义分析原则和角色")
    print("  3. 多步推理          - Agent 自动规划分析步骤")
    print("  4. 结果持久化        - 将分析结果保存到文件")
    print("  5. SQL 聚合          - 利用数据库能力做统计计算")
    print("  6. 上下文关联        - 跨表查询（如员工+项目）")
    print("=" * 50)
