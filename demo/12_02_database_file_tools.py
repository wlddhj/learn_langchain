"""
第12章 Demo 2：数据库与文件操作工具

演示 SQLite 查询工具、文件读写工具的开发。
可独立运行。
"""

import os
import sys
import sqlite3
import tempfile
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

# 使用临时目录存放数据库和文件
WORK_DIR = tempfile.mkdtemp(prefix="langchain_demo_")
DB_PATH = os.path.join(WORK_DIR, "shop.db")


# ============================================================
# 初始化示例数据库
# ============================================================

def setup_database():
    """创建示例数据库和测试数据"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        stock INTEGER
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        product_id INTEGER,
        quantity INTEGER,
        total_price REAL,
        status TEXT,
        created_at TEXT
    )""")

    # 插入示例数据
    products = [
        (1, "MacBook Pro 14", "电脑", 14999.0, 23),
        (2, "iPhone 15 Pro", "手机", 8999.0, 56),
        (3, "AirPods Pro 2", "配件", 1899.0, 120),
        (4, "iPad Air M2", "平板", 4799.0, 45),
        (5, "Apple Watch Ultra 2", "手表", 5999.0, 18),
        (6, "Magic Keyboard", "配件", 2499.0, 67),
        (7, "Studio Display", "显示器", 11499.0, 12),
        (8, "Mac Mini M2", "电脑", 4499.0, 34),
    ]

    orders = [
        (1, 1, 1, 14999.0, "已发货", "2024-01-10"),
        (2, 2, 2, 17998.0, "备货中", "2024-01-12"),
        (3, 3, 3, 5697.0, "已签收", "2024-01-08"),
        (4, 4, 1, 4799.0, "待付款", "2024-01-13"),
        (5, 5, 1, 5999.0, "已发货", "2024-01-11"),
        (6, 2, 1, 8999.0, "已签收", "2024-01-05"),
        (7, 8, 2, 8998.0, "备货中", "2024-01-14"),
        (8, 6, 5, 12495.0, "已发货", "2024-01-09"),
    ]

    c.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)", products)
    c.executemany("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
    conn.commit()
    conn.close()


# 创建示例文件
def setup_files():
    """创建示例文件"""
    # 创建销售数据 CSV
    csv_content = """日期,商品,数量,金额
2024-01-08,AirPods Pro 2,3,5697
2024-01-09,Magic Keyboard,5,12495
2024-01-10,MacBook Pro 14,1,14999
2024-01-11,Apple Watch Ultra 2,1,5999
2024-01-12,iPhone 15 Pro,2,17998
2024-01-13,iPad Air M2,1,4799
2024-01-14,Mac Mini M2,2,8998"""

    csv_path = os.path.join(WORK_DIR, "sales.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # 创建分析报告
    report_path = os.path.join(WORK_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("2024年1月销售分析报告\n")
        f.write("=" * 30 + "\n")
        f.write("总销售额: ¥71,085\n")
        f.write("总订单数: 8\n")
        f.write("最热销商品: AirPods Pro 2 (3件)\n")
        f.write("最高单价: MacBook Pro 14 (¥14,999)\n")

    return WORK_DIR


# ============================================================
# 数据库查询工具
# ============================================================

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询并返回格式化的结果。
    仅支持 SELECT 查询，不允许修改数据。
    可用的表:
      - products(商品表): id, name, category, price, stock
      - orders(订单表): id, product_id, quantity, total_price, status, created_at
    sql: SELECT 查询语句"""
    # 安全检查
    sql_stripped = sql.strip()
    sql_upper = sql_stripped.upper()

    if not sql_upper.startswith("SELECT"):
        return "错误: 仅支持 SELECT 查询，不允许执行 INSERT/UPDATE/DELETE/DROP 等操作。"

    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH", "DETACH"]
    for kw in dangerous:
        if kw in sql_upper.split():
            return f"错误: 检测到危险关键词 '{kw}'，操作被拒绝。"

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(sql_stripped)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "查询结果为空。"

        # 格式化输出
        col_widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))

        header = " | ".join(c.ljust(w) for c, w in zip(columns, col_widths))
        separator = "-+-".join("-" * w for w in col_widths)

        result_lines = [header, separator]
        for row in rows[:20]:
            result_lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

        total_info = f"\n(共 {len(rows)} 行)" if len(rows) > 20 else f"\n(共 {len(rows)} 行)"
        return "\n".join(result_lines) + total_info

    except sqlite3.Error as e:
        return f"SQL 错误: {e}"
    except Exception as e:
        return f"查询失败: {type(e).__name__} - {e}"


# ============================================================
# 文件操作工具
# ============================================================

@tool
def list_files(directory: str = "") -> str:
    """列出指定目录下的文件和文件夹。
    当需要查看有哪些文件时使用。
    directory: 目录路径（为空表示工作目录）"""
    target = os.path.join(WORK_DIR, directory) if directory else WORK_DIR
    try:
        items = os.listdir(target)
        if not items:
            return f"目录 '{directory or '工作目录'}' 为空。"

        result = f"目录内容 ({len(items)} 项):\n"
        for item in sorted(items):
            path = os.path.join(target, item)
            if os.path.isdir(path):
                result += f"  [目录] {item}/\n"
            else:
                size = os.path.getsize(path)
                if size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                result += f"  [文件] {item} ({size_str})\n"
        return result

    except FileNotFoundError:
        return f"目录不存在: {directory}"
    except Exception as e:
        return f"错误: {e}"


@tool
def read_file(file_path: str) -> str:
    """读取文件内容。
    当需要查看文件内容时使用。
    file_path: 文件名（如 'sales.csv'、'report.txt'）"""
    full_path = os.path.join(WORK_DIR, file_path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
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
    """将内容写入文件。如果文件已存在则覆盖。
    当需要保存数据到文件时使用。
    file_path: 文件名
    content: 要写入的内容"""
    full_path = os.path.join(WORK_DIR, file_path)
    try:
        os.makedirs(os.path.dirname(full_path) or WORK_DIR, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入 {len(content)} 字符到 {file_path}"
    except Exception as e:
        return f"写入失败: {e}"


# ============================================================
# Demo 1: 数据库工具
# ============================================================

def demo_database_tools():
    print("=" * 50)
    print("Demo 12-2 (1/3): 数据库查询工具")
    print("=" * 50)

    setup_database()

    # 直接测试
    queries = [
        "SELECT * FROM products WHERE category = '电脑'",
        "SELECT category, COUNT(*) as cnt, ROUND(AVG(price), 0) as avg_price FROM products GROUP BY category",
        "SELECT p.name, o.quantity, o.total_price, o.status FROM orders o JOIN products p ON o.product_id = p.id ORDER BY o.total_price DESC",
    ]

    for sql in queries:
        print(f"SQL: {sql}")
        result = query_database.invoke({"sql": sql})
        print(result)
        print()

    # 安全测试
    print("--- 安全检查测试 ---")
    dangerous = [
        "DROP TABLE products",
        "UPDATE products SET price = 0",
        "INSERT INTO products VALUES (99, 'hack', 'x', 0, 0)",
    ]
    for sql in dangerous:
        result = query_database.invoke({"sql": sql})
        print(f"  {sql[:40]}... → {result[:60]}")
    print()


# ============================================================
# Demo 2: 文件操作工具
# ============================================================

def demo_file_tools():
    print("=" * 50)
    print("Demo 12-2 (2/3): 文件操作工具")
    print("=" * 50)

    setup_files()

    # 列出文件
    print("--- 列出文件 ---")
    result = list_files.invoke({"directory": ""})
    print(result)

    # 读取 CSV
    print("--- 读取 sales.csv ---")
    result = read_file.invoke({"file_path": "sales.csv"})
    print(result)
    print()

    # 写入新文件
    print("--- 写入新文件 ---")
    result = write_file.invoke({
        "file_path": "summary.txt",
        "content": "销售摘要\n========\n总订单: 8笔\n总金额: ¥71,085"
    })
    print(result)

    # 验证
    result = read_file.invoke({"file_path": "summary.txt"})
    print(f"验证: {result}")
    print()


# ============================================================
# Demo 3: Agent 使用数据库和文件工具
# ============================================================

def demo_agent_tools():
    print("=" * 50)
    print(f"Demo 12-2 (3/3): Agent 使用数据库+文件工具 [{QWEN_MODEL}]")
    print("=" * 50)

    setup_database()
    setup_files()

    agent = create_agent(
        model=llm,
        tools=[query_database, list_files, read_file, write_file],
        system_prompt=f"""你是一个数据分析助手。你可以：
1. 使用 query_database 查询 SQLite 数据库（仅支持 SELECT）
2. 使用 list_files 查看文件列表
3. 使用 read_file 读取文件内容
4. 使用 write_file 写入文件

请根据用户需求选择合适的工具，用中文简洁回答。
工作目录: {WORK_DIR}""",
    )

    questions = [
        "帮我看看有哪些文件",
        "查看一下销量最高的3个商品",
        "把 products 表中价格低于5000的商品保存到 cheap_products.txt",
    ]

    for q in questions:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 10},
        )
        print(f"用户: {q}")
        print(f"AI:   {result['messages'][-1].content[:200]}")
        print()


if __name__ == "__main__":
    demo_database_tools()
    demo_file_tools()
    demo_agent_tools()

    # 清理临时文件
    import shutil
    try:
        shutil.rmtree(WORK_DIR)
    except Exception:
        pass

    print("=" * 50)
    print("Demo 12-2 完成!")
    print()
    print("数据库和文件工具要点:")
    print("  1. SQL 安全检查    - 只允许 SELECT，拦截危险操作")
    print("  2. 结果格式化      - 表格形式输出，便于 Agent 理解")
    print("  3. 文件路径安全    - 限制在工作目录内")
    print("  4. 内容截断        - 防止返回过多内容消耗 token")
    print("  5. 错误友好返回    - 返回可读的错误文本而非抛异常")
    print("=" * 50)
