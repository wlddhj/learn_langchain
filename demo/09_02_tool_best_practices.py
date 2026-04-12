"""
第9章 Demo 2：工具描述最佳实践

演示好描述 vs 差描述对 LLM 工具选择的影响。
可独立运行。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")


def demo_good_vs_bad():
    print("=" * 50)
    print(f"Demo 9-2: 工具描述质量对比 [{QWEN_MODEL}]")
    print("=" * 50)

    # ---- 差的工具描述 ----

    @tool
    def search_vague(query: str) -> str:
        """查询"""  # 太模糊，LLM 不知道何时使用
        return f"查询结果: {query}"

    # ---- 好的工具描述 ----

    @tool
    def search_precise(query: str) -> str:
        """搜索公司内部知识库，返回与关键词匹配的文档片段。
        当用户询问公司政策、产品说明、技术文档等内部信息时使用此工具。
        query: 搜索关键词，尽量使用具体词汇"""
        return f"知识库搜索结果: {query}"

    # 对比测试
    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    questions = [
        "公司年假政策是什么？",
        "你好，今天几号？",
        "怎么申请报销？",
    ]

    print("--- 好的描述: search_precise ---")
    llm_good = llm.bind_tools([search_precise])
    for q in questions:
        resp = llm_good.invoke([HumanMessage(content=q)])
        tool_name = resp.tool_calls[0]["name"] if resp.tool_calls else "无"
        print(f"  Q: {q}")
        print(f"  → 工具: {tool_name}")
    print()

    print("--- 差的描述: search_vague ---")
    llm_bad = llm.bind_tools([search_vague])
    for q in questions:
        resp = llm_bad.invoke([HumanMessage(content=q)])
        tool_name = resp.tool_calls[0]["name"] if resp.tool_calls else "无"
        print(f"  Q: {q}")
        print(f"  → 工具: {tool_name}")
    print()

    print("结论: 清晰的描述帮助 LLM 准确判断何时使用工具")
    print()


def demo_tool_count():
    print("=" * 50)
    print("工具数量对 LLM 选择的影响")
    print("=" * 50)

    # 少量工具
    @tool
    def lookup_order(order_id: str) -> str:
        """根据订单号查询订单状态和物流信息"""
        return f"订单 {order_id}: 已发货"

    @tool
    def check_inventory(product_id: str) -> str:
        """查询商品库存数量"""
        return f"商品 {product_id}: 库存充足"

    llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    llm_few = llm.bind_tools([lookup_order, check_inventory])

    resp = llm_few.invoke("我的订单 ORD-123 到哪了？")
    print(f"工具数=2, 问题='订单查询'")
    print(f"  LLM 选择: {resp.tool_calls[0]['name'] if resp.tool_calls else '无工具调用'}")

    # 过多工具（故意使用模糊描述）
    @tool
    def tool_a(x: str) -> str:
        """处理数据"""
        return f"A: {x}"

    @tool
    def tool_b(x: str) -> str:
        """处理数据"""
        return f"B: {x}"

    @tool
    def tool_c(x: str) -> str:
        """处理数据"""
        return f"C: {x}"

    @tool
    def tool_d(x: str) -> str:
        """处理数据"""
        return f"D: {x}"

    @tool
    def tool_e(x: str) -> str:
        """处理数据"""
        return f"E: {x}"

    @tool
    def tool_f(x: str) -> str:
        """描述和处理信息"""
        return f"F: {x}"

    llm_many = llm.bind_tools([tool_a, tool_b, tool_c, tool_d, tool_e, tool_f])

    resp = llm_many.invoke("帮我处理一下这些信息")
    print(f"\n工具数=6(描述相似), 问题='处理信息'")
    print(f"  LLM 选择: {resp.tool_calls[0]['name'] if resp.tool_calls else '无工具调用'}")
    print()
    print("结论: 3-5 个描述清晰的工具效果最好，工具过多且描述模糊时 LLM 容易选错")
    print()


if __name__ == "__main__":
    demo_good_vs_bad()
    demo_tool_count()

    print("=" * 50)
    print("Demo 9-2 完成!")
    print()
    print("工具描述最佳实践:")
    print("  1. 说明工具做什么 + 何时使用")
    print("  2. 描述参数的含义和格式")
    print(" 3. 3-5个工具最佳，避免过多")
    print("  4. 不同工具的描述要有区分度")
    print("=" * 50)
