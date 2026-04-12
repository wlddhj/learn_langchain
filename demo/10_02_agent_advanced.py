"""
第10章 Demo 2：Agent 进阶 —— 多轮对话、System Prompt、条件控制

演示 Agent 多轮对话、自定义 System Prompt、recursion_limit。
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
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 工具定义
# ============================================================

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库，获取公司内部信息。
    当用户询问公司政策、制度、流程时使用。
    query: 搜索关键词"""
    kb = {
        "年假": "入职满1年5天，满3年10天，满5年15天。可跨年累积，最多累积2年额度。",
        "加班": "工作日加班1.5倍工资，周末加班2倍，法定节假日3倍。需提前申请。",
        "远程办公": "每周最多2天远程办公，需提前1天申请并获得主管批准。",
        "体检": "公司每年组织一次免费体检，时间通常在10-11月。",
        "团建": "每季度一次部门团建，预算每人200元，由部门自行组织。",
    }
    for key, value in kb.items():
        if key in query:
            return value
    return f"未找到 '{query}' 相关信息。"


@tool
def get_employee_info(employee_id: str) -> str:
    """查询员工基本信息，包括姓名、部门、入职时间、年假余额。
    employee_id: 工号，格式如 E001"""
    employees = {
        "E001": "姓名: 张三 | 部门: 技术部 | 入职: 2021-03-15 | 年假余额: 7天 | 职级: P6",
        "E002": "姓名: 李四 | 部门: 产品部 | 入职: 2022-07-01 | 年假余额: 5天 | 职级: P5",
        "E003": "姓名: 王五 | 部门: 设计部 | 入职: 2019-11-20 | 年假余额: 12天 | 职级: P7",
    }
    return employees.get(employee_id, f"未找到工号 {employee_id} 的员工信息。")


@tool
def submit_leave(employee_id: str, leave_type: str, days: float, reason: str) -> str:
    """提交请假申请。
    employee_id: 工号
    leave_type: 假期类型 (年假/事假/病假)
    days: 请假天数
    reason: 请假原因"""
    return (f"请假申请已提交: 工号 {employee_id}，类型: {leave_type}，"
            f"天数: {days}天，原因: {reason}。状态: 待主管审批。")


# ============================================================
# 1. 带 System Prompt 的 Agent
# ============================================================

def demo_system_prompt():
    print("=" * 50)
    print(f"Demo 10-2 (1/3): 带 System Prompt 的 Agent [{QWEN_MODEL}]")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_knowledge, get_employee_info, submit_leave],
        system_prompt="""你是一个专业的 HR 助手「小智」。请遵循以下规则：

1. 始终使用中文回答，态度友好专业
2. 查询公司政策时使用 search_knowledge 工具
3. 查询员工信息时使用 get_employee_info 工具
4. 提交请假时使用 submit_leave 工具
5. 回答要简洁明了，不超过150字
6. 如果信息不足，礼貌地请用户补充""",
    )

    # 测试不同类型的问题
    questions = [
        "你好，请问公司年假政策是什么？",
        "帮我查一下工号 E001 的员工信息",
        "我是E002，想请2天年假去旅游",
    ]

    for q in questions:
        result = agent.invoke(
            {"messages": [("user", q)]},
            config={"recursion_limit": 8},
        )
        print(f"用户: {q}")
        print(f"小智: {result['messages'][-1].content[:200]}")
        print()


# ============================================================
# 2. 多轮对话 Agent
# ============================================================

def demo_multi_turn():
    print("=" * 50)
    print("Demo 10-2 (2/3): 多轮对话 Agent")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_knowledge, get_employee_info, submit_leave],
        system_prompt="你是 HR 助手，用中文简洁回答。",
    )

    # 模拟连续对话
    messages = []
    conversations = [
        "你好，我是技术部的张三，工号 E001",
        "帮我查一下我的年假余额",
        "公司年假政策是什么？",
        "我想请3天年假回家看看，可以吗？",
    ]

    for user_msg in conversations:
        messages.append(("user", user_msg))
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 8},
        )
        messages = result["messages"]

        ai_response = messages[-1].content
        print(f"用户: {user_msg}")
        print(f"AI:   {ai_response[:150]}")
        print(f"  (当前消息数: {len(messages)})")
        print()

    print("Agent 能记住之前对话中的上下文（如工号、部门等）")
    print()


# ============================================================
# 3. recursion_limit 安全控制
# ============================================================

def demo_recursion_limit():
    print("=" * 50)
    print("Demo 10-2 (3/3): recursion_limit 控制执行步数")
    print("=" * 50)

    agent = create_agent(
        model=llm,
        tools=[search_knowledge, get_employee_info, submit_leave],
        system_prompt="你是 HR 助手。",
    )

    # 正常请求
    print("测试1: 正常请求 (recursion_limit=5)")
    try:
        result = agent.invoke(
            {"messages": [("user", "公司体检政策是什么？")]},
            config={"recursion_limit": 5},
        )
        print(f"  结果: {result['messages'][-1].content[:100]}")
    except Exception as e:
        print(f"  错误: {e}")
    print()

    # 限制很小的 recursion_limit
    print("测试2: 极小限制 (recursion_limit=2)")
    try:
        result = agent.invoke(
            {"messages": [("user", "帮我查 E001 的年假余额，然后帮我提交请假申请")]},
            config={"recursion_limit": 2},
        )
        print(f"  结果: {result['messages'][-1].content[:100]}")
    except Exception as e:
        print(f"  错误 (达到步数限制): {str(e)[:100]}")
    print()

    print("建议: 一般设置 recursion_limit=10~15，足够大多数场景")
    print()


if __name__ == "__main__":
    demo_system_prompt()
    demo_multi_turn()
    demo_recursion_limit()

    print("=" * 50)
    print("Demo 10-2 完成!")
    print()
    print("Agent 进阶要点:")
    print("  prompt 参数        - 自定义 Agent 的行为和角色")
    print("  多轮对话           - 追加 messages 实现 Agent 记忆")
    print("  recursion_limit    - 限制最大执行步数")
    print("  配置项             - 通过 config 字典传递运行参数")
    print("=" * 50)
