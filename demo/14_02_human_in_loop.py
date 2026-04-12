"""
第14章 Demo 2：人机交互 (Human-in-the-Loop)

演示 interrupt_before、人工审批流程、状态恢复。
可独立运行。
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Annotated

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.environ.get("QWEN_API_KEY") or os.environ["QWEN_API_KEY"].startswith("your-"):
    print("错误: 未设置 QWEN_API_KEY")
    sys.exit(1)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

QWEN_API_KEY = os.environ["QWEN_API_KEY"]
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

llm = ChatOpenAI(model=QWEN_MODEL, temperature=0, api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ============================================================
# 邮件审批 State
# ============================================================

class EmailState(TypedDict):
    topic: str
    recipient: str
    email_draft: str
    approved: bool
    sent: bool


def draft_email(state: EmailState):
    """起草邮件"""
    prompt = ChatPromptTemplate.from_template(
        "写一封关于「{topic}」的商务邮件，收件人: {recipient}。"
        "邮件要专业、简洁，不超过150字。"
    )
    chain = prompt | llm
    result = chain.invoke({"topic": state["topic"], "recipient": state["recipient"]})
    return {"email_draft": result.content, "sent": False}


def review_node(state: EmailState):
    """审批节点（实际由人工审批，这里只是占位）"""
    return {}


def send_email(state: EmailState):
    """发送邮件"""
    if state.get("approved"):
        return {"sent": True}
    return {"sent": False}


# ============================================================
# 1. 基本人机交互审批
# ============================================================

def demo_approval_flow():
    print("=" * 50)
    print(f"Demo 14-2 (1/3): 邮件审批流程 [{QWEN_MODEL}]")
    print("=" * 50)

    checkpointer = MemorySaver()

    builder = StateGraph(EmailState)
    builder.add_node("draft", draft_email)
    builder.add_node("review", review_node)
    builder.add_node("send", send_email)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "send")
    builder.add_edge("send", END)

    # 在 review 节点前暂停，等待人工确认
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["review"],
    )

    config = {"configurable": {"thread_id": "email-1"}}

    # Step 1: 生成草稿（执行到 review 前暂停）
    print("Step 1: 生成邮件草稿...")
    result = graph.invoke({
        "topic": "项目进度更新",
        "recipient": "张总",
        "email_draft": "",
        "approved": False,
        "sent": False,
    }, config)

    # 查看草稿
    state = graph.get_state(config)
    draft = state.values.get("email_draft", "")
    print(f"  邮件草稿:\n{preview(draft, 200)}")
    print(f"  当前暂停在节点: {state.next}")
    print()

    # Step 2: 人工审批通过
    print("Step 2: 人工审批通过...")
    graph.invoke({"approved": True}, config)

    # 查看最终状态
    state = graph.get_state(config)
    print(f"  邮件已发送: {state.values.get('sent', False)}")
    print()


# ============================================================
# 2. 拒绝并重新生成
# ============================================================

def demo_rejection():
    print("=" * 50)
    print(f"Demo 14-2 (2/3): 拒绝并重新生成 [{QWEN_MODEL}]")
    print("=" * 50)

    checkpointer = MemorySaver()

    builder = StateGraph(EmailState)
    builder.add_node("draft", draft_email)
    builder.add_node("review", review_node)
    builder.add_node("send", send_email)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "send")
    builder.add_edge("send", END)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["review"],
    )

    config = {"configurable": {"thread_id": "email-reject"}}

    # 第一次生成
    result = graph.invoke({
        "topic": "季度销售报告",
        "recipient": "王经理",
        "email_draft": "",
        "approved": False,
        "sent": False,
    }, config)

    state = graph.get_state(config)
    print(f"第一次草稿:\n{preview(state.values.get('email_draft', ''), 150)}")
    print()

    # 拒绝（不批准）
    print("审批: 拒绝（模拟）")
    graph.invoke({"approved": False}, config)

    state = graph.get_state(config)
    print(f"发送状态: {state.values.get('sent', False)} (未发送)")
    print()


# ============================================================
# 3. 多种中断点
# ============================================================

class TaskState(TypedDict):
    task: str
    analysis: str
    action_plan: str
    approved: bool
    result: str


def analyze_task(state: TaskState):
    """分析任务"""
    response = llm.invoke([
        ("system", "分析以下任务，给出简要分析（50字以内）。"),
        ("human", state["task"]),
    ])
    return {"analysis": response.content}


def plan_action(state: TaskState):
    """制定行动计划"""
    response = llm.invoke([
        ("system", "基于分析结果，制定行动计划（100字以内）。"),
        ("human", f"任务: {state['task']}\n分析: {state['analysis']}"),
    ])
    return {"action_plan": response.content}


def execute_task(state: TaskState):
    """执行任务"""
    if state.get("approved"):
        return {"result": f"已执行: {state['action_plan'][:80]}"}
    return {"result": "任务被拒绝，未执行"}


def demo_multi_interrupt():
    print("=" * 50)
    print(f"Demo 14-2 (3/3): 多阶段审批 [{QWEN_MODEL}]")
    print("=" * 50)

    checkpointer = MemorySaver()

    builder = StateGraph(TaskState)
    builder.add_node("analyze", analyze_task)
    builder.add_node("plan", plan_action)
    builder.add_node("execute", execute_task)

    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", END)

    # 在 execute 前暂停
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"],
    )

    config = {"configurable": {"thread_id": "task-1"}}

    # 阶段1: 分析 + 计划
    print("阶段1: 分析任务并制定计划...")
    graph.invoke({
        "task": "优化数据库查询性能",
        "analysis": "",
        "action_plan": "",
        "approved": False,
        "result": "",
    }, config)

    state = graph.get_state(config)
    print(f"  分析: {preview(state.values.get('analysis', ''), 80)}")
    print(f"  计划: {preview(state.values.get('action_plan', ''), 100)}")
    print(f"  暂停在: {state.next}")
    print()

    # 阶段2: 人工审批后执行
    print("阶段2: 审批通过，继续执行...")
    graph.invoke({"approved": True}, config)

    state = graph.get_state(config)
    print(f"  结果: {state.values.get('result', '')}")
    print()


def preview(text, max_len):
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


if __name__ == "__main__":
    demo_approval_flow()
    demo_rejection()
    demo_multi_interrupt()

    print("=" * 50)
    print("Demo 14-2 完成!")
    print()
    print("人机交互要点:")
    print("  interrupt_before   - 在指定节点前暂停")
    print("  interrupt_after    - 在指定节点后暂停")
    print("  get_state()        - 查看暂停时的状态")
    print("  invoke(None/更新)  - 恢复执行")
    print("  MemorySaver        - 持久化暂停点状态")
    print("  适用场景           - 审批、确认、敏感操作")
    print("=" * 50)
