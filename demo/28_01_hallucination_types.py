"""
第28章 Demo 1：幻觉类型识别与示例

演示 LLM 幻觉的各种类型：事实幻觉、来源幻觉、逻辑幻觉等。
可独立运行，无需 API Key。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def demo_hallucination_types():
    """演示各种幻觉类型"""
    print("=" * 60)
    print("Demo 28-1: LLM 幻觉类型识别")
    print("=" * 60)
    print()

    # 幻觉类型定义
    hallucination_types = {
        "事实幻觉": {
            "描述": "编造不存在的事实",
            "示例": "李白出生于 2020 年",
            "正确信息": "李白出生于公元 701 年",
        },
        "来源幻觉": {
            "描述": "错误引用来源",
            "示例": "根据《2024年人工智能报告》指出...",
            "正确信息": "该报告可能不存在",
        },
        "逻辑幻觉": {
            "描述": "推理链条错误",
            "示例": "因为 A 是 B，所以 C 是 D（实际上无因果关系）",
            "正确信息": "逻辑推理错误",
        },
        "数量幻觉": {
            "描述": "错误的数量信息",
            "示例": "全球人口约 100 亿",
            "正确信息": "全球人口约 80 亿",
        },
        "时间幻觉": {
            "描述": "时间信息错误",
            "示例": "iPhone 15 发布于 2010 年",
            "正确信息": "iPhone 15 发布于 2023 年",
        },
        "细节幻觉": {
            "描述": "细节描述错误",
            "示例": "鲁迅的代表作是《红楼梦》",
            "正确信息": "鲁迅代表作是《狂人日记》等",
        },
    }

    print("幻觉类型分类：")
    print("-" * 60)

    for type_name, info in hallucination_types.items():
        print(f"\n【{type_name}】")
        print(f"  描述: {info['描述']}")
        print(f"  示例: {info['示例']}")
        print(f"  正确: {info['正确信息']}")

    print()
    print("-" * 60)


def demo_hallucination_causes():
    """演示幻觉产生的原因"""
    print("=" * 60)
    print("幻觉产生原因分析")
    print("=" * 60)
    print()

    causes = {
        "数据问题": [
            "训练数据不完整或过时",
            "数据偏见或噪声",
            "知识边界模糊",
        ],
        "模型问题": [
            "模型过度自信",
            "生成策略（温度过高）",
            "缺乏不确定性表达",
        ],
        "Prompt问题": [
            "问题超出模型知识范围",
            "诱导性问题",
            "缺乏上下文约束",
        ],
        "RAG问题": [
            "检索内容不相关",
            "检索内容不足",
            "模型忽略检索内容",
        ],
    }

    print("幻觉原因分类：")
    print("-" * 60)

    for cause_type, reasons in causes.items():
        print(f"\n【{cause_type}】")
        for reason in reasons:
            print(f"  - {reason}")

    print()
    print("-" * 60)


def demo_detection_indicators():
    """演示幻觉检测指标"""
    print("=" * 60)
    print("幻觉风险指标")
    print("=" * 60)
    print()

    indicators = [
        ("知识边界风险", "问题是否超出模型知识范围", "高"),
        ("细节过度风险", "是否提供了过多无法验证的细节", "中"),
        ("时间敏感风险", "回答是否涉及可能过时的信息", "高"),
        ("来源引用风险", "是否引用了可能不存在的内容", "高"),
        ("数量精确风险", "是否给出过于精确的数量数据", "中"),
        ("逻辑连贯风险", "推理链条是否合理", "中"),
    ]

    print("风险指标列表：")
    print("-" * 60)
    print(f"{'指标名称':<20} {'说明':<30} {'风险等级':<10}")
    print("-" * 60)

    for name, desc, level in indicators:
        print(f"{name:<20} {desc:<30} {level:<10}")

    print()


def demo_safe_prompt_design():
    """演示安全 Prompt 设计"""
    print("=" * 60)
    print("安全 Prompt 设计示例")
    print("=" * 60)
    print()

    safe_prompt_template = """
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
"""

    print("安全 Prompt 模板：")
    print("-" * 60)
    print(safe_prompt_template)
    print("-" * 60)

    print("\n设计要点：")
    print("  1. 明确知识边界，要求模型承认无知")
    print("  2. 强制标注不确定性")
    print("  3. 禁止编造具体细节")
    print("  4. 警惕虚假来源引用")
    print("  5. 分层输出（确认/不确定/缺失）")


def demo_threshold_effect():
    """演示温度参数对幻觉的影响"""
    print("=" * 60)
    print("温度参数对幻觉的影响")
    print("=" * 60)
    print()

    temperature_guidelines = [
        ("事实查询", 0.0, "最低温度，减少幻觉风险"),
        ("知识问答", 0.1, "低温度，保证准确性"),
        ("一般对话", 0.3, "适中温度，平衡准确和流畅"),
        ("创意写作", 0.7, "较高温度，允许创意发挥"),
        ("故事生成", 0.9, "高温度，最大化创意"),
    ]

    print("温度设置建议：")
    print("-" * 60)
    print(f"{'场景':<15} {'温度':<10} {'说明':<30}")
    print("-" * 60)

    for scene, temp, desc in temperature_guidelines:
        print(f"{scene:<15} {temp:<10} {desc:<30}")

    print()
    print("结论：事实性问题使用低温度（0-0.1），创意问题可适当提高")


if __name__ == "__main__":
    demo_hallucination_types()
    demo_hallucination_causes()
    demo_detection_indicators()
    demo_safe_prompt_design()
    demo_threshold_effect()

    print("=" * 60)
    print("Demo 28-1 完成!")
    print()
    print("幻觉类型总结：")
    print("  - 事实幻觉：编造不存在的事实")
    print("  - 来源幻觉：错误引用来源")
    print("  - 逻辑幻觉：推理链条错误")
    print("  - 数量/时间/细节幻觉：具体信息错误")
    print()
    print("预防策略：")
    print("  - 安全 Prompt 设计（知识边界声明）")
    print("  - 低温度参数（事实查询场景）")
    print("  - 强约束 RAG（只基于检索内容回答）")
    print("=" * 60)