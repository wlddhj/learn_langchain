"""
第26章 Demo：模型微调概念演示

展示 LoRA 微调的核心概念和数据准备流程。
无需 GPU，侧重概念理解。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def demo_finetuning_concepts():
    """微调概念"""
    print("=" * 60)
    print("Demo 26-1 (1/3): 模型微调概念")
    print("=" * 60)
    print()

    print("微调方案对比：")
    print("-" * 60)
    print("""
| 方案 | 训练参数 | 显存需求 | 推荐指数 |
|------|----------|---------|---------|
| 全量微调 | 100% | 极高 | ⭐⭐ |
| LoRA | ~0.1% | 中等 | ⭐⭐⭐⭐⭐ |
| QLoRA | ~0.1% | 低 | ⭐⭐⭐⭐ |
| Adapter | ~0.5% | 中等 | ⭐⭐⭐⭐ |
| Prompt Tuning | ~0.0003% | 极低 | ⭐⭐⭐ |
""")
    print()

    print("LoRA 原理：")
    print("-" * 60)
    print("""
原始权重矩阵 W（大）: 4096 × 4096 = 16M 参数

LoRA 新增矩阵 A × B（小）:
┌─────────┐   ┌─────────┐
│ 4096×16 │ × │ 16×4096 │ = 128K 参数
└─────────┘   └─────────┘

参数减少约 99.94%，效果接近全量微调
"")


def demo_lora_config():
    """LoRA 配置示例"""
    print("=" * 60)
    print("Demo 26-1 (2/3): LoRA 配置代码")
    print("=" * 60)
    print()

    code = """
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # 任务类型
    r=16,                            # 低秩维度
    lora_alpha=32,                   # 缩放系数 (推荐 2×r)
    lora_dropout=0.05,               # 防过拟合
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",     # MLP层
    ],
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: ~4M || trainable%: 0.06%
"""
    print("LoRA 配置示例：")
    print("-" * 60)
    print(code)
    print()

    print("关键参数说明：")
    print("-" * 60)
    print("""
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| r | 低秩维度 | 16-64 |
| lora_alpha | 缩放系数 | 2×r |
| lora_dropout | 防过拟合 | 0.05-0.1 |
| target_modules | 应用位置 | 注意力层 |
""")


def demo_data_preparation():
    """数据准备"""
    print("=" * 60)
    print("Demo 26-1 (3/3): 训练数据准备")
    print("=" * 60)
    print()

    print("数据格式标准：")
    print("-" * 60)
    print()

    print("Alpaca 格式（最常用）：")
    alpaca_example = """
[
  {
    "instruction": "请解释以下金融术语",
    "input": "什么是市盈率（P/E）？",
    "output": "市盈率是股票价格与每股收益的比值..."
  },
  {
    "instruction": "将以下文本翻译成英文",
    "input": "人工智能正在改变世界",
    "output": "Artificial intelligence is changing the world"
  }
]
"""
    print(alpaca_example)

    print("ShareGPT 格式（对话场景）：")
    sharegpt_example = """
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！有什么可以帮你的？"},
      {"from": "human", "value": "我想了解 Python"},
      {"from": "gpt", "value": "Python 是一门流行的编程语言..."}
    ]
  }
]
"""
    print(sharegpt_example)

    print("数据质量要点：")
    print("-" * 60)
    print("""
好数据特征：
├── 准确：答案正确无误
├── 一致：格式风格统一
├── 相关：与任务高度相关
├── 多样：覆盖各种情况
└── 清晰：无歧义无噪声

数据量建议：
├── 最少：500-1000 条（效果有限）
├── 推荐：2000-5000 条（较好效果）
├── 良好：10000+ 条（稳定效果）
└── 理想：100000+ 条（最佳效果）
"")


def demo_data_validation():
    """数据验证脚本"""
    print("=" * 60)
    print("Demo 26-1 (补充): 数据验证")
    print("=" * 60)
    print()

    code = """
import json

def validate_dataset(filepath: str):
    """验证数据集"""
    with open(filepath) as f:
        data = json.load(f)

    issues = []

    for i, item in enumerate(data):
        # 检查必填字段
        if not item.get("instruction"):
            issues.append(f"第{i}条：缺少 instruction")

        if not item.get("output"):
            issues.append(f"第{i}条：缺少 output")

        # 检查长度
        if len(item.get("output", "")) < 20:
            issues.append(f"第{i}条：output 过短")

        if len(item.get("output", "")) > 2000:
            issues.append(f"第{i}条：output 过长")

    print(f"总数据: {len(data)} 条")
    print(f"问题数: {len(issues)} 个")

    return len(issues) == 0

# 使用
validate_dataset("train_data.json")
"""
    print("数据验证脚本：")
    print("-" * 60)
    print(code)


if __name__ == "__main__":
    demo_finetuning_concepts()
    demo_lora_config()
    demo_data_preparation()
    demo_data_validation()

    print("=" * 60)
    print("Demo 26-1 完成!")
    print()
    print("微调要点：")
    print("  - LoRA 是最推荐的方案")
    print("  - 参数减少 99%+，效果接近全量微调")
    print("  - 数据质量比数量更重要")
    print("  - Alpaca 格式是最常用的数据格式")
    print()
    print("何时选择微调：")
    print("  - Prompt/RAG 无法解决的问题")
    print("  - 需要模型内化知识或行为")
    print("  - 数据量充足 (>2000条)")
    print()
    print("实际训练需要：")
    print("  - GPU 显存 >16GB (LoRA)")
    print("  - GPU 显存 >4GB (QLoRA)")
    print("  - transformers + peft + accelerate")
    print("=" * 60)