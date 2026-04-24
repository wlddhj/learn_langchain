# 第26章：模型微调入门与实践

## 26.1 什么是模型微调

### 概念理解

**预训练模型**就像一个"通用人才"，学习了大量基础知识，但不擅长特定任务。

**微调**就像"专业培训"，让通用人才变成某个领域的专家。

```
预训练阶段（通用知识）
├── 学习语言结构、词汇、语法
├── 学习世界知识、常识
├── 学习基本的推理能力
└── 结果：能理解语言，但回答可能不够专业

微调阶段（专业训练）
├── 学习特定领域的术语和知识
├── 学习特定的回答格式和风格
├── 学习特定任务的执行方式
└── 结果：在特定领域表现出色
```

### 为什么需要微调

| 问题 | 不微调的表现 | 微调后的效果 |
|------|------------|------------|
| **回答不专业** | 泛泛而谈 | 深入精准 |
| **格式不规范** | 格式混乱 | 结构清晰 |
| **术语不准确** | 通用表述 | 专业术语 |
| **风格不匹配** | 通用风格 | 符合品牌风格 |
| **任务执行差** | 理解偏差 | 准确执行 |

### 微调 vs 其他方案

| 方案 | 改变模型 | 效果上限 | 成本 | 适用场景 |
|------|---------|---------|------|---------|
| **Prompt Engineering** | 不改变 | 中等 | 极低 | 简单定制 |
| **RAG** | 不改变 | 中高 | 低 | 知识注入 |
| **微调** | 改变权重 | 高 | 中等 | 行为定制 |

**何时选择微调？**

```
1. Prompt/RAG 无法解决的问题
   ├── 输出格式顽固不对
   ├── 专业术语使用错误
   ├── 回答风格无法调整
   └── 特定任务执行失败

2. 需要模型"内化"知识或行为
   ├── 需要记住特定风格
   ├── 需要自动使用专业术语
   ├── 需要特定的推理模式

3. 数据量充足（建议 >1000 条高质量数据）
```

---

## 26.2 微调的核心原理

### 神经网络权重

模型由数十亿个"权重参数"组成，每个参数控制模型的一部分行为：

```
模型参数示例（7B 模型）
├── 总参数量：约 7,000,000,000（70亿）
├── 参数类型：矩阵权重、偏置项
├── 存储大小：约 14GB（float16）
└── 微调时：调整部分参数，改变模型行为
```

### 微调的本质

微调就是**调整权重参数**，让模型在特定任务上表现更好：

```python
# 概念示意（不是实际代码）
原始权重 W = [[0.1, 0.2], [0.3, 0.4]]

微调后权重 W' = [[0.15, 0.18], [0.32, 0.35]]

# 微小的权重变化 → 显著的行为改变
```

### 梯度下降

微调使用**梯度下降**算法来更新权重：

```
训练流程：
1. 输入数据 → 模型预测 → 得到输出
2. 计算预测与真实答案的差距（损失）
3. 计算每个参数对损失的影响（梯度）
4. 根据梯度调整参数，减少损失
5. 重复多次，模型越来越准确
```

---

## 26.3 微调方案详解

### 方案对比总览

| 方案 | 训练参数比例 | 显存需求 | 训练速度 | 推荐指数 | 适用场景 |
|------|------------|---------|---------|---------|---------|
| **全量微调** | 100% | 极高 | 慢 | ⭐⭐ | 大规模定制 |
| **LoRA** | ~0.1% | 中等 | 快 | ⭐⭐⭐⭐⭐ | 大多数场景（推荐） |
| **QLoRA** | ~0.1% | 低 | 中等 | ⭐⭐⭐⭐ | 显存受限场景 |
| **AdaLoRA** | 动态调整 | 中等 | 快 | ⭐⭐⭐⭐ | 自适应优化 |
| **Adapter** | ~0.5% | 中等 | 快 | ⭐⭐⭐⭐ | 多任务切换 |
| **Prompt Tuning** | ~0.0003% | 极低 | 极快 | ⭐⭐⭐ | 简单定制 |

### 全量微调

**原理**：调整模型的所有参数。

```
优点：
├── 定制程度最高
├── 效果上限最高
└── 适合大规模数据（>100K）

缺点：
├── 显存需求极大（7B 需要 >40GB）
├── 训练时间长
├── 容易过拟合
└── 需要大量数据

适用场景：
└── 企业级大规模定制，有充足的算力和数据
```

### LoRA（强烈推荐）

**原理**：冻结原模型参数，只训练新增的"低秩矩阵"。

```
核心思想：
├── 参数变化 ΔW 可以分解为两个小矩阵：ΔW = A × B
├── A 和 B 远小于 W，参数量减少 99%+
├── 训练只更新 A 和 B，原模型不变
└── 推理时合并：W' = W + A × B

参数对比（7B 模型）：
├── 全量微调：训练 7,000,000,000 参数
├── LoRA (r=16)：训练 ~4,000,000 参数
└── 减少约 99.94%
```

**可视化理解**：

```
原始模型权重矩阵 W（大）:
┌────────────────────┐
│  4096 × 4096       │  ← 16M 参数
└────────────────────┘

LoRA 新增矩阵 A × B（小）:
┌─────────┐   ┌─────────┐
│ 4096×16 │ × │ 16×4096 │  ← 128K 参数
└─────────┘   ┌─────────┘

两者效果相近，但参数量相差 100 倍以上
```

**LoRA 的关键参数**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` | 低秩维度 | 8-64（越大效果越好但参数越多） |
| `lora_alpha` | 缩放系数 | 2×r（如 r=16 则 alpha=32） |
| `lora_dropout` | 防过拟合 | 0.05-0.1 |
| `target_modules` | 应用位置 | 注意力层（q_proj, k_proj, v_proj） |

### QLoRA

**原理**：LoRA + 4-bit 量化，进一步降低显存需求。

```
量化技术：
├── 将 float16（16位）压缩为 4-bit（4位）
├── 显存减少约 75%
├── 训练效果基本不变
└── 适合显存受限的情况

显存对比（7B 模型）：
├── 全量微调：>40GB
├── LoRA：约 16GB
├── QLoRA：约 4GB  ← 单卡 GPU 就能跑
```

### Adapter（适配器）

**原理**：在模型层之间插入小型神经网络模块。

```
核心思想：
├── 冻结原模型参数
├── 在 Transformer 层之间插入小型 Adapter 模块
├── 只训练 Adapter 模块（参数量 <1%）
└── 不同任务使用不同的 Adapter，可随时切换

Adapter 结构：
┌─────────────────────────────┐
│     Transformer 层          │
├─────────────────────────────┤
│  ↓ 输出                     │
├─────────────────────────────┤
│  Adapter（小型瓶颈结构）      │  ← 新增模块，约 0.5% 参数
│  ├── Down-project（降维）    │
│  ├── Non-linearity（激活）   │
│  ├── Up-project（升维）      │
├─────────────────────────────┤
│  ↓ + 原输出（残差连接）       │
└─────────────────────────────┘
```

**Adapter vs LoRA 对比**：

| 维度 | Adapter | LoRA |
|------|---------|------|
| **插入位置** | 层之间 | 权重矩阵旁 |
| **参数量** | ~0.5% | ~0.1% |
| **推理延迟** | 略增加 | 无增加（合并后） |
| **切换能力** | 优秀（多个 Adapter） | 需动态加载 |
| **适用场景** | 多任务切换 | 单任务优化 |

**使用 Adapter 的场景**：

```
场景：一个基座模型服务多个业务线

├── 业务A：客服对话 → Adapter-A
├── 业务B：技术问答 → Adapter-B
├── 业务C：营销文案 → Adapter-C
└── 推理时：根据请求动态切换 Adapter

优点：
├── 一个基座模型 + 多个小型 Adapter
├── 切换成本低（加载不同 Adapter）
├── 存储成本低（Adapter 只有几 MB）
└── 维护成本低（独立更新各 Adapter）
```

**Adapter 实现示例**：

```python
from peft import AdapterConfig, get_peft_model

# Adapter 配置
adapter_config = AdapterConfig(
    adapter_type="houlsby",        # Houlsby Adapter（经典结构）
    adapter_size=64,               # Adapter 内部维度
    adapter_dropout=0.1,
    target_modules=[
        "attention",               # 注意力层后插入
        "output"                   # 输出层后插入
    ],
)

# 应用 Adapter
model = get_peft_model(model, adapter_config)
model.print_trainable_parameters()
# 输出: trainable params: 35,000,000 || trainable%: 0.5%

# 训练完成后，可以切换不同 Adapter
# 加载 Adapter-A
model.load_adapter("./adapter_customer", adapter_name="customer")
# 加载 Adapter-B
model.load_adapter("./adapter_technical", adapter_name="technical")

# 推理时激活特定 Adapter
model.set_adapter("customer")  # 切换到客服 Adapter
response = model.generate(...)

model.set_adapter("technical")  # 切换到技术 Adapter
response = model.generate(...)
```

### Prompt Tuning（软提示调优）

**原理**：在输入前添加可学习的"软提示"向量，不改变模型权重。

```
核心思想：
├── 在输入 embedding 前添加一组可学习的向量
├── 这些向量通过训练优化，引导模型输出
├── 模型权重完全冻结，不改变
├── 参数量几乎为 0（只有几十个 prompt 向量）
└── 效果介于 Prompt Engineering 和 微调之间

输入结构：
┌─────────────────────────────────────────────────┐
│ [软提示向量] [用户输入 tokens]                   │
│   ↓           ↓                                  │
│  可学习      固定                                 │
│   ↓           ↓                                  │
│     → 合成 embedding → 模型处理 →               │
└─────────────────────────────────────────────────┘

参数对比：
├── 全量微调：7B 参数
├── LoRA：~4M 参数
├── Adapter：~35M 参数
├── Prompt Tuning：~20 个向量（几乎 0 参数）
```

**Prompt Tuning 适用场景**：

```
适合：
├── 简单的任务定制（分类、简单问答）
├── 需要极低成本微调
├── 需要快速尝试多个任务
└── 模型部署受限（不能修改权重）

不适合：
├── 复杂任务（需要深度改变模型行为）
├── 需要改变输出格式
├── 需要学习新的专业知识
└── 需要显著的性能提升
```

**Prompt Tuning 实现示例**：

```python
from peft import PromptTuningConfig, get_peft_model, TaskType

# Prompt Tuning 配置
prompt_tuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init="TEXT",           # 用文本初始化软提示
    prompt_tuning_init_text="请回答以下问题：",  # 初始化文本
    num_virtual_tokens=20,               # 软提示长度
    tokenizer_name_or_path=model_name,
)

# 应用 Prompt Tuning
model = get_peft_model(model, prompt_tuning_config)
model.print_trainable_parameters()
# 输出: trainable params: 20,480 || trainable%: 0.0003%
# （只有 20 个虚拟 token 的 embedding 参数）

# 训练
trainer.train()

# 保存软提示（只有几 KB）
model.save_pretrained("./prompt_tuning_weights")
```

**Prompt Tuning vs Prefix Tuning**：

| 维度 | Prompt Tuning | Prefix Tuning |
|------|--------------|---------------|
| **插入位置** | 仅输入端 | 每层都添加 |
| **参数量** | 最少 | 较少 |
| **效果** | 较弱 | 较强 |
| **复杂度** | 简单 | 中等 |
| **适用** | 简单任务 | 中等任务 |

---

## 26.4 微调前的准备工作

### 确定微调目标

```
1. 明确任务类型
   ├── 问答任务：教模型回答特定问题
   ├── 分类任务：教模型分类文本
   ├── 生成任务：教模型生成特定格式文本
   └── 对话任务：教模型特定的对话风格

2. 定义评估标准
   ├── 准确率：回答是否正确
   ├── 格式合规率：格式是否符合要求
   ├── 用户满意度：人工评估
   └── 任务完成率：任务是否完成
```

### 准备训练数据

**数据质量比数量更重要**：

```
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
```

### 数据格式标准

#### Alpaca 格式（最常用）

```json
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
```

#### ShareGPT 格式（对话场景）

```json
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
```

#### 自定义格式示例

```json
[
  {
    "question": "客户投诉产品质量问题，如何回复？",
    "answer": "感谢您的反馈。我们非常重视产品质量...",
    "category": "客服回复",
    "style": "礼貌专业"
  }
]
```

### 数据清洗技巧

```python
import json

def clean_dataset(data):
    """清洗数据集"""
    cleaned = []
    
    for item in data:
        # 1. 检查必填字段
        if not item.get("instruction") or not item.get("output"):
            continue
        
        # 2. 去除过长数据
        if len(item["output"]) > 1000:
            continue
        
        # 3. 去除重复数据
        # ...
        
        # 4. 格式标准化
        item["instruction"] = item["instruction"].strip()
        item["output"] = item["output"].strip()
        
        cleaned.append(item)
    
    return cleaned

# 使用
with open("raw_data.json") as f:
    raw_data = json.load(f)

cleaned_data = clean_dataset(raw_data)
print(f"原始: {len(raw_data)} 条, 清洗后: {len(cleaned_data)} 条")
```

---

## 26.5 LoRA 微调实战

### 环境安装

```bash
# 创建虚拟环境
conda create -n finetune python=3.10
conda activate finetune

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft bitsandbytes

# 安装训练工具
pip install datasets evaluate tqdm

# 可选：安装 Axolotl（配置化训练工具）
pip install axolotl
```

### 最简 LoRA 训练代码

```python
"""
最简 LoRA 微调示例
适合初学者理解完整流程
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ===== 1. 加载模型 =====
model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或其他模型

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 float16 节省显存
    device_map="auto",           # 自动分配设备
)

# ===== 2. 配置 LoRA =====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
    r=16,                           # 低秩维度
    lora_alpha=32,                  # 缩放系数
    lora_dropout=0.05,              # dropout 比率
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj",     # MLP层（可选）
    ],
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== 3. 准备数据 =====
# 示例数据
train_data = [
    {"instruction": "什么是 Python？", "output": "Python 是一门高级编程语言..."},
    {"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支..."},
    # ... 更多数据
]

def format_example(example):
    """格式化单个样本"""
    prompt = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    return {"text": prompt}

# 转换为 Dataset
dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_example)

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

# ===== 4. 配置训练参数 =====
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=4,   # 每设备批次大小
    gradient_accumulation_steps=4,   # 梯度累积（模拟更大批次）
    learning_rate=2e-4,              # 学习率
    fp16=True,                       # 使用 fp16
    logging_steps=10,                # 日志频率
    save_steps=100,                  # 保存频率
    save_total_limit=3,              # 最多保存3个checkpoint
)

# ===== 5. 创建训练器 =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

# ===== 6. 开始训练 =====
trainer.train()

# ===== 7. 保存模型 =====
model.save_pretrained("./my_lora_model")
tokenizer.save_pretrained("./my_lora_model")

print("训练完成！LoRA 权重已保存到 ./my_lora_model")
```

### QLoRA 版本（显存优化）

```python
"""
QLoRA 微调：使用 4-bit 量化，大幅降低显存需求
适合显存 <16GB 的 GPU
"""

from transformers import BitsAndBytesConfig

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",       # 量化类型
    bnb_4bit_compute_dtype=torch.float16,  # 计算精度
    bnb_4bit_use_double_quant=True,  # 双量化（进一步节省）
)

# 加载量化模型（7B 只需要 ~4GB 显存）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 准备模型以支持梯度计算
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# 后续 LoRA 配置和训练步骤相同...
```

---

## 26.6 使用 Axolotl 简化训练

Axolotl 是一个配置化的训练工具，无需编写代码。

### 安装

```bash
pip install axolotl
```

### 配置文件

```yaml
# config.yml - Axolotl 配置文件

# ===== 模型配置 =====
base_model: Qwen/Qwen2.5-7B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

# ===== LoRA 配置 =====
lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj

# ===== 数据配置 =====
datasets:
  - path: ./train_data.json
    type: alpaca  # 数据格式

# ===== 训练配置 =====
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
warmup_ratio: 0.03

# ===== 硬件配置 =====
bf16: false  # 使用 fp16
fp16: true
gradient_checkpointing: true  # 节省显存

# ===== 输出配置 =====
output_dir: ./output
save_steps: 100
save_total_limit: 3

# ===== 可选：QLoRA =====
load_in_4bit: true  # 启用 QLoRA
```

### 运行训练

```bash
# 使用配置文件训练
accelerate launch -m axolotl.cli config.yml

# 或者直接运行
python -m axolotl.cli config.yml
```

---

## 26.7 训练数据准备实战

### 从现有对话创建训练数据

```python
"""
将对话记录转换为训练数据
"""

import json

def conversation_to_alpaca(conversations):
    """将对话转换为 Alpaca 格式"""
    training_data = []
    
    for conv in conversations:
        # 提取用户问题
        user_msg = ""
        assistant_msg = ""
        
        for msg in conv["messages"]:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]
        
        if user_msg and assistant_msg:
            training_data.append({
                "instruction": user_msg,
                "input": "",
                "output": assistant_msg,
            })
    
    return training_data

# 示例：从对话日志创建
conversations = [
    {
        "messages": [
            {"role": "user", "content": "什么是 RAG？"},
            {"role": "assistant", "content": "RAG 是检索增强生成..."},
        ]
    },
    # ... 更多对话
]

data = conversation_to_alpaca(conversations)

# 保存
with open("train_data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### 使用 LLM 生成训练数据

```python
"""
用 LLM 自动生成训练数据（数据增强）
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_training_data(topic: str, num_samples: int = 10):
    """生成特定主题的训练数据"""
    
    prompt = f"""请生成 {num_samples} 条关于"{topic}"的问答数据。
    
格式要求（JSON）：
[
  {{
    "instruction": "问题",
    "input": "",
    "output": "答案"
  }}
]

要求：
- 问题要有多样性
- 答案要准确、专业
- 答案长度 100-300 字"""

    response = llm.invoke(prompt)
    
    # 解析 JSON
    import re
    json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    
    return []

# 使用
data = generate_training_data("金融知识", num_samples=20)

with open("finance_train_data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### 数据验证脚本

```python
"""
验证训练数据质量
"""

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
        if len(item.get("instruction", "")) < 5:
            issues.append(f"第{i}条：instruction 过短")
        
        if len(item.get("output", "")) < 20:
            issues.append(f"第{i}条：output 过短")
        
        if len(item.get("output", "")) > 2000:
            issues.append(f"第{i}条：output 过长")
    
    print(f"总数据: {len(data)} 条")
    print(f"问题数: {len(issues)} 个")
    
    if issues:
        for issue in issues[:10]:  # 只显示前10个
            print(f"  - {issue}")
    
    return len(issues) == 0

# 使用
validate_dataset("train_data.json")
```

---

## 26.8 训练监控与评估

### 训练过程监控

```python
# 在训练代码中添加监控

from transformers import TrainerCallback

class MonitorCallback(TrainerCallback):
    """训练监控回调"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时"""
        if logs:
            loss = logs.get("loss", 0)
            lr = logs.get("learning_rate", 0)
            epoch = state.epoch
            
            print(f"Epoch {epoch:.2f} | Loss: {loss:.4f} | LR: {lr:.6f}")

# 添加到训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    callbacks=[MonitorCallback()],
)

# 训练过程中可以看到：
# Epoch 0.50 | Loss: 2.3456 | LR: 0.000200
# Epoch 1.00 | Loss: 1.8923 | LR: 0.000180
# Epoch 1.50 | Loss: 1.4567 | LR: 0.000150
# ...
```

### 使用 TensorBoard

```bash
# 在训练参数中添加
training_args = TrainingArguments(
    logging_dir="./logs",  # TensorBoard 日志目录
)

# 启动 TensorBoard
tensorboard --logdir ./logs

# 打开浏览器查看 http://localhost:6006
```

### Loss 曲线解读

```
理想 Loss 曲线：
├── 快速下降：模型正在学习
├── 平缓下降：接近最优
└── 稳定：训练完成

异常情况：
├── Loss 不下降：学习率太小或数据有问题
├── Loss 波动大：学习率太大
├── Loss 上升：过拟合或数据冲突
├── Loss = 0：模型崩溃（检查数据）
```

### 训练后评估

```python
"""
评估微调效果
"""

import torch

def evaluate_model(model, tokenizer, test_cases):
    """评估模型"""
    results = []
    
    for case in test_cases:
        prompt = f"<|im_start|>user\n{case['question']}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        answer = response.split("assistant\n")[-1].strip()
        
        results.append({
            "question": case["question"],
            "expected": case["expected"],
            "actual": answer,
            "match": case["expected"] in answer,
        })
    
    # 计算准确率
    accuracy = sum(r["match"] for r in results) / len(results)
    
    print(f"准确率: {accuracy:.2%}")
    
    return results

# 使用
test_cases = [
    {"question": "什么是 Python？", "expected": "编程语言"},
    {"question": "什么是机器学习？", "expected": "人工智能"},
]

results = evaluate_model(model, tokenizer, test_cases)
```

---

## 26.9 模型合并与部署

### LoRA 权重合并

```python
"""
将 LoRA 权重合并到基座模型
合并后可以像普通模型一样使用
"""

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    "./my_lora_model",  # LoRA 权重路径
)

# 3. 合并权重
model = model.merge_and_unload()

# 4. 保存合并后的完整模型
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

print("合并完成！模型保存在 ./merged_model")
```

### 合并 vs 不合并

| 方式 | 优点 | 缺点 |
|------|------|------|
| **不合并（动态加载）** | 可以切换多个 LoRA | 推理稍慢 |
| **合并** | 推理速度正常 | 只能用一个 LoRA |

### 动态加载 LoRA

```python
"""
推理时动态加载 LoRA
适合需要切换不同 LoRA 的场景
"""

from peft import PeftModel

# 加载基座模型（只加载一次）
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 推理时加载不同的 LoRA
def inference_with_lora(prompt, lora_path):
    # 动态加载 LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 推理
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    
    return tokenizer.decode(outputs[0])

# 使用不同的 LoRA
result1 = inference_with_lora("问题", "./lora_customer_service")
result2 = inference_with_lora("问题", "./lora_technical")
```

### 推理示例

```python
"""
使用微调后的模型进行推理
"""

def chat(model, tokenizer, user_input: str):
    """对话推理"""
    
    # 构建 prompt
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取回答
    answer = response.split("assistant\n")[-1].strip()
    
    return answer

# 使用
model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

answer = chat(model, tokenizer, "什么是机器学习？")
print(answer)
```

---

## 26.10 常见问题与解决方案

### 显存不足

```python
# 解决方案1：使用 QLoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# 解决方案2：减小 batch size
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # 减小批次
    gradient_accumulation_steps=16,     # 增加累积
)

# 解决方案3：使用梯度检查点
model.gradient_checkpointing_enable()

# 解决方案4：减小 LoRA rank
lora_config = LoraConfig(r=8)  # 从 16 减到 8
```

### Loss 不下降

```python
# 问题：Loss 一直不下降或波动大

# 解决方案1：调整学习率
training_args = TrainingArguments(learning_rate=1e-5)  # 减小学习率

# 解决方案2：检查数据质量
# 确保数据准确、一致、无冲突

# 解决方案3：增加 warmup
training_args = TrainingArguments(warmup_ratio=0.1)  # 10% warmup

# 解决方案4：使用不同的优化器
training_args = TrainingArguments(optimizer="adamw_torch")
```

### 过拟合

```python
# 问题：训练数据表现好，但实际应用表现差

# 解决方案1：增加 dropout
lora_config = LoraConfig(lora_dropout=0.1)  # 增加 dropout

# 解决方案2：减少训练轮数
training_args = TrainingArguments(num_train_epochs=2)  # 减少轮数

# 解决方案3：增加数据量
# 收集更多多样化的数据

# 解决方案4：使用验证集
trainer = Trainer(
    eval_dataset=valid_dataset,  # 添加验证集
)
```

### 回答格式不对

```python
# 问题：微调后回答格式仍然不符合预期

# 解决方案：在数据中强化格式

# 训练数据示例（强化格式）
{
    "instruction": "请列出 Python 的优点",
    "output": """Python 的优点如下：

1. 简洁易学：语法简单，适合初学者
2. 功能强大：丰富的库和框架
3. 应用广泛：Web、AI、数据分析等

以上是 Python 的主要优点。"""
}

# 在每条数据中使用相同格式，模型会学习这个模式
```

---

## 26.11 实战案例：客服对话微调

### 场景描述

训练一个客服对话模型，要求：
- 回答礼貌专业
- 使用固定格式
- 包含特定术语

### 数据准备

```python
"""
准备客服对话训练数据
"""

customer_service_data = [
    {
        "instruction": "客户问：订单迟迟未发货怎么办？",
        "output": """感谢您的咨询！关于订单发货问题，我来为您解答：

1. 请先确认订单状态，可在"我的订单"页面查看
2. 若显示"待发货"，说明正在处理中，请耐心等待
3. 若超过承诺发货时间，请联系客服处理

如有其他问题，随时联系我们！感谢您的支持！"""
    },
    {
        "instruction": "客户问：产品有质量问题怎么退货？",
        "output": """感谢您的反馈！关于退货流程，我来为您说明：

1. 登录账户，进入"售后服务"页面
2. 选择问题类型："产品质量问题"
3. 上传问题描述和照片作为凭证
4. 提交申请，等待审核（1-3工作日）

审核通过后，我们会安排退货事宜。感谢您的理解！"""
    },
    # ... 更多数据（建议至少 2000 条）
]

# 保存
import json
with open("customer_service_train.json", "w", encoding="utf-8") as f:
    json.dump(customer_service_data, f, indent=2, ensure_ascii=False)
```

### 配置文件

```yaml
# customer_service_config.yml

base_model: Qwen/Qwen2.5-7B-Instruct

lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

datasets:
  - path: ./customer_service_train.json
    type: alpaca

num_epochs: 3
batch_size: 4
learning_rate: 2e-4

output_dir: ./customer_service_lora
```

### 训练运行

```bash
accelerate launch -m axolotl.cli customer_service_config.yml
```

### 测试验证

```python
def test_customer_service():
    """测试客服模型"""
    
    test_cases = [
        "订单发货问题",
        "产品质量退货",
        "退款流程",
        "会员优惠",
    ]
    
    for case in test_cases:
        prompt = f"<|im_start|>user\n客户问：{case}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("assistant\n")[-1].strip()
        
        print(f"问题: {case}")
        print(f"回答: {answer}")
        print("---")

test_customer_service()
```

---

## 26.12 最佳实践总结

### 微调决策流程

```
1. 需求分析
   ├── Prompt Engineering 能解决吗？→ 尝试后再决定
   ├── RAG 能解决吗？→ 尝试后再决定
   └── 确定需要改变模型行为 → 微调

2. 方案选择
   ├── 简单任务、极低成本 → Prompt Tuning
   ├── 多任务需要切换 → Adapter
   ├── 显存充足（>16GB）→ LoRA
   ├── 显存有限（<16GB）→ QLoRA
   └── 企业级大规模定制 → 全量微调

3. 数据准备
   ├── 确定数据格式
   ├── 收集高质量数据（>2000条）
   └── 数据清洗和验证

4. 训练配置
   ├── LoRA rank: 16-32
   ├── Adapter size: 64-256
   ├── Prompt Tuning tokens: 20-100
   ├── Learning rate: 1e-4 ~ 5e-4
   └── Epochs: 2-5

5. 监控评估
   ├── 观察 Loss 曲线
   ├── 测试集评估
   └── 人工抽样检查

6. 部署应用
   ├── 合并权重（LoRA）
   ├── 或动态加载（Adapter）
   └── 集成到应用
```

### 检查清单

| 检查项 | 说明 |
|--------|------|
| **数据质量** | 准确、一致、无噪声 |
| **数据数量** | 建议 >2000 条 |
| **格式一致** | 所有数据使用相同格式模板 |
| **学习率** | LoRA 推荐 2e-4 |
| **训练轮数** | 2-5 轮，避免过拟合 |
| **验证集** | 保留部分数据用于验证 |
| **Loss 监控** | 观察 Loss 曲线是否正常下降 |
| **人工测试** | 抽样检查实际效果 |

### 避坑指南

```
常见错误：
├── 数据太少 → 效果不稳定
├── 数据质量差 → 学到错误行为
├── 格式不一致 → 模型困惑
├── 训练轮数太多 → 过拟合
├── 学习率太大 → Loss 波动
├── 学习率太小 → Loss 不下降
├── 没有验证 → 不知道效果
└── 期望过高 → 微调不是万能药
```

---

## 26.13 本章小结

- **微调**是让预训练模型适应特定任务的核心技术
- **LoRA**是最推荐的方案：参数少、显存低、效果好，适合大多数场景
- **QLoRA**适合显存受限场景，用 4-bit 量化进一步节省资源
- **Adapter**适合多任务切换场景，一个基座模型服务多个业务线
- **Prompt Tuning**成本最低，适合简单任务的快速定制
- **数据质量**比数量更重要，建议 >2000 条高质量数据
- **Alpaca 格式**是最常用的数据格式
- **Axolotl**简化训练流程，配置文件即可训练
- 训练时监控 Loss 曲线，异常时调整学习率或数据
- 微调后合并权重，部署时像普通模型一样使用
- 先尝试 Prompt Engineering 和 RAG，不足时再考虑微调
- 微调改变模型行为，但不是万能的，需理性评估需求

### 方案选择速查表

| 需求场景 | 推荐方案 |
|---------|---------|
| 一般定制任务 | LoRA |
| 显存不足（<16GB）| QLoRA |
| 多业务线共用模型 | Adapter |
| 简单任务、极低成本 | Prompt Tuning |
| 大规模企业定制 | 全量微调 |