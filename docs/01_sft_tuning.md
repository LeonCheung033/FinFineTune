# SFT 监督微调指南
详细文档参考：https://ja638ntz3wj.feishu.cn/wiki/CO8swPrWRizoCckWmy6cNPBwnVd?from=from_copylink
## 1. 项目背景与目标

### 1.1 项目愿景
构建一个专业级的金融AI助手，旨在帮助普通投资者理解财经新闻并提供投资参考。

### 1.2 核心需求
1. **准确解读**：分析财经新闻背后的含义及市场影响。
2. **风险提示**：针对投资组合提供个性化的风险预警。
3. **通俗解释**：用易懂语言解释复杂金融概念。
4. **合规边界**：**严禁提供明确的"买入/卖出"建议**，保持客观中立。
5. **异常识别**：识别市场异常波动并给出合理解释。

### 1.3 技术路线
采用 **两阶段微调策略**：
1. **SFT阶段**：构建金融领域基础能力（知识注入）。
2. **RLHF/GRPO阶段**：优化用户体验和安全边界（对齐人类偏好）。

---

## 2. 模型选型

经过对推理增强型、知识密集型、领域专精型和多模态模型的对比分析，本项目选用 **DeepSeek-R1-Distill-Qwen-7B**。

### 2.1 选择理由
- **推理能力强**：DeepSeek-R1 系列在数学和逻辑推理任务上表现优异，非常适合金融分析中的因果推导和风险评估。
- **资源效率高**：7B 参数量蒸馏模型，在保持强大推理能力的同时，大大降低了部署和训练成本。
- **CoT 能力**：经过链式思考（Chain-of-Thought）训练，能给出结构化的分析过程，增加可解释性。

---

## 3. 微调技术选型：LoRA

在对比了 LoRA, QLoRA, P-Tuning v2, Prefix-Tuning, Adapter Tuning 后，本项目推荐使用 **LoRA (Low-Rank Adaptation)**。

### 3.1 为什么选择 LoRA？
1. **保留推理能力**：通过冻结原模型权重，最大程度保留了预训练模型的数学和逻辑推理能力。
2. **知识注入平衡**：LoRA 在低秩子空间中能有效捕获特定领域的知识分布。
3. **训练效率**：训练速度比全量微调快 3-5 倍，显存占用大幅降低。

### 3.2 推荐参数 (本项目实际配置)
- **Rank (r)**: `32` (在显存和效果间取得平衡)
- **Alpha**: `64` (通常为 rank 的 2 倍)
- **Target Modules**: 覆盖所有 Linear 层 (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Dropout**: `0.15`
- **Bias**: `none`

---

## 4. 数据构建

### 4.1 数据来源

**主要数据集 (本项目使用)**

本项目 SFT 阶段核心使用以下数据集构建基础金融知识能力：

| 数据集名称 | HuggingFace ID | 描述 |
|---|---|---|
| **Financial Services News** | `gunnybd01/Financial_Services_News_smr` | 包含大量金融服务领域的新闻报道，适用于训练模型对金融事件的理解和摘要能力。 |

**获取与下载**
您可以直接使用 `datasets` 库下载。建议指定 `cache_dir` 以避免占用系统盘空间：

```python
from datasets import load_dataset

# 下载数据集到指定缓存目录
ds = load_dataset("gunnybd01/Financial_Services_News_smr", cache_dir="./data/cache")
```

**其他参考数据集**
还可以用以下数据作为补充来源：
*   **国际新闻**: [`ashraq/financial-news-articles`](https://huggingface.co/datasets/ashraq/financial-news-articles)
*   **投资决策**: [`ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset`](https://huggingface.co/datasets/ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset)
*   **理财顾问**: [`nihiluis/financial-advisor-100`](https://huggingface.co/datasets/nihiluis/financial-advisor-100)


### 4.2 数据格式 (JSONL)

```json
{
  "messages": [
    {"role": "user", "content": "通俗解释一下什么是'量化宽松'？"},
    {"role": "assistant", "content": "<think>\n用户需要通俗解释，应避免过多专业术语，使用类比法...\n1. 定义：央行印钱买债券\n2. 目的：向市场注水，增加流动性\n3. 类比：就像给干旱的农田灌水\n</think>\n\n量化宽松（QE）简单来说，就是央行通过'印钱'来购买国债等资产，从而向市场注入大量资金...\n\n就好比干旱时，水库（央行）开闸放水，让农田（市场）有水灌溉，庄稼（企业）才能生长..."}
  ]
}
```

### 4.3 数据处理流程
1. **加载与过滤**：加载 Arrow 数据，并执行长度过滤（`len(Article) < 5000`），确保符合模型上下文窗口限制。
2. **生成问题 (Phase 1)**：调用 DeepSeek API，基于英文原文生成中文金融问题。提示词要求提取关键信息、数据、事实，并构建无需阅读原文即可回答的详细问题。
3. **生成思考与回答 (Phase 2)**：调用 DeepSeek API，基于生成的问题，生成包含 `<think>` 标签的思考过程和最终专业回答。强制要求 CoT（思维链）分析，包含背景分析、数据解读、原因探究等多维度思考。
4. **分层随机采样**：为了保证数据的代表性同时兼顾随机性，采用 **分层随机采样 (Stratified Random Sampling)** 替代简单的随机采样，确保覆盖不同分布的数据。
5. **格式化**：统一转换为 SFT 训练所需的 User-Assistant 对话格式 (JSONL)。

---

## 5. 训练实施

详情请参考 [训练脚本](../../scripts/train_sft.sh) 和 [执行指南](EXECUTION_GUIDE.md)。

关键超参数建议：
- **Learning Rate**: `5e-6` (微调预训练模型通常使用较小 LR)
- **Batch Size**: `192` (Per Device 3 * Accumulation 16 * 4 GPUs)
- **Epochs**: `70` (金融语料较少，通过多轮次充分训练)
- **Max Length**: `1024` (适配显存限制)

---

## 6. 常见问题 (FAQ)

**Q: 如何评估模型效果？**
A: 
1. **困惑度 (Perplexity)**: 验证集上的 Loss 指标。
2. **专业测试集**: 使用 C-Eval (Finance), CMMLU (Finance) 等子集测试。
3. **人工评估**: 邀请金融从业者对生成结果的准确性、中立性打分。

