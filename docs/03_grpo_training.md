# GRPO 强化学习训练指南

## 1. 为什么是 GRPO？

### 1.1 从 SFT 到 RLHF
SFT (监督微调) 只能让模型学会"怎么说话"，但也存在局限：
- **静态目标**：无法动态评估回答的好坏。
- **主观性**：依赖人工标注文案，难以捕捉复杂的偏好。
- **缺乏探索**：模型不敢尝试新的、可能更好的策略。

因此，我们需要 RLHF (基于人类反馈的强化学习) 来实现 **有用性**、**安全性** 和 **诚实性** 的对齐。

### 1.2 GRPO vs PPO
传统的 PPO (Proximal Policy Optimization) 算法虽然经典，但在大模型时代面临挑战。本项目采用 DeepSeek 团队提出的 **GRPO (Grouped Relative Policy Optimization)**。

| 特性 | PPO (传统) | GRPO (本项目) | 优势 |
|------|-----------|--------------|------|
| **架构** | Actor-Critic (双网络) | Actor Only (单网络) | **显存节省 50%**，训练更稳 |
| **基线估计**| 单独的 Value Network | **组内平均 (Group Average)** | 无需训练 Critic，自适应基线 |
| **比较机制**| 绝对价值评估 | **相对优势比较** | 相对比较更符合人类直觉，噪声更小 |
| **采样效率**| 单样本学习 | **成组学习** | 信息利用率更高 |

**核心原理**：GRPO 不训练 Critic 网络，而是对同一个 Prompt 采样一组回答 $\{o_1, o_2, ..., o_G\}$，计算它们的奖励 $\{r_1, r_2, ..., r_G\}$。然后，利用这组奖励的均值作为基线，计算每个回答的相对优势。

---

## 2. 训练架构

我们使用 **HuggingFace TRL + DeepSpeed** 框架。这是 DeepSeek-R1 的官方算法实现方案。

- **算法库**: `trl.GRPOTrainer`
- **分布式**: `DeepSpeed ZeRO-2/3`
- **环境**: 支持多机多卡训练

### 2.1 训练流程
1. **输入**: Prompt 数据集（如金融新闻标题）。
2. **采样**: 策略模型 (Policy) 生成 $G$ 个候选回答 (Group Size)。
3. **打分**: 奖励模型 (Reward Model) 对 $G$ 个回答打分。
4. **更新**: 计算组内优势，更新策略模型参数，最大化均分并约束 KL 散度。

---

## 3. 数据准备 (Information Theory Guided)

基于信息论 $I(P;Q)$ 最大化原则：**Prompt (P) 应该包含足够的信息，以减少回答质量 (Q) 的不确定性。**

### 3.1 Prompt 设计原则
- **背景明确**：提供足够的时间、事件背景。
- **数据翔实**：包含具体的财务指标。
- **任务清晰**：明确要求分析什么（如"分析业务结构变化"）。

### 3.2 数据结构
```json
{
  "prompt": [
    {"role": "user", "content": "苹果公司2024年Q1营收1195亿美元...基于这些数据，分析业务结构变化..."}
  ]
}
```

---

## 4. 训练配置

**关键超参数** (`src/grpo_financial_tuning/configs/config.json`):
```yaml
training:
  learning_rate: 1e-6        # RL 阶段 LR 通常极低
grpo:
  num_generations: 4         # Group Size，每组生成多少个回答
  max_completion_length: 1024 # 生成长度 (覆盖长思维链)
  beta: 0.0                  # KL 惩罚系数 (Beta)，此处设为0表示主要依赖 Epsilon Clipping
  temperature: 0.7           # 平衡探索与利用
```

---

## 5. 常见问题

**Q: 为什么 Loss 是负的？**
A: 在 RLHF 中，Objective 是最大化 Reward，而在深度学习框架中通常是最小化 Loss。通常 `Loss = -Reward`，所以 Loss 为负且绝对值变大通常意味着 Reward 在增加（是好事）。但具体要看 TensorBoard 中的 `reward/mean` 指标。

**Q: 显存不够怎么办？**
A: 
1. 减小 `num_generations` (例如从 8 减到 4)。
2. 减小 `max_completion_length`。
3. 启用 DeepSpeed ZeRO-3 或 Offload。
4. 使用 QLoRA 训练。

**Q: 模型开始胡言乱语 (Gibberish)？**
A: 可能是 KL 惩罚系数 (`lam`) 太小，或者 Reward Model 被 hack 了（过度优化）。增大 `lam` 或检查 Reward Model 的评分分布。
