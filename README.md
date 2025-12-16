# 🏦 FinanceTuning - 金融领域大模型微调全流程

<p align="center">
  <b>金融领域私有数据专家模型微调解决方案</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#项目特性">项目特性</a> •
  <a href="#项目架构">项目架构</a> •
  <a href="#文档">文档</a> •
  <a href="#贡献指南">贡献指南</a>
</p>

---

## 📖 项目简介

FinanceTuning 是一个完整的金融领域大语言模型微调框架，涵盖从监督微调（SFT）到强化学习（GRPO）再到模型量化部署的全流程。本项目基于真实企业场景设计，提供了详细的理论说明和可复现的代码实现。

### 核心目标

- 🎯 将通用大模型转变为金融专家模型
- 🔒 确保模型输出符合金融监管要求（不给出明确投资建议）
- 📊 提供专业、准确、通俗易懂的金融分析
- ⚡ 支持多机多卡分布式训练与高效部署

## ✨ 项目特性

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| **SFT 监督微调** | 金融领域知识注入 | LoRA, DeepSpeed, Transformers |
| **奖励模型训练** | 人类偏好对齐 | Freeze Tuning (Last 4 Layers), 5层质量分级 |
| **GRPO 强化学习** | 策略优化 | TRL GRPOTrainer, vLLM |
| **模型量化** | 高效部署 | GPTQ Model, INT4 |

## 🏗️ 项目架构

```
financeTuning/
├── 📚 docs/                    # 详细文档
├── ⚙️ configs/                 # 配置文件
├── 📦 src/                     # 源代码
│   ├── sft/                    # SFT 微调模块
│   ├── reward_model/           # 奖励模型模块
│   ├── grpo_financial_tuning/  # GRPO 训练模块
│   └── gptq_model/             # 模型量化模块
├── 📝 scripts/                 # SFT 训练脚本

```

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- CUDA >= 12.1
- PyTorch >= 2.0

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/financeTuning.git
cd financeTuning

# 创建虚拟环境
conda create -n finance-tuning python=3.10
conda activate finance-tuning

# 安装依赖
pip install -r requirements.txt

# 可选：安装为可编辑包
pip install -e .
```

### 训练流程

```bash
# 1. SFT 监督微调
bash scripts/sft/train_sft.sh

# 2. 奖励模型训练
bash src/reward_model/financial_reward_model/scripts/training/run_training.sh

# 3. GRPO 强化学习
bash src/grpo_financial_tuning/run_training.sh

# 4. 模型量化 (请先修改脚本中的路径配置)
python src/gptq_model/quantize_model.py
```

## 📚 文档

| 文档 | 描述 |
|------|------|
| [🎯 执行指南](docs/EXECUTION_GUIDE.md) | **完整的项目执行流程和产出说明** |
| [SFT 监督微调](docs/01_sft_tuning.md) | LoRA 微调原理与实践 |
| [奖励模型训练](docs/02_reward_model.md) | 偏好数据构建与模型训练 |
| [GRPO 强化学习](docs/03_grpo_training.md) | GRPO 算法与分布式训练 |
| [模型量化](docs/04_quantization.md) | GPTQ 量化与部署 |

## 🤖 预训练模型

项目已包含各阶段的预训练模型，可直接使用：

| 阶段 | 模型文件 | 大小 |
|------|----------|------|
| Step 1: SFT | `models/sft/*.zip` | 12.77 GB |
| Step 2: 奖励模型 | `models/reward_model/*.zip` | 11.04 GB |
| Step 3: GRPO | `models/grpo/*.zip` | 11.27 GB |
| Step 4: 量化 | `models/quantized/*.zip` | 4.03 GB |

详见 [models/README.md](models/README.md)

## 🔧 支持的模型

本项目支持以下基座模型：

- DeepSeek-R1-Distill-Qwen-7B（推荐）
- Qwen 系列
- LLaMA 系列
- 其他 HuggingFace 兼容模型

## 📊 训练效果

| 阶段 | 指标 | 效果 |
|------|------|------|
| SFT | 专业知识覆盖率 | 0.72 → 0.88 |
| 奖励模型 | 偏好预测准确率 | 85%+ |
| GRPO | 安全性对齐 | 显著提升 |
| 量化 | 模型大小 | 减少 75% |

## 🤝 贡献指南

欢迎贡献代码、文档或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📄 开源协议

本项目采用 [Apache 2.0](LICENSE) 协议开源。

## 🙏 致谢

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [TRL](https://github.com/huggingface/trl)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [vLLM](https://github.com/vllm-project/vllm)

---

<p align="center">
  如果这个项目对你有帮助，请给一个 ⭐ Star！
</p>
