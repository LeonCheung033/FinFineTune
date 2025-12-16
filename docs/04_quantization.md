# 模型量化 (Quantization) 指南

## 1. 为什么需要量化？

### 1.1 背景
随着大模型参数量的爆炸式增长（从 BERT 的 3.4亿 到 GPT-4 的万亿级），计算资源和存储成本成为巨大瓶颈。
量化技术旨在将高精度浮点数（FP16/BF16）转换为低精度整数（INT8/INT4），从而：
- **减小显存占用**：INT4 模型体积仅为 FP16 的 **1/4**（例如 7B 模型从 14GB -> 3.5GB）。
- **提升推理速度**：低精度计算更快，吞吐量更高。
- **降低部署门槛**：使大模型能在消费级显卡甚至 CPU/手机上运行。

---

## 2. 核心概念

### 2.1 量化算法 (Algorithm)
- **GPTQ (Recommended for GPU)**: 基于二阶信息的量化算法，需要校准数据。在 GPU 上推理性能极佳。
- **AWQ**: 激活感知权重量化。对激活值敏感的权重保留高精度。
- **GGUF/llama.cpp (Recommended for CPU)**: 专门为 CPU 推理优化的格式标准，支持 Apple Silicon 加速。

### 2.2 激活数据 (Calibration Data) 的重要性
**"炒菜论"**：
- **权重**就像调料（盐、糖、味精...）。
- **激活数据**就像"试菜"。
- 如果盲目减少所有调料（不使用激活数据），菜会很难吃。
- 如果通过"试菜"发现盐最重要（权重敏感度高），胡椒粉不重要，那么我们可以精细保留盐的量，大幅减少胡椒粉。

**结论**：GPTQ 量化必须使用**真实分布的校准数据**，才能保证模型不仅"变小了"，而且"不傻"。使用金融领域的校准数据，能保留 90-95% 的金融专业能力。

---

## 3. 技术选型决策树

```
1. 部署环境？
   ├── CPU / Mac / 边缘设备 → GGUF (llama.cpp)
   └── GPU (NVIDIA) → 继续下一步
   
2. 使用场景？
   ├── 生产环境 (高并发) → GPTQ (vLLM) 或 AWQ
   ├── 显存极度受限 → INT4 / INT3
   └── 精度要求极高 → INT8 或 FP8
```

本项目推荐：**GPTQ INT4** (平衡了精度、速度和显存占用)。

---

## 4. 实战操作

### 4.1 准备校准数据
我们建议融合 SFT 数据 (基础能力) 和 GRPO 提示词 (强化场景) 来构建校准集：
```bash
python src/gptq_model/calibration_fusion.py \
    --input_files data/sft/deepspeek_sft_dataset_1_1k.jsonl data/grpo/grpo_prompts_dataset_5k.jsonl \
    --output_file src/gptq_model/calibration_data.jsonl \
    --total_samples 512 \
    --ratios 0.7 0.3
```

### 4.2 执行量化
使用 `GPTQModel` 进行量化。请先编辑 `src/gptq_model/quantize_model.py` 中的 `main()` 函数，修改 `model_path` 和 `output_path`：

```python
def main():
    # 配置路径
    model_path = "/shared/grpo_financial_tuning/output/best_model"  # 原始模型路径
    calibration_data_path = "src/gptq_model/calibration_data.jsonl"  # 校准数据路径
    output_path = "./output/best_model_gptq_int4"     # 量化模型输出路径
    # ...
```

然后运行脚本：
```bash
python src/gptq_model/quantize_model.py
```

### 4.3 加载使用
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/quantized/gptq_int4",
    device_map="auto",
    trust_remote_code=True
)
```

---

## 5. 常见问题

**Q: 为什么量化后效果变差很多？**
A: 
1. **校准数据不匹配**：用通用文本校准金融模型会导致性能下降。请确保使用金融数据校准。
2. **量化位宽过低**：INT4 通常是底线，INT3/INT2 精度损失极大。

**Q: GGUF 和 GPTQ 选哪个？**
A: 
- 如果你是 **Mac 用户** 或主要在 **CPU** 上跑，选 **GGUF**。
- 如果你有 **NVIDIA 显卡** 并在服务器上部署，选 **GPTQ**。

**Q: 量化需要训练吗？**
A: PTQ (Post-Training Quantization) 不需要训练，只需要几分钟的"校准"（Calibration）过程。这比 QAT (Quantization-Aware Training) 快得多且资源消耗极小。
