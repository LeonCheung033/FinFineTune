# 金融大模型微调项目简历 (FinanceTuning)

> 💡 提示：此文档包含基于通过项目文档提取的详细技术实现与量化成果，经过润色后的 **中文版** 与 **英文版** 简历段落，可直接用于简历的"项目经历"部分。

---

## 🇨🇳 中文版 (Chinese Version)

### **金融领域大模型全链路微调与强化学习对齐 (FinanceTuning) | 核心开发者**
**项目背景**：针对通用大模型在金融场景中存在的“幻觉严重、风险意识匮乏、专业度不足”等痛点，主导开发了基于 DeepSeek 与 Llama 架构的金融垂类专家模型。通过 SFT、Reward Modeling、GRPO 强化学习及量化部署的全流程优化，显著提升了模型在金融资讯解读、风险提示及逻辑推理上的表现，实现了生产环境的高效落地。

**核心职责**：负责从数据工程、监督微调、奖励模型构建到强化学习对齐的全链路技术方案设计与实施，以及最终的模型压缩与性能调优。

**主要工作与技术实现**：
*   **深度定制数据与高效微调**：设计“分层随机采样”策略，基于 *Financial Services News* 清洗构建 **1.5w+** 条高质量指令数据，首创 `<think>` 标签引导的 **CoT（思维链）数据增强方案**，显著增强模型逻辑深度；基于 **DeepSeek-R1-Distill-Qwen-7B**，采用 **LoRA** (Rank=32, Alpha=64) 结合 **DeepSpeed** 进行高效微调，在仅更新 <1% 参数的情况下实现金融领域知识的高效注入与推理能力跃升。
*   **多维度奖励模型体系构建**：基于 **Skywork-Reward-Llama-3.1-8B** 构建判别模型，创新性设计涵盖准确性、逻辑性及合规性的 **5层质量分级体系** (Low 至 High)；生成 pairwise 偏好数据并采用 **Freeze Tuning** 策略（仅训练最后 4 层及 Score Head），有效解决灾难性遗忘问题并降低显存开销。
*   **GRPO 强化学习对齐**：摒弃传统 PPO 架构，采用 **GRPO** 算法，利用 Group Average 作基线省去 Critic 网络，**降低显存占用 50%** 且训练更稳定；通过多维奖励引导模型在“有用性”与“安全性”间取得平衡，确保模型主动输出风险提示并严守合规底线。
*   **模型量化与高效部署**：基于 **GPTQ INT4** 算法进行模型量化，构建融合 SFT 基础数据与 GRPO 场景数据的**校准集 (Calibration Data)**，确保量化过程中权重对敏感特征的保留；实现了模型体积与显存占用的极致压缩，支持在消费级显卡上进行高吞吐推理。

**项目成果**：
*   **领域理解力跃升**：SFT 阶段模型困惑度 (Perplexity) 从 13.0 降至 **2.8**，专业知识覆盖率从 0.72 提升至 **0.88**。
*   **判别准确率突破**：奖励模型在金融偏好测试集上的准确率从 70% 提升至 **97%**，为 RL 阶段提供了精准的优化方向。
*   **生成质量显著改善**：经过 GRPO 对齐，模型在内部逻辑与合规性评分中，平均分从 12 分提升至 **25 分**，完全消除了违规投资建议。
*   **极致性能优化**：量化后模型体积减少 **~60%** (12GB → 5GB)，推理显存占用降低 **15GB**，在保持 **~90%** 语义准确度的前提下，大幅降低了线上部署成本。

---

## 🇺🇸 英文版 (English Version)

### **End-to-End Financial LLM Fine-Tuning & Alignment (FinanceTuning) | Core Developer**
**Background**: To address issues like hallucinations, lack of risk awareness, and insufficient domain expertise in general LLMs within financial contexts, I co-led the development of a vertical financial expert model based on DeepSeek and Llama architectures. By implementing a full pipeline including SFT, Reward Modeling, GRPO Reinforcement Learning, and Quantization, we significantly enhanced the model's capabilities in financial news interpretation, risk warning, and logical reasoning.

**Core Responsibilities**: Designed and implemented the technical roadmap covering data engineering, Supervised Fine-Tuning (SFT), Reward Model construction, RL alignment, and final model compression.

**Key Implementations**:
*   **Financial Data Engineering & SFT**:
    *   Designed a "Stratified Random Sampling" strategy to construct a high-quality instruction dataset (15k+ samples) derived from *Financial Services News*.
    *   Pioneered a **Chain-of-Thought (CoT) Data Augmentation** scheme, introducing `<think>` tags to explicitly guide the model through background analysis, data interpretation, and risk assessment, enhancing interpretability.
    *   Fine-tuned the **DeepSeek-R1-Distill-Qwen-7B** base model using **LoRA** (Rank=32, Alpha=64) and **DeepSpeed**, enabling efficient injection of domain knowledge while updating <1% of parameters.
*   **Multi-Dimensional Reward Modeling**:
    *   Built a discriminator based on **Skywork-Reward-Llama-3.1-8B**, designing a novel **5-Level Quality Grading System** (Low to High) instead of binary classification, evaluating accuracy, logic, compliance, and risk awareness.
    *   Generated pairwise preference data and adopted **Freeze Tuning** (freezing the first 28 layers, training only the last 4 Transformer layers & Score Head) to mitigate catastrophic forgetting and reduce memory overhead.
*   **GRPO Reinforcement Learning Alignment**:
    *   Adopted **GRPO (Grouped Relative Policy Optimization)**, discarding the traditional PPO Critic network. By using Group Average as the baseline, we achieved **50% GPU memory savings** and greater training stability.
    *   Aligned the model to balance "Helpfulness" and "Safety," ensuring it proactively provides risk warnings and strictly adheres to compliance boundaries in investment advice.
*   **Model Quantization & Deployment**:
    *   Performed **GPTQ INT4** quantization using a custom **Calibration Dataset** (fusing SFT data & GRPO prompts) to preserve sensitivity in critical weights.
    *   Achieved extreme compression for deployment on consumer-grade GPUs without sacrificing performance.

**Key Achievements**:
*   **Domain Mastery**: Reduced SFT Perplexity from 13.0 to **2.8**; increased domain knowledge coverage from 0.72 to **0.88**.
*   **Reward Accuracy**: Boosted Reward Model accuracy on financial preference test sets from 70% to **97%**, providing precise guidance for RL.
*   **Quality Improvement**: Post-GRPO, the model's average score in logic and compliance evaluations rose from 12 to **25**, eliminating non-compliant investment advice.
*   **Performance Optimization**: Reduced model size by **~60%** (12GB → 5GB) and inference memory usage by **15GB**, maintaining **~90%** semantic accuracy for efficient production deployment.
