"""
FinanceTuning - 金融领域大模型微调框架

模块结构:
- sft: 监督微调模块
- reward_model: 奖励模型训练与推理
- grpo: GRPO 强化学习训练
- quantization: 模型量化与部署
"""

__version__ = "0.1.0"
__author__ = "FinanceTuning Contributors"

from . import sft
from . import reward_model
from . import grpo
from . import quantization

__all__ = [
    "sft",
    "reward_model", 
    "grpo",
    "quantization",
]
