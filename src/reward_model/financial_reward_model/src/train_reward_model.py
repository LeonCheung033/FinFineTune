#!/usr/bin/env python3
"""
Skywork奖励模型微调脚本
基于LlamaForSequenceClassification架构
"""

import os
import sys
import torch
import json
import deepspeed
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
)

# 解决PyTorch weights_only问题的补丁
# 这是为了兼容不同版本的PyTorch，避免加载checkpoint时出现权限错误
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    # 如果是加载checkpoint相关文件，强制设置weights_only=False
    # 这样可以避免某些版本的PyTorch过于严格的安全检查
    if weights_only is True and (
        isinstance(f, str) and ('rng_state' in f or 'checkpoint' in f)
    ):
        weights_only = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                              weights_only=weights_only, **kwargs)

torch.load = patched_torch_load

# 添加项目路径到Python搜索路径
# 这样可以导入项目中的自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入项目自定义模块
from data.dataset import create_reward_dataset, create_data_collator


@dataclass
class FinetuningArguments:
    """
    微调参数配置类
    
    这个类定义了freeze tuning的相关参数
    freeze tuning是一种只训练模型部分层的技术，可以节省显存和计算资源
    """
    freeze_trainable_layers: int = field(default=4)        # 可训练的层数（从后往前数）
    freeze_trainable_modules: str = field(default="all")   # 可训练的模块类型
    freeze_extra_modules: Optional[str] = field(default=None)  # 额外的可训练模块


def setup_freeze_tuning(model, finetuning_args: FinetuningArguments):
    """
    设置freeze tuning参数
    
    这个函数实现部分参数微调，只训练模型的最后几层和分类头
    这样可以在保持模型大部分知识的同时，用较少的计算资源进行微调
    
    参数:
        model: 要设置的模型
        finetuning_args: 微调参数配置
    
    工作原理:
        1. 获取模型总层数
        2. 确定哪些层需要训练（通常是最后几层）
        3. 冻结其他层的参数，只训练指定层
        4. 统计可训练参数数量
    """
    print("配置Freeze Tuning...")
    
    # 获取模型的总层数
    # 对于LLaMA模型，这通常是32层
    num_layers = model.config.num_hidden_layers
    
    # 确定可训练层的范围
    # 例如：如果总共32层，freeze_trainable_layers=4，则训练第28-31层（最后4层）
    trainable_layer_ids = range(
        max(0, num_layers - finetuning_args.freeze_trainable_layers), 
        num_layers
    )
    
    # 构建可训练参数的匹配模式
    # 这些模式用于识别哪些参数需要训练
    trainable_patterns = []
    for idx in trainable_layer_ids:
        if finetuning_args.freeze_trainable_modules == "all":
            # 如果是"all"，则该层的所有模块都可训练
            trainable_patterns.append(f".{idx}.")
        else:
            # 否则只训练指定的模块（如attention、mlp等）
            modules = [m.strip() for m in finetuning_args.freeze_trainable_modules.split(",")]
            for module in modules:
                trainable_patterns.append(f".{idx}.{module}")
    
    # 添加额外的可训练模块
    # 例如分类头（score层）通常总是需要训练的
    if finetuning_args.freeze_extra_modules:
        extra_modules = [m.strip() for m in finetuning_args.freeze_extra_modules.split(",")]
        trainable_patterns.extend(extra_modules)
    
    # 遍历模型的所有参数，设置是否可训练
    trainable_params = 0  # 可训练参数数量
    total_params = 0      # 总参数数量
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        # 检查参数名是否匹配可训练模式
        is_trainable = any(pattern in name for pattern in trainable_patterns)
        param.requires_grad = is_trainable  # 设置是否计算梯度
        if is_trainable:
            trainable_params += param.numel()
    
    # 打印配置信息
    print(f"模型层数: {num_layers}")
    print(f"可训练层: {list(trainable_layer_ids)}")
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


class RewardTrainer(Trainer):
    """
    奖励模型训练器
    
    这个类继承自HuggingFace的Trainer，专门用于训练奖励模型
    重写了损失计算和评估方法，实现了pairwise对比学习
    
    奖励模型的核心思想：
    - 输入两个回答（chosen和rejected）
    - 模型分别给出分数
    - 训练目标是让chosen的分数高于rejected的分数
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算奖励模型的对比损失
        
        这是奖励模型训练的核心方法，实现了pairwise对比学习
        
        参数:
            model: 奖励模型
            inputs: 输入数据，包含chosen和rejected的token序列
            return_outputs: 是否返回模型输出
            num_items_in_batch: 批次中的样本数量
        
        返回:
            损失值或(损失值, 模型输出)
        
        工作原理:
            1. 分别对chosen和rejected回答进行推理
            2. 获取两个回答的分数
            3. 计算对比损失：-log(sigmoid(chosen_score - rejected_score))
            4. 这个损失函数鼓励chosen分数高于rejected分数
        """
        # 对chosen回答进行推理
        # chosen回答是人类标注的更好的回答
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )
        # 提取chosen回答的分数
        # squeeze(-1)是为了去掉最后一个维度，得到标量分数
        chosen_rewards = chosen_outputs.logits.squeeze(-1)
        
        # 对rejected回答进行推理
        # rejected回答是人类标注的较差的回答
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )
        # 提取rejected回答的分数
        rejected_rewards = rejected_outputs.logits.squeeze(-1)
        
        # 计算对比损失
        # logsigmoid(chosen - rejected)鼓励chosen分数高于rejected分数
        # 负号是因为我们要最小化损失（最大化chosen相对于rejected的优势）
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        if return_outputs:
            return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """
        重写预测步骤，正确处理自定义数据格式
        
        这个方法在评估时被调用，用于计算验证集上的损失和预测结果
        
        参数:
            model: 模型
            inputs: 输入数据
            prediction_loss_only: 是否只返回损失
            ignore_keys: 忽略的键
        
        返回:
            (损失, 预测结果, 标签)
        """
        model.eval()  # 设置为评估模式
        
        with torch.no_grad():  # 不计算梯度，节省内存
            # 使用自定义的compute_loss方法计算损失
            loss = self.compute_loss(model, inputs)
            
            # 如果只需要损失，直接返回
            if prediction_loss_only:
                return (loss, None, None)
            
            # 计算chosen和rejected的奖励分数
            chosen_outputs = model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"]
            )
            rejected_outputs = model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"]
            )
                
            chosen_rewards = chosen_outputs.logits.squeeze(-1)
            rejected_rewards = rejected_outputs.logits.squeeze(-1)
            
            # 创建预测结果：chosen > rejected 为正确预测
            # 这里将布尔值转换为浮点数（True->1.0, False->0.0）
            predictions = (chosen_rewards > rejected_rewards).float()
            
            # 创建标签：全部为1（chosen应该总是比rejected好）
            # 在奖励模型中，理想情况下chosen总是应该得到更高分数
            labels = torch.ones_like(predictions)
            
            return (loss, predictions, labels)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        重写评估方法，计算奖励模型的准确率
        
        这个方法在验证时被调用，计算模型在验证集上的表现
        
        参数:
            eval_dataset: 评估数据集
            ignore_keys: 忽略的键
            metric_key_prefix: 指标前缀
        
        返回:
            评估结果字典
        
        准确率计算：
            准确率 = 正确预测数量 / 总预测数量
            正确预测 = chosen_score > rejected_score
        """
        # 调用父类评估方法，获取基本的评估结果
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 手动计算准确率
        if eval_dataset is not None:
            correct = 0  # 正确预测的数量
            total = 0    # 总预测数量
            
            self.model.eval()  # 设置为评估模式
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
            with torch.no_grad():  # 不计算梯度
                for batch in eval_dataloader:
                    # 将数据移动到GPU
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}
                    
                    # 分别对chosen和rejected进行推理
                    chosen_outputs = self.model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"]
                    )
                    rejected_outputs = self.model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"]
                    )
                    
                    chosen_rewards = chosen_outputs.logits.squeeze(-1)
                    rejected_rewards = rejected_outputs.logits.squeeze(-1)
                    
                    # 统计正确预测的数量
                    # 如果chosen分数高于rejected分数，则预测正确
                    correct += (chosen_rewards > rejected_rewards).sum().item()
                    total += chosen_rewards.size(0)
            
            # 计算准确率
            accuracy = correct / total if total > 0 else 0.0
            eval_results[f"{metric_key_prefix}_accuracy"] = accuracy
            
            # 打印评估结果
            print(f"评估结果 - Loss: {eval_results[f'{metric_key_prefix}_loss']:.6f}, "
                  f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return eval_results


def main():
    """
    主训练函数
    
    这是整个训练脚本的入口点，负责：
    1. 解析命令行参数
    2. 初始化分布式训练环境
    3. 加载模型和数据
    4. 配置训练参数
    5. 开始训练
    6. 保存最终模型
    """
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Skywork奖励模型微调")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从checkpoint恢复训练")
    parser.add_argument("--config", type=str, default="configs/training/config.json",
                       help="配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="DeepSpeed本地rank")
    parser.add_argument("--deepspeed", type=str, default=None,
                       help="DeepSpeed配置文件")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化DeepSpeed分布式训练环境
    # DeepSpeed是微软开发的深度学习优化库，支持大模型训练
    deepspeed.init_distributed()
    
    # 获取分布式训练的相关信息
    # 这些环境变量由DeepSpeed或其他分布式训练框架设置
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank if args.local_rank != -1 else 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 总进程数
    rank = int(os.environ.get('RANK', 0))              # 当前进程的全局rank
    
    print(f"启动进程 {rank}/{world_size} (GPU {local_rank})")
    
    # 如果指定了checkpoint，打印恢复信息
    if args.resume_from_checkpoint:
        print(f"从checkpoint恢复训练: {args.resume_from_checkpoint}")
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)
    
    # 加载训练配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建输出目录（只在主进程中创建，避免竞争条件）
    if rank == 0:
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["logging_dir"], exist_ok=True)
    
    # 设置随机种子，确保实验可重现
    set_seed(config["seed"])
    
    # 创建微调参数配置
    finetuning_args = FinetuningArguments(
        freeze_trainable_layers=config["freeze_trainable_layers"],
        freeze_trainable_modules=config["freeze_trainable_modules"],
        freeze_extra_modules=config.get("freeze_extra_modules")
    )
    
    # 打印配置信息（只在主进程中打印）
    if rank == 0:
        print(f"基础模型: {config['model_name_or_path']}")
        print(f"Freeze层数: {finetuning_args.freeze_trainable_layers}")
        print(f"输出目录: {config['output_dir']}")
    
    # 加载tokenizer（文本分词器）
    # tokenizer负责将文本转换为模型可以理解的数字序列
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], 
        trust_remote_code=config["trust_remote_code"]
    )
    # 如果没有pad_token，使用eos_token作为pad_token
    # pad_token用于将不同长度的序列填充到相同长度
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置
    model_config = AutoConfig.from_pretrained(
        config["model_name_or_path"], 
        trust_remote_code=config["trust_remote_code"]
    )
    # 设置输出标签数为1，因为奖励模型输出单个分数
    model_config.num_labels = 1
    
    # 在CPU上初始化模型
    # 这是为了节省GPU内存，DeepSpeed会自动将模型移动到GPU
    with torch.device('cpu'):
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name_or_path"],
            config=model_config,
            torch_dtype=getattr(torch, config["torch_dtype"]),
            trust_remote_code=config["trust_remote_code"]
        )
    
    # 设置freeze tuning，只训练部分参数
    setup_freeze_tuning(model, finetuning_args)
    
    # 创建数据集
    # 这会加载训练数据和验证数据，并进行预处理
    train_dataset, eval_dataset = create_reward_dataset(
        config["data_path"], tokenizer, config["max_length"]
    )
    # 创建数据整理器，负责将数据组织成批次
    data_collator = create_data_collator(tokenizer)
    
    # 打印数据集信息
    if rank == 0:
        print(f"数据集 - 训练: {len(train_dataset)}, 验证: {len(eval_dataset)}")
    
    # 创建训练参数配置
    training_args = TrainingArguments(
        # 基本训练参数
        output_dir=config["output_dir"],                    # 输出目录
        num_train_epochs=config["num_train_epochs"],        # 训练轮数
        learning_rate=config["learning_rate"],              # 学习率
        weight_decay=config["weight_decay"],                # 权重衰减（正则化）
        warmup_ratio=config["warmup_ratio"],                # 学习率预热比例
        max_grad_norm=config["max_grad_norm"],              # 梯度裁剪阈值
        seed=config["seed"],                                # 随机种子
        
        # 批次大小设置
        per_device_train_batch_size=config["per_device_train_batch_size"],    # 每个设备的训练批次大小
        per_device_eval_batch_size=config["per_device_eval_batch_size"],      # 每个设备的评估批次大小
        gradient_accumulation_steps=config["gradient_accumulation_steps"],    # 梯度累积步数
        
        # 评估策略
        eval_strategy=config["evaluation_strategy"],        # 评估策略（按步数评估）
        eval_steps=config["eval_steps"],                    # 评估间隔步数
        do_eval=True,                                       # 启用评估
        
        # 保存策略
        save_strategy=config["save_strategy"],              # 保存策略（按步数保存）
        save_steps=config["save_steps"],                    # 保存间隔步数
        save_total_limit=config["save_total_limit"],        # 最多保存的checkpoint数量
        save_safetensors=True,                              # 使用safetensors格式保存
        
        # 模型选择策略
        load_best_model_at_end=config["load_best_model_at_end"],        # 训练结束时加载最佳模型
        metric_for_best_model=config["metric_for_best_model"],          # 最佳模型的评估指标
        greater_is_better=config["greater_is_better"],                  # 指标是否越大越好
        
        # 日志设置
        logging_steps=config["logging_steps"],              # 日志记录间隔
        logging_dir=config["logging_dir"],                  # 日志目录
        
        # 精度设置
        bf16=config["bf16"],                                # 使用bfloat16精度
        fp16=config["fp16"],                                # 使用float16精度
        tf32=config["tf32"],                                # 使用TensorFloat-32
        
        # DeepSpeed配置
        deepspeed="configs/training/deepspeed_only.json",   # DeepSpeed配置文件
        
        # 分布式训练设置
        local_rank=local_rank,                              # 本地rank
        ddp_find_unused_parameters=False,                   # 不查找未使用的参数
        dataloader_num_workers=0,                           # 数据加载器工作进程数
        remove_unused_columns=False,                        # 不移除未使用的列（重要：保留自定义数据格式）
        report_to=[],                                       # 不上报到外部服务
    )
    
    # 创建训练器
    trainer = RewardTrainer(
        model=model,                                        # 要训练的模型
        args=training_args,                                 # 训练参数
        train_dataset=train_dataset,                        # 训练数据集
        eval_dataset=eval_dataset,                          # 验证数据集
        data_collator=data_collator,                        # 数据整理器
        processing_class=tokenizer,                         # 处理类（tokenizer）
        callbacks=[
            # 早停回调：如果验证损失不再改善，提前停止训练
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 5),  # 容忍的评估次数
                early_stopping_threshold=0.001                                      # 改善的最小阈值
            )
        ]
    )
    
    # 开始训练
    if rank == 0:
        print("开始训练...")
    
    # 执行训练过程
    # 如果指定了checkpoint，会从该checkpoint恢复训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 训练完成后的处理（只在主进程中执行）
    if rank == 0:
        print("训练完成")
        
        # 保存最终模型
        # 这会保存完整的模型和tokenizer到final_model目录
        final_model_path = os.path.join(config["output_dir"], "final_model")
        trainer.save_model(final_model_path)
        print(f"最终模型保存到: {final_model_path}")


# 如果这个文件被直接运行（而不是被导入），则执行主函数
if __name__ == "__main__":
    main()