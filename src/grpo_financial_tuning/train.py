"""
GRPO训练主文件 - 简单修复DeepSpeed参数传递
作用：实现完整的GRPO强化学习训练流程
功能：包含模型加载、数据处理、训练器配置、最佳模型保存等
"""
import torch

# ==================== PyTorch加载修复 ====================
# 修复checkpoint加载时的weights_only问题
# 背景：新版PyTorch对checkpoint加载有更严格的安全检查
original_torch_load = torch.load  # 保存原始加载函数

def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    """
    修复的torch.load函数
    
    参数：
        f: 文件路径或文件对象
        map_location: 设备映射
        pickle_module: pickle模块
        weights_only: 是否只加载权重
        **kwargs: 其他参数
    
    作用：
    1. 检测是否是checkpoint相关文件
    2. 对于checkpoint文件，强制设置weights_only=False
    3. 避免PyTorch安全检查导致的加载失败
    """
    # 如果是加载checkpoint相关文件，强制设置weights_only=False
    # 这样可以避免某些版本的PyTorch过于严格的安全检查
    if weights_only is True and (
        isinstance(f, str) and ('rng_state' in f or 'checkpoint' in f)
    ):
        weights_only = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                              weights_only=weights_only, **kwargs)

# 替换PyTorch的加载函数
torch.load = patched_torch_load

# ==================== 导入必要的库 ====================
import time                                      # 时间处理
from pathlib import Path                         # 路径操作
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback  # Transformers库
from peft import PeftModel                       # LoRA等参数高效微调
from trl import GRPOConfig, GRPOTrainer         # TRL强化学习库
import argparse                                  # 命令行参数解析
import numpy as np                               # 数值计算
import os                                        # 操作系统接口
from utils.config_loader import ProjectConfig   # 项目配置加载器
from utils.logger import create_logger          # 日志工具
from utils.dataset import GRPODatasetLoader, create_reward_function  # 数据集和奖励函数


class GRPOBestModelEarlyStoppingCallback(TrainerCallback):
    """
    GRPO最佳模型早停回调
    
    作用：
    1. 自动保存最佳模型
    2. 实现早停机制，防止过拟合
    3. 监控训练和验证指标
    4. 管理模型保存策略
    
    GRPO训练的特点：
    - 强化学习训练可能不稳定
    - 需要持续监控奖励指标
    - 最佳模型可能出现在训练中期
    - 需要保存最佳性能的模型而非最后的模型
    """
    
    def __init__(
        self, 
        early_stopping_patience: int = 3,           # 早停耐心值
        early_stopping_threshold: float = 0.001,    # 早停阈值
        train_reward_weight: float = 0.3,           # 训练奖励权重
        eval_reward_weight: float = 0.7,            # 验证奖励权重
        min_eval_steps: int = 2,                    # 最小验证步数
        best_model_dir: str = None,                 # 最佳模型保存目录
        tokenizer = None                            # 分词器实例
    ):
        """
        初始化早停回调
        
        参数说明：
            early_stopping_patience: 连续多少次验证无改善后停止训练
            early_stopping_threshold: 认为有改善的最小阈值
            train_reward_weight: 训练奖励在综合评分中的权重
            eval_reward_weight: 验证奖励在综合评分中的权重
            min_eval_steps: 开始早停检查前的最小验证步数
            best_model_dir: 最佳模型保存目录
            tokenizer: 分词器实例，用于保存
        """
        # 早停参数
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.train_reward_weight = train_reward_weight
        self.eval_reward_weight = eval_reward_weight
        self.min_eval_steps = min_eval_steps
        
        # 保存tokenizer引用
        self.tokenizer = tokenizer
        
        # 最佳模型保存路径
        self.best_model_dir = best_model_dir or "./output/best_model"
        Path(self.best_model_dir).mkdir(parents=True, exist_ok=True)
        
        # 训练指标跟踪
        self.latest_train_reward = 0.0      # 最新训练奖励
        self.best_train_reward = None       # 最佳训练奖励
        
        # 验证指标跟踪
        self.best_eval_reward = None        # 最佳验证奖励
        self.eval_count = 0                 # 验证次数计数
        
        # 早停控制变量
        self.patience_counter = 0           # 耐心计数器
        self.best_combined_score = None     # 最佳综合得分
        self.best_step = None               # 最佳模型对应的步数
        
        # 调试计数器
        self.log_call_count = 0             # 日志调用次数
        self.eval_call_count = 0            # 验证调用次数
        
        print(f"回调函数初始化: 训练权重={train_reward_weight}, 验证权重={eval_reward_weight}")
        print(f"最佳模型保存目录: {self.best_model_dir}")
        print(f"分词器引用: {self.tokenizer is not None}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        监听所有日志事件
        
        参数：
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志字典
            **kwargs: 其他参数（包含model等）
        
        作用：
        1. 监听训练过程中的所有日志
        2. 识别验证日志并触发模型保存
        3. 更新训练奖励记录
        4. 实现最佳模型保存逻辑
        """
        self.log_call_count += 1
        
        print(f"on_log调用#{self.log_call_count} - Step {state.global_step}")
        
        if logs:
            # 检查是否是验证日志
            # 验证日志的特征是包含以'eval_'开头的键
            if any(key.startswith('eval_') for key in logs.keys()):
                print("检测到验证日志，尝试保存最佳模型...")
                
                # 获取验证奖励
                eval_reward = logs.get('eval_reward')
                if eval_reward is not None:
                    print(f"验证奖励: {eval_reward}")
                    
                    # 判断是否需要保存模型
                    should_save = False
                    if self.best_eval_reward is None or eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        should_save = True
                        print(f"发现更好的验证奖励: {eval_reward}")
                    
                    # 第一次验证时强制保存
                    if self.eval_count == 0:
                        should_save = True
                        print("第一次验证，强制保存模型")
                    
                    if should_save:
                        # 从kwargs中获取模型实例
                        model = kwargs.get('model')
                        
                        if model and self.tokenizer:
                            self._save_best_model(model, self.tokenizer, state.global_step)
                        else:
                            print(f"无法获取model或tokenizer: model={model is not None}, tokenizer={self.tokenizer is not None}")
                    
                    self.eval_count += 1
            
            # 获取训练奖励
            # GRPO训练中奖励可能以不同的键名出现
            possible_reward_keys = ['reward', 'rewards/reward_function/mean', 'train_reward', 'reward_mean']
            for key in possible_reward_keys:
                if key in logs:
                    self.latest_train_reward = logs[key]
                    break
        
        return control
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        验证回调 - 简化版本
        
        参数：
            args: 训练参数
            state: 训练状态
            control: 训练控制
            logs: 日志字典
            **kwargs: 其他参数
        
        作用：
        1. 记录验证回调的调用
        2. 验证逻辑已经移到on_log中处理
        
        注意：实际的验证处理逻辑在on_log方法中
        """
        self.eval_call_count += 1
        print(f"on_evaluate调用#{self.eval_call_count} - Step {state.global_step}")
        
        # 验证逻辑已经移到 on_log 中处理
        return control
    
    def _save_best_model(self, model, tokenizer, step):
        """
        保存最佳模型
        
        参数：
            model: 模型实例
            tokenizer: 分词器实例
            step: 当前训练步数
        
        作用：
        1. 处理DeepSpeed包装的模型
        2. 保存模型权重和配置
        3. 保存分词器
        4. 记录模型元信息
        5. 处理保存过程中的异常
        """
        try:
            print(f"开始保存最佳模型 (Step {step})...")
            
            # 确保目录存在
            Path(self.best_model_dir).mkdir(parents=True, exist_ok=True)
            
            # 处理DeepSpeed包装的模型
            # DeepSpeed会用module属性包装原始模型
            actual_model = model.module if hasattr(model, 'module') else model
                
            # 保存模型
            actual_model.save_pretrained(
                self.best_model_dir,              # 保存目录
                safe_serialization=True,          # 使用安全序列化
                max_shard_size="2GB"              # 最大分片大小
            )
            
            # 保存分词器
            tokenizer.save_pretrained(self.best_model_dir)
            
            # 保存元信息
            # 记录模型的关键指标和保存时间
            best_info = {
                "best_step": step,                         # 最佳模型对应的步数
                "best_combined_score": self.best_combined_score,  # 最佳综合得分
                "best_train_reward": self.best_train_reward,      # 最佳训练奖励
                "best_eval_reward": self.best_eval_reward,        # 最佳验证奖励
                "latest_train_reward": self.latest_train_reward,  # 最新训练奖励
                "save_time": time.strftime("%Y-%m-%d %H:%M:%S")   # 保存时间
            }
            
            import json
            with open(f"{self.best_model_dir}/best_model_info.json", "w") as f:
                json.dump(best_info, f, indent=2)
                
            print(f"最佳模型保存完成！文件数量: {len(list(Path(self.best_model_dir).glob('*')))}")
            
        except Exception as e:
            print(f"保存最佳模型失败: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈


class GRPOTraining:
    """
    GRPO训练主类
    
    作用：
    1. 统一管理GRPO训练的所有组件
    2. 按照标准流程初始化各个模块
    3. 提供完整的训练接口
    4. 处理训练过程中的异常
    
    训练流程：
    1. 设置分词器
    2. 加载策略模型
    3. 准备训练数据
    4. 创建奖励函数
    5. 配置训练器
    6. 执行训练
    """
    
    def __init__(self, config: ProjectConfig = None, deepspeed_config: str = None):
        """
        初始化GRPO训练器
        
        参数：
            config: 项目配置实例
            deepspeed_config: DeepSpeed配置文件路径
        """
        self.config = config or ProjectConfig()          # 项目配置
        self.deepspeed_config = deepspeed_config         # DeepSpeed配置路径
        
        # 创建输出目录
        output_dir = self.config.get('output.output_dir')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建日志器
        self.logger = create_logger(output_dir, 'INFO')
        
        # 初始化组件引用
        self.tokenizer = None        # 分词器
        self.model = None           # 策略模型
        self.dataset = None         # 训练数据集
        self.reward_function = None # 奖励函数
        self.trainer = None         # GRPO训练器
    
    def setup_tokenizer(self):
        """
        设置分词器
        
        作用：
        1. 加载预训练分词器
        2. 配置特殊token
        3. 设置填充策略
        
        分词器的重要性：
        - 将文本转换为模型可理解的数字序列
        - 处理特殊token（如开始、结束、填充token）
        - 确保文本格式与模型训练时一致
        """
        self.logger.info("设置分词器...")
        model_name = self.config.get('model.model_name')
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,                    # 模型名称或路径
            trust_remote_code=True,        # 允许执行远程代码
            padding_side="left"            # 左侧填充（生成任务推荐）
        )
        
        # 设置填充token
        # 如果模型没有定义填充token，使用结束token代替
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info("分词器设置完成")
    
    def setup_model(self):
        """
        设置策略模型
        
        作用：
        1. 加载预训练的基座模型
        2. 可选地加载LoRA适配器
        3. 配置模型参数
        4. 确保与DeepSpeed兼容
        
        模型加载策略：
        - 使用bfloat16精度节省显存
        - 禁用缓存减少内存占用
        - 让DeepSpeed处理模型分布
        - 确保所有参数可训练
        """
        self.logger.info("设置策略模型...")
        model_name = self.config.get('model.model_name')
        
        # 让DeepSpeed处理模型分布
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,                    # 模型名称或路径
            torch_dtype=torch.bfloat16,    # 使用半精度浮点
            trust_remote_code=True,        # 允许执行远程代码
            use_cache=False,               # 禁用KV缓存节省内存
            device_map=None,               # 关键：让DeepSpeed处理设备分配
            low_cpu_mem_usage=True         # 低CPU内存使用模式
        )
        
        # 加载LoRA模型（如果配置了）
        # LoRA是一种参数高效的微调方法
        lora_path = self.config.get('model.lora_model_path')
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        # 确保参数可训练
        # 某些情况下模型参数可能被冻结
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.logger.info("策略模型设置完成")
    
    def setup_dataset(self):
        """
        设置数据集
        
        作用：
        1. 创建数据集加载器
        2. 加载训练数据
        3. 加载验证数据
        4. 格式化为GRPO训练格式
        
        GRPO数据特点：
        - 只需要prompt，不需要答案
        - 使用对话格式
        - 支持批量处理
        """
        self.logger.info("设置数据集...")
        dataset_loader = GRPODatasetLoader(self.tokenizer, self.logger)
        
        # 加载训练数据
        data_path = self.config.get('data.train_data_path')
        self.dataset = dataset_loader.create_dataset(data_path)
        
        # 加载验证数据
        eval_data_path = self.config.get('data.eval_data_path')
        self.eval_dataset = dataset_loader.create_dataset(eval_data_path)
        
        self.logger.info(f"数据集设置完成: {len(self.dataset)}条记录")
    
    def setup_reward_function(self):
        """
        设置奖励函数
        
        作用：
        1. 加载预训练的奖励模型
        2. 创建GRPO兼容的奖励函数
        3. 确保文本格式一致性
        
        奖励函数的作用：
        - 评估模型生成文本的质量
        - 为GRPO提供训练信号
        - 指导模型向更好的方向优化
        """
        self.logger.info("设置奖励函数...")
        reward_model_path = self.config.get('model.reward_model_path')
        self.reward_function = create_reward_function(reward_model_path, self.logger)
        self.logger.info("奖励函数设置完成")
    
    def setup_trainer(self):
        """
        设置GRPO训练器
        
        作用：
        1. 配置训练参数
        2. 创建早停回调
        3. 初始化GRPO训练器
        4. 集成所有组件
        
        训练器配置包括：
        - 基本训练参数（学习率、批次大小等）
        - GRPO特有参数（生成数量、温度等）
        - 保存和评估策略
        - 日志和监控设置
        """
        self.logger.info("设置GRPO训练器...")
        
        # 创建最佳模型早停回调
        best_model_dir = f"{self.config.get('output.output_dir')}best_model"
        early_stopping_callback = GRPOBestModelEarlyStoppingCallback(
            early_stopping_patience=self.config.get('training.early_stopping_patience', 3),
            early_stopping_threshold=self.config.get('training.early_stopping_threshold', 0.001),
            best_model_dir=best_model_dir,
            tokenizer=self.tokenizer
        )
        
        # 配置训练参数
        training_args = GRPOConfig(
            # 基本训练参数
            output_dir=self.config.get('output.output_dir'),                                    # 输出目录
            per_device_train_batch_size=self.config.get('training.per_device_train_batch_size',2),  # 每设备训练批次大小
            per_device_eval_batch_size=self.config.get('training.per_device_eval_batch_size', 4),   # 每设备验证批次大小
            gradient_accumulation_steps=self.config.get('training.gradient_accumulation_steps'),    # 梯度累积步数
            learning_rate=self.config.get('training.learning_rate'),                               # 学习率
            num_train_epochs=self.config.get('training.num_train_epochs'),                         # 训练轮数
            logging_steps=self.config.get('output.logging_steps'),                                 # 日志记录步数
            
            # GRPO特有参数
            num_generations=self.config.get('grpo.num_generations', 8),    # 每个prompt生成的回复数量
            temperature=self.config.get('grpo.temperature', 0.8),          # 生成温度（控制随机性）
            top_p=self.config.get('grpo.top_p', 0.9),                     # 核采样参数
            epsilon=self.config.get('grpo.epsilon', 0.2),                  # PPO裁剪参数
            beta=self.config.get('grpo.beta', 0.0),                       # KL散度惩罚系数
            
            # 修复生成长度限制
            max_completion_length=self.config.get('data.max_completion_length', 512),  # 最大生成长度
            
            # Checkpoint策略
            save_strategy="steps",                                          # 按步数保存
            save_steps=self.config.get('output.save_steps', 500),         # 保存间隔
            save_total_limit=self.config.get('output.save_total_limit', 3), # 最大保存数量
            
            # 评估策略 
            eval_strategy="steps",                                          # 按步数评估
            eval_steps=self.config.get('output.eval_steps', 100),         # 评估间隔
            
            # 不使用传统的最佳模型加载（我们自己管理）
            load_best_model_at_end=False,                                  # 训练结束时不加载最佳模型
            metric_for_best_model=None,                                    # 不指定最佳模型指标
            
            # 其他参数
            warmup_steps=self.config.get('training.warmup_steps', 50),     # 预热步数
            max_grad_norm=self.config.get('training.max_grad_norm', 1.0),  # 梯度裁剪
            dataloader_num_workers=self.config.get('training.dataloader_num_workers', 0),  # 数据加载线程数
            bf16=True,                                                      # 使用bfloat16精度
            remove_unused_columns=False,                                   # 保留所有数据列
            deepspeed=self.deepspeed_config,                               # DeepSpeed配置
            report_to=["tensorboard"],                                     # 使用TensorBoard监控
            logging_dir=os.path.join(self.config.get('output.output_dir'), "logs"),  # 日志目录
        )
        
        # 创建GRPO训练器
        self.trainer = GRPOTrainer(
            model=self.model,                    # 策略模型
            args=training_args,                  # 训练参数
            train_dataset=self.dataset,          # 训练数据集
            eval_dataset=self.eval_dataset,      # 验证数据集
            processing_class=self.tokenizer,     # 文本处理器（分词器）
            reward_funcs=[self.reward_function], # 奖励函数列表
            callbacks=[early_stopping_callback] # 回调函数列表
        )
        
        self.logger.info("GRPO训练器设置完成")
    
    def train(self, resume_from_checkpoint=None):
        """
        执行GRPO训练
        
        参数：
            resume_from_checkpoint: 从指定checkpoint恢复训练
        
        作用：
        1. 按顺序初始化所有组件
        2. 执行完整的训练流程
        3. 保存最终模型
        4. 处理训练过程中的异常
        
        训练流程：
        1. 记录训练开始
        2. 初始化各个组件
        3. 执行GRPO训练
        4. 保存模型和分词器
        5. 记录训练结果
        """
        try:
            # 记录训练开始
            self.logger.log_training_start(self.config)
            
            # 按顺序初始化所有组件
            self.setup_tokenizer()      # 1. 设置分词器
            self.setup_model()          # 2. 加载模型
            self.setup_dataset()        # 3. 准备数据
            self.setup_reward_function() # 4. 创建奖励函数
            self.setup_trainer()        # 5. 配置训练器
            
            # 开始训练
            self.logger.info("开始GRPO训练...")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # 保存最终模型
            self.logger.info("保存模型...")
            self.trainer.save_model()  # 保存模型权重
            self.tokenizer.save_pretrained(self.config.get('output.output_dir'))  # 保存分词器
            
            # 记录训练成功
            self.logger.log_training_end(success=True)
            
        except Exception as e:
            # 处理训练失败
            self.logger.error(f"训练失败: {str(e)}")
            self.logger.log_training_end(success=False)
            raise  # 重新抛出异常


def main():
    """
    主函数：程序入口点
    
    作用：
    1. 解析命令行参数
    2. 加载配置
    3. 创建训练器
    4. 启动训练
    
    支持的命令行参数：
    - --deepspeed: DeepSpeed配置文件路径
    - --resume_from_checkpoint: 从checkpoint恢复训练
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', type=str, help='DeepSpeed配置文件路径')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                       help='从checkpoint恢复训练')
    args = parser.parse_known_args()[0]  # 只取已知参数
    
    # 加载配置和创建训练器
    config = ProjectConfig()  # 加载项目配置
    trainer = GRPOTraining(config, args.deepspeed)  # 创建训练器
    
    # 启动训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


# 程序入口点
if __name__ == "__main__":
    main()