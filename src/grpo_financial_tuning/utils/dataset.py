"""
数据集处理工具类
作用：为GRPO训练提供数据加载和预处理功能
功能：支持JSONL格式数据加载、对话格式化、奖励函数创建
"""
import json                              # JSON数据处理
from typing import List, Dict, Any       # 类型提示
from pathlib import Path                 # 路径操作
from datasets import Dataset             # Hugging Face数据集库
from transformers import PreTrainedTokenizer  # 分词器基类
import torch                            # PyTorch深度学习框架


class GRPODatasetLoader:
    """
    GRPO数据集加载器
    
    作用：专门为GRPO训练加载和处理数据
    功能：
    1. 加载JSONL格式的训练数据
    2. 将数据格式化为对话格式
    3. 创建适合GRPO训练的数据集
    4. 处理数据加载过程中的错误
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, logger=None):
        """
        初始化数据集加载器
        
        参数：
            tokenizer: 分词器实例，用于文本处理
            logger: 日志器实例，用于记录处理过程
        """
        self.tokenizer = tokenizer  # 保存分词器引用
        self.logger = logger        # 保存日志器引用
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSONL格式数据文件
        
        参数：
            file_path: 数据文件路径
            
        返回：
            List[Dict]: 解析后的数据列表，每个元素是一个字典
        
        作用：
        1. 逐行读取JSONL文件
        2. 解析每行的JSON数据
        3. 验证数据格式
        4. 统计加载结果
        """
        # 检查文件是否存在
        if not Path(file_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        data = []           # 存储有效数据
        error_count = 0     # 错误计数
        
        # 逐行读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()  # 去除首尾空白
                if not line:         # 跳过空行
                    continue
                
                try:
                    # 解析JSON数据
                    item = json.loads(line)
                    # 验证必需字段
                    if 'prompt' in item:
                        data.append(item)
                    else:
                        if self.logger:
                            self.logger.warning(f"第{line_num}行缺少'prompt'字段")
                        error_count += 1
                except json.JSONDecodeError as e:
                    # JSON解析错误
                    if self.logger:
                        self.logger.error(f"第{line_num}行JSON解析失败: {e}")
                    error_count += 1
        
        # 记录加载结果
        if self.logger:
            self.logger.info(f"数据加载完成: {len(data)}条有效记录, {error_count}条错误记录")
        
        return data
    
    def format_chat_prompt(self, prompt: str) -> List[Dict[str, str]]:
        """
        将prompt格式化为对话格式
        
        参数：
            prompt: 原始prompt文本
            
        返回：
            List[Dict]: 格式化后的对话消息列表
        
        作用：
        1. 将单一的prompt转换为多轮对话格式
        2. 添加系统提示词，定义AI助手的角色
        3. 将用户问题包装为用户消息
        
        GRPO训练需要对话格式的数据，因为：
        - 现代语言模型都是基于对话训练的
        - 对话格式可以更好地控制模型行为
        - 系统提示词可以定义专业领域角色
        """
        return [
            {
                "role": "system",  # 系统角色
                "content": "你是一个专业的金融分析师，擅长投资分析和风险评估。请基于提供的信息进行详细的分析和推理。"
            },
            {
                "role": "user",    # 用户角色
                "content": prompt  # 用户问题
            }
        ]
    
    def create_dataset(self, data_path: str) -> Dataset:
        """
        创建GRPO训练数据集
        
        参数：
            data_path: 数据文件路径
            
        返回：
            Dataset: HuggingFace数据集对象
        
        作用：
        1. 加载原始数据
        2. 格式化为对话格式
        3. 转换为HuggingFace数据集
        4. 保留原始数据的其他字段
        
        GRPO训练的数据特点：
        - 只需要prompt，不需要预定义答案
        - 训练时模型会生成多个回复
        - 使用奖励模型评分和优化
        """
        if self.logger:
            self.logger.info(f"开始加载数据集: {data_path}")
        
        # 加载原始数据
        raw_data = self.load_jsonl(data_path)
        
        # 格式化数据
        formatted_data = []
        for item in raw_data:
            # 格式化prompt为对话格式
            formatted_item = {
                "prompt": self.format_chat_prompt(item["prompt"])
            }
            # 保留其他字段（如果有的话）
            for key, value in item.items():
                if key != "prompt":
                    formatted_item[key] = value
            
            formatted_data.append(formatted_item)
        
        # 转换为HuggingFace数据集
        dataset = Dataset.from_list(formatted_data)
        
        if self.logger:
            self.logger.info(f"数据集创建完成: {len(dataset)}条记录")
        
        return dataset


def create_reward_function(reward_model_path: str, logger=None):
    """
    创建奖励函数
    
    参数：
        reward_model_path: 奖励模型路径
        logger: 日志器实例
    
    返回：
        reward_function: 奖励计算函数
    
    作用：
    1. 加载预训练的奖励模型
    2. 创建GRPO兼容的奖励函数
    3. 确保文本格式与奖励模型训练时一致
    
    奖励函数的重要性：
    - GRPO是强化学习算法，需要奖励信号指导训练
    - 奖励模型评估生成文本的质量
    - 文本格式必须与奖励模型训练时完全一致
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    if logger:
        logger.info(f"加载奖励模型: {reward_model_path}")
    
    # 为奖励模型使用专用的分词器
    # 重要：奖励模型可能使用不同的分词器配置
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_model_path,
        trust_remote_code=True
    )
    
    # 设置填充token
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # 加载奖励模型
    # 奖励模型是一个分类模型，输出单一分数
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,                    # 输出单一奖励分数
        torch_dtype=torch.bfloat16,      # 使用半精度节省显存
        trust_remote_code=True,
        use_cache=False                  # 禁用缓存节省内存
    )
    reward_model.eval()  # 设置为评估模式
    
    if logger:
        logger.info("奖励模型加载完成")
    
    def reward_function(prompts, completions, **kwargs):
        """
        GRPO兼容的奖励函数 - 使用与奖励模型训练一致的LLaMA-3格式
        
        参数：
            prompts: 输入提示列表
            completions: 模型生成的完成文本列表
            **kwargs: 其他参数
        
        返回：
            rewards: 奖励分数列表
        
        关键修复：
        - 使用与奖励模型训练完全相同的文本格式
        - LLaMA-3格式的特殊token必须一致
        - 避免格式不匹配导致的奖励异常
        """
        rewards = []
        
        # 禁用梯度计算，节省内存
        with torch.no_grad():
            for prompt, completion in zip(prompts, completions):
                try:
                    # 提取用户问题
                    if isinstance(prompt, list):
                        # 从对话格式中提取用户问题
                        question = next(msg["content"] for msg in prompt if msg["role"] == "user")
                    else:
                        question = prompt
                    
                    # 处理completion
                    if isinstance(completion, list):
                        completion_text = completion[0]["content"]
                    else:
                        completion_text = completion
                    
                    # 关键修复：使用与奖励模型训练完全相同的LLaMA-3格式
                    # 这个格式必须与奖励模型训练时的格式完全一致
                    formatted_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion_text}<|eot_id|>"
                    
                    # 使用奖励模型计算分数
                    inputs = reward_tokenizer(
                        formatted_text,
                        truncation=True,           # 截断过长文本
                        padding=True,              # 填充到统一长度
                        max_length=2048,          # 与奖励模型训练时一致
                        return_tensors="pt"        # 返回PyTorch张量
                    )
                    
                    # 将输入移动到模型设备
                    inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                    
                    # 模型推理
                    outputs = reward_model(**inputs)
                    
                    # 提取奖励分数
                    reward = outputs.logits.squeeze().cpu().item()
                    rewards.append(reward)
                    
                except Exception as e:
                    # 处理计算失败的情况
                    if logger:
                        logger.warning(f"奖励计算失败: {e}")
                    rewards.append(0.0)  # 使用默认分数
        
        return rewards
    
    return reward_function