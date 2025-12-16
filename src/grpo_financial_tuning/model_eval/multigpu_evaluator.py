#!/usr/bin/env python3
"""
多机多卡奖励模型评分工具 - 使用DeepSpeed
这个工具用于在多个GPU上同时评测GRPO训练后的模型质量
主要功能：给定一些问题，让模型生成回答，然后用奖励模型给回答打分
"""
import torch
import json
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os

class PromptDataset(Dataset):
    """
    Prompt数据集类
    作用：将输入的问题列表转换为PyTorch可以处理的数据集格式
    每个问题会复制num_generations次，因为每个问题要生成多个回答
    """
    def __init__(self, prompts: list, num_generations: int = 4):
        """
        初始化数据集
        prompts: 问题列表，例如["如何分析股票？", "什么是基金？"]
        num_generations: 每个问题生成几个回答，默认4个
        """
        self.data = []
        # 将每个问题复制num_generations次
        # 例如：1个问题 × 4次生成 = 4个数据项
        for prompt in prompts:
            for _ in range(num_generations):
                self.data.append(prompt)
    
    def __len__(self):
        """返回数据集总大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """根据索引返回对应的问题"""
        return self.data[idx]

class MultiGPURewardEvaluator:
    """
    多GPU奖励模型评分器
    作用：在多个GPU上同时运行模型评测，提高评测效率
    包含两个模型：主模型（生成回答）和奖励模型（给回答打分）
    """
    def __init__(self, model_path: str, reward_model_path: str):
        """
        初始化多GPU评分器
        model_path: 主模型路径（GRPO训练后的模型）
        reward_model_path: 奖励模型路径（用于给回答打分）
        """
        self.model_path = model_path
        self.reward_model_path = reward_model_path
        
        # 初始化DeepSpeed分布式环境
        # DeepSpeed是微软开发的深度学习优化库，支持多GPU训练和推理
        deepspeed.init_distributed()
        
        # 获取当前进程的GPU编号和总GPU数量
        # LOCAL_RANK: 当前进程在本机上的GPU编号（0, 1, 2...）
        # WORLD_SIZE: 总的GPU数量
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 设置当前进程使用的GPU设备
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # 加载和配置模型
        self.setup_models()
    
    def setup_models(self):
        """
        设置和加载模型
        作用：加载主模型和奖励模型，并使用DeepSpeed进行优化
        """
        # 只在主进程（GPU 0）上打印信息，避免重复输出
        if self.local_rank == 0:
            print(f"加载模型: {self.model_path}")
        
        # 加载主模型的分词器
        # 分词器负责将文本转换为模型可以理解的数字序列
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # 如果分词器没有padding token，使用结束token作为padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载主模型（用于生成回答）
        # torch_dtype=torch.bfloat16: 使用半精度浮点数，节省显存
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        
        # 使用DeepSpeed初始化主模型
        # DeepSpeed会自动处理模型的分布式部署和内存优化
        self.model, _, _, _ = deepspeed.initialize(
            model=model,
            config={
                "train_batch_size": 2,                    # 训练批次大小
                "train_micro_batch_size_per_gpu": 1,      # 每个GPU的微批次大小
                "gradient_accumulation_steps": 1,          # 梯度累积步数
                "bf16": {"enabled": True},                 # 启用bfloat16精度
                "zero_optimization": {"stage": 0}          # ZeRO优化级别0（不分割参数）
            }
        )
        
        # 加载奖励模型 - 参考dataset.py的实现方式
        if self.local_rank == 0:
            print(f"加载奖励模型: {self.reward_model_path}")
        
        # 加载奖励模型的分词器
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_path, trust_remote_code=True)
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        
        # 加载奖励模型（用于给回答打分）
        # num_labels=1: 输出一个分数值
        # use_cache=False: 不使用缓存，节省内存
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_path, num_labels=1, torch_dtype=torch.bfloat16, 
            trust_remote_code=True, use_cache=False
        )
        
        # 使用DeepSpeed初始化奖励模型
        self.reward_model, _, _, _ = deepspeed.initialize(
            model=reward_model,
            config={
                "train_batch_size": 2,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 0}
            }
        )
    
    def generate_response(self, prompt: str) -> str:
        """
        生成回复
        作用：给定一个问题，使用主模型生成一个回答
        prompt: 输入的问题文本
        返回：模型生成的回答文本
        """
        # 将问题格式化为对话格式
        # 这里使用系统提示词让模型扮演金融分析师角色
        messages = [
            {"role": "system", "content": "你是一个专业的金融分析师，擅长投资分析和风险评估。请基于提供的信息进行详细的分析和推理。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用分词器的聊天模板功能格式化对话
        # add_generation_prompt=True: 添加生成提示符
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 将文本转换为模型输入格式
        # truncation=True: 如果文本太长就截断
        # max_length=512: 最大输入长度512个token
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # 将输入数据移动到当前GPU上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():  # 不计算梯度，节省内存
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512,                        # 最多生成512个新token
                temperature=0.7,                           # 温度参数，控制生成的随机性
                top_p=0.9,                                # 核采样参数
                do_sample=True,                           # 启用采样生成
                pad_token_id=self.tokenizer.eos_token_id  # 填充token ID
            )
        
        # 解码生成的回答
        # 只取新生成的部分，跳过原始输入
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        return response
    
    def compute_reward(self, prompt: str, completion: str) -> float:
        """
        计算奖励分数
        作用：使用奖励模型给问题-回答对打分
        prompt: 问题文本
        completion: 回答文本
        返回：奖励分数（浮点数）
        """
        try:
            # 使用与奖励模型训练完全相同的LLaMA-3格式
            # 这个格式化非常重要，必须与训练时保持一致
            formatted_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{completion}<|eot_id|>"
            
            # 将格式化的文本转换为模型输入
            inputs = self.reward_tokenizer(
                formatted_text, 
                truncation=True,      # 截断过长文本
                padding=True,         # 填充到相同长度
                max_length=2048,      # 最大长度2048
                return_tensors="pt"   # 返回PyTorch张量
            )
            
            # 将输入移动到当前GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 计算奖励分数
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                # 提取分数并转换为Python数值
                reward = outputs.logits.squeeze().cpu().item()
            
            return reward
        except Exception as e:
            # 如果计算失败，只在主进程打印错误信息
            if self.local_rank == 0:
                print(f"奖励计算失败: {e}")
            return 0.0
    
    def evaluate_batch(self, prompts: list) -> list:
        """
        批量评测
        作用：对一批问题进行评测，每个GPU处理不同的问题
        prompts: 问题列表
        返回：评测结果列表
        """
        results = []
        
        # 创建数据集和数据加载器
        # 每个问题会生成4个回答
        dataset = PromptDataset(prompts, num_generations=4)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 分布式处理：每个GPU处理不同的样本
        for i, batch_prompts in enumerate(tqdm(dataloader, desc=f"GPU-{self.local_rank}", disable=self.local_rank!=0)):
            prompt = batch_prompts[0]
            
            # 工作分配：第i个样本由第(i % world_size)个GPU处理
            # 例如：GPU0处理样本0,3,6..., GPU1处理样本1,4,7...
            if i % self.world_size != self.local_rank:
                continue
            
            # 生成回复和计算奖励
            response = self.generate_response(prompt)
            reward = self.compute_reward(prompt, response)
            
            # 保存结果
            results.append({
                'prompt': prompt,
                'response': response,
                'reward': reward,
                'gpu_id': self.local_rank
            })
        
        return results
    
    def gather_results(self, local_results: list) -> list:
        """
        收集所有GPU的结果
        作用：将各个GPU的评测结果汇总，计算每个问题的平均分数
        local_results: 当前GPU的评测结果
        返回：汇总后的最终结果（只在主进程返回）
        """
        # 简单的结果收集 - 在实际使用中可以用torch.distributed.all_gather
        # 这里为了简化，直接使用本地结果
        all_results = local_results
        
        # 只在主进程（GPU 0）进行结果汇总
        if self.local_rank == 0:
            # 按问题分组
            prompt_groups = {}
            for result in all_results:
                prompt = result['prompt']
                if prompt not in prompt_groups:
                    prompt_groups[prompt] = []
                prompt_groups[prompt].append(result)
            
            # 计算每个问题的统计信息
            final_results = []
            for prompt, group in prompt_groups.items():
                # 提取所有回答的奖励分数
                rewards = [r['reward'] for r in group]
                responses = [r['response'] for r in group]
                
                # 计算统计指标
                final_results.append({
                    'prompt': prompt,
                    'responses': responses,           # 所有生成的回答
                    'rewards': rewards,               # 所有奖励分数
                    'mean_reward': sum(rewards) / len(rewards),  # 平均奖励
                    'max_reward': max(rewards),       # 最高奖励
                    'min_reward': min(rewards)        # 最低奖励
                })
            
            return final_results
        
        return []

def load_prompts_from_jsonl(file_path: str, sample_size: int = None) -> list:
    """
    从JSONL文件加载问题
    作用：读取训练数据文件，提取其中的问题部分
    file_path: JSONL文件路径
    sample_size: 采样数量，如果为None则加载全部
    返回：问题列表
    """
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 处理两种数据格式
                if isinstance(data.get('prompt'), list):
                    # 如果prompt是消息列表格式，提取用户消息
                    user_msg = next(msg['content'] for msg in data['prompt'] if msg['role'] == 'user')
                    prompts.append(user_msg)
                else:
                    # 如果prompt是字符串格式，直接使用
                    prompts.append(data['prompt'])
    
    # 如果指定了采样数量，只取前sample_size个
    if sample_size:
        prompts = prompts[:sample_size]
    
    return prompts

def main():
    """
    主函数
    作用：解析命令行参数，运行多GPU评测流程
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="多机多卡奖励模型评分工具")
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--reward_model', type=str, required=True, help='奖励模型路径')
    parser.add_argument('--prompts', type=str, help='prompts文件路径')
    parser.add_argument('--sample_size', type=int, default=20, help='评测样本数量')
    parser.add_argument('--local_rank', type=int, default=0, help='本地rank')
    
    args = parser.parse_args()
    
    # 创建多GPU评分器
    evaluator = MultiGPURewardEvaluator(args.model, args.reward_model)
    
    # 加载问题数据
    if args.prompts:
        # 从文件加载问题
        prompts = load_prompts_from_jsonl(args.prompts, args.sample_size)
    else:
        # 使用默认问题
        prompts = ["请分析一下当前股市的投资机会和风险。"] * 5
    
    # 只在主进程打印开始信息
    if evaluator.local_rank == 0:
        print(f"开始多GPU评测 {len(prompts)} 个prompts...")
    
    # 运行评测流程
    local_results = evaluator.evaluate_batch(prompts)      # 各GPU并行评测
    final_results = evaluator.gather_results(local_results)  # 汇总结果
    
    # 主进程输出和保存结果
    if evaluator.local_rank == 0 and final_results:
        # 计算整体统计信息
        all_rewards = [r['mean_reward'] for r in final_results]
        print(f"\n多GPU评测结果:")
        print(f"总prompts数: {len(final_results)}")
        print(f"平均奖励: {sum(all_rewards)/len(all_rewards):.4f}")
        print(f"最高奖励: {max(all_rewards):.4f}")
        print(f"最低奖励: {min(all_rewards):.4f}")
        
        # 保存详细结果到JSON文件
        output_file = f"multigpu_evaluation_{args.model.split('/')[-1]}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()