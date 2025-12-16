#!/usr/bin/env python3
"""
金融模型量化效果评估系统

这个脚本的主要作用：
1. 全面评估量化模型与原始模型的性能差异
2. 从三个维度进行评估：输出一致性、稳定性、推理性能
3. 生成详细的评估报告，帮助判断量化效果是否满足要求
4. 为模型部署决策提供数据支持

评估方法说明：
- 输出一致性：对比两个模型在相同输入下的输出相似度
- 稳定性：测试模型在多次生成中的一致性
- 推理性能：测试模型的推理速度和吞吐量
"""

import json
import random
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from gptqmodel import GPTQModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

# 配置日志系统：设置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """
    评估配置类
    
    这个类定义了评估过程中的所有配置参数
    使用dataclass装饰器自动生成构造函数和其他方法
    """
    # 模型路径配置
    original_model_path: str = "/shared/grpo_financial_tuning/output/best_model"      # 原始模型路径
    quantized_model_path: str = "/shared/gptq_model/output/best_model_gptq_int4"     # 量化模型路径
    
    # 数据和输出配置
    test_data_path: str = "/shared/gptq_model/grpo_prompts_dataset_5k.jsonl"         # 测试数据路径
    output_dir: str = "./evaluation_results"                                          # 结果输出目录
    
    # 评估参数配置
    num_test_samples: int = 10           # 测试样本数量：用于一致性评估的样本数
    num_consistency_trials: int = 10     # 一致性测试轮数：每个样本重复生成的次数
    max_new_tokens: int = 512           # 最大生成token数：控制生成文本的长度
    temperature: float = 0.7            # 温度参数：控制生成文本的随机性，0.7是平衡值
    random_seed: int = 42               # 随机种子：确保实验结果可复现
    warmup_rounds: int = 3              # 预热轮数：GPU预热次数，提高计时准确性

class FinancialModelEvaluator:
    """
    金融模型评估器主类
    
    功能概述：
    1. 加载原始模型和量化模型
    2. 执行三个维度的评估
    3. 生成详细的评估报告
    4. 保存评估结果
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        初始化评估器
        
        参数说明：
        config: 评估配置对象，包含所有评估参数
        """
        self.config = config  # 存储配置对象
        
        # 语义相似度模型：用于计算文本之间的语义相似度
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 模型存储变量：初始化为None，在load_models方法中赋值
        self.original_model = None      # 原始模型
        self.quantized_model = None     # 量化模型
        self.original_tokenizer = None  # 原始模型的分词器
        self.quantized_tokenizer = None # 量化模型的分词器
        
        # 设置随机种子，确保结果可复现
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(exist_ok=True)
    
    def load_models(self):
        """
        加载原始模型和量化模型
        
        这是评估的第一步，需要将两个模型都加载到内存中
        加载过程包括：
        1. 加载分词器（将文本转换为token）
        2. 加载模型权重
        3. 设置合适的精度和设备分配
        """
        logger.info("开始加载模型...")
        
        try:
            # 加载原始模型
            logger.info("正在加载原始模型...")
            # 加载分词器：用于文本预处理
            self.original_tokenizer = AutoTokenizer.from_pretrained(self.config.original_model_path)
            # 加载模型：使用transformers库加载标准模型
            self.original_model = AutoModelForCausalLM.from_pretrained(
                self.config.original_model_path,  # 模型路径
                device_map="auto",                # 自动分配设备（CPU/GPU）
                torch_dtype=torch.bfloat16,       # 使用bfloat16精度节省内存
                trust_remote_code=True            # 允许执行自定义代码
            )
            logger.info("原始模型加载完成")
            
            # 清理GPU缓存，为加载第二个模型腾出空间
            torch.cuda.empty_cache()
            
            # 加载量化模型
            logger.info("正在加载量化模型...")
            # 加载量化模型的分词器
            self.quantized_tokenizer = AutoTokenizer.from_pretrained(self.config.quantized_model_path)
            # 加载量化模型：使用GPTQModel框架加载量化后的模型
            self.quantized_model = GPTQModel.from_quantized(
                self.config.quantized_model_path,  # 量化模型路径
                device_map="auto",                 # 自动分配设备
                torch_dtype=torch.bfloat16         # 使用bfloat16精度
            )
            logger.info("量化模型加载完成")
            
            # 记录模型设备分布信息
            self._log_model_device_info()
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _log_model_device_info(self):
        """
        记录模型设备分布信息
        
        功能：
        1. 显示GPU内存使用情况
        2. 帮助用户了解模型在设备上的分布
        3. 用于调试和优化内存使用
        """
        logger.info("=== 模型设备分布信息 ===")
        
        # 检查是否有可用的GPU
        if torch.cuda.is_available():
            # 遍历所有GPU设备
            for i in range(torch.cuda.device_count()):
                # 获取已分配的GPU内存（单位：字节）
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # 转换为GB
                # 获取已缓存的GPU内存（单位：字节）
                memory_cached = torch.cuda.memory_reserved(i) / 1024**3      # 转换为GB
                logger.info(f"GPU {i}: 已分配 {memory_allocated:.2f}GB, 已缓存 {memory_cached:.2f}GB")
    
    def load_test_data(self) -> List[str]:
        """
        加载测试数据
        
        功能：
        1. 从指定文件加载测试数据
        2. 过滤掉质量不好的数据
        3. 随机选择指定数量的样本
        4. 返回用于评估的prompt列表
        
        返回：
        prompts: 测试prompt列表，每个元素是一个字符串
        """
        logger.info(f"从 {self.config.test_data_path} 加载测试数据...")
        
        prompts = []  # 存储所有有效prompt的列表
        
        # 逐行读取测试数据文件
        with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 解析JSON格式的数据
                    data = json.loads(line.strip())
                    # 提取prompt字段
                    prompt = data.get('prompt', '').strip()
                    # 过滤条件：prompt不为空且长度大于50个字符
                    if prompt and len(prompt) > 50:
                        prompts.append(prompt)
                except:
                    # 如果数据格式错误，跳过该行
                    continue
        
        # 随机抽取指定数量的样本
        if len(prompts) > self.config.num_test_samples:
            prompts = random.sample(prompts, self.config.num_test_samples)
        
        logger.info(f"成功加载了 {len(prompts)} 个测试样本")
        return prompts
    
    def generate_response(self, model, tokenizer, prompt: str) -> str:
        """
        生成模型响应
        
        这是核心的文本生成方法，用于获取模型对给定prompt的响应
        
        参数说明：
        model: 要使用的模型（原始模型或量化模型）
        tokenizer: 对应的分词器
        prompt: 输入的提示文本
        
        返回：
        response: 模型生成的响应文本（去除了原始prompt）
        """
        # 将文本转换为模型输入格式
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 限制输入长度，避免超出模型最大长度限制
        max_input_length = 1536  # 为生成的token预留空间
        if inputs['input_ids'].shape[1] > max_input_length:
            # 如果输入过长，进行截断
            inputs['input_ids'] = inputs['input_ids'][:, :max_input_length]
            inputs['attention_mask'] = inputs['attention_mask'][:, :max_input_length]
        
        # 生成文本
        with torch.no_grad():  # 禁用梯度计算，节省内存
            outputs = model.generate(
                **inputs,                                    # 输入数据
                max_new_tokens=self.config.max_new_tokens,   # 最大生成token数
                temperature=self.config.temperature,         # 温度参数：控制随机性
                do_sample=True,                              # 启用采样：产生多样化的输出
                pad_token_id=tokenizer.eos_token_id,        # 填充token ID
                repetition_penalty=1.1,                     # 重复惩罚：减少重复内容
                top_p=0.9                                   # 核采样：保留概率前90%的token
            )
        
        # 解码生成的token为文本
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始prompt，只保留生成的部分
        response = response[len(prompt):].strip()
        
        return response
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本之间的语义相似度
        
        使用预训练的sentence-transformers模型计算语义相似度
        这比简单的文本匹配更能反映内容的实际相似性
        
        参数说明：
        text1: 第一个文本
        text2: 第二个文本
        
        返回：
        similarity: 相似度分数，范围[0,1]，1表示完全相似
        """
        try:
            # 将文本转换为语义向量
            embeddings1 = self.similarity_model.encode([text1])
            embeddings2 = self.similarity_model.encode([text2])
            
            # 计算余弦相似度
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            return 0.0
    
    def evaluate_output_consistency(self, prompts: List[str]) -> Dict:
        """
        评估输出一致性
        
        这是核心评估方法之一，用于测试量化模型和原始模型的输出相似度
        
        评估原理：
        1. 对每个prompt，让两个模型都生成响应
        2. 计算两个响应之间的语义相似度
        3. 统计所有样本的相似度分布
        4. 评估量化是否显著影响了模型输出
        
        参数说明：
        prompts: 测试prompt列表
        
        返回：
        consistency_stats: 包含相似度统计信息的字典
        """
        logger.info("=== 开始输出一致性评估 ===")
        
        similarities = []        # 存储所有相似度分数
        detailed_results = []    # 存储详细的对比结果
        
        # 对每个prompt进行评估
        for i, prompt in enumerate(prompts):
            logger.info(f"处理第 {i+1}/{len(prompts)} 个样本...")
            
            try:
                # 生成原始模型的响应
                original_response = self.generate_response(
                    self.original_model, self.original_tokenizer, prompt
                )
                
                # 生成量化模型的响应
                quantized_response = self.generate_response(
                    self.quantized_model, self.quantized_tokenizer, prompt
                )
                
                # 计算语义相似度
                similarity = self.calculate_semantic_similarity(original_response, quantized_response)
                similarities.append(similarity)
                
                # 记录详细结果（用于后续分析）
                detailed_results.append({
                    "prompt": prompt[:100] + "...",  # 截断prompt用于显示
                    "original_response": original_response,
                    "quantized_response": quantized_response,
                    "similarity": similarity
                })
                
                logger.info(f"相似度: {similarity:.3f}")
                
            except Exception as e:
                logger.error(f"处理样本 {i+1} 时出错: {e}")
                continue
        
        # 计算统计指标
        consistency_stats = {
            "mean_similarity": np.mean(similarities),      # 平均相似度
            "std_similarity": np.std(similarities),        # 相似度标准差
            "min_similarity": np.min(similarities),        # 最小相似度
            "max_similarity": np.max(similarities),        # 最大相似度
            "percentiles": {                               # 百分位数分析
                "25th": np.percentile(similarities, 25),   # 25%分位数
                "50th": np.percentile(similarities, 50),   # 中位数
                "75th": np.percentile(similarities, 75),   # 75%分位数
                "90th": np.percentile(similarities, 90)    # 90%分位数
            },
            "samples_processed": len(similarities),        # 成功处理的样本数
            "detailed_results": detailed_results           # 详细结果
        }
        
        logger.info(f"一致性评估完成，平均相似度: {consistency_stats['mean_similarity']:.3f}")
        return consistency_stats
    
    def evaluate_stability(self, test_prompts: List[str]) -> Dict:
        """
        评估模型稳定性（自一致性）
        
        稳定性评估的目的：
        1. 测试模型在相同输入下的输出一致性
        2. 评估量化是否增加了模型的不稳定性
        3. 比较原始模型和量化模型的稳定性保持程度
        
        评估方法：
        1. 选择部分prompt
        2. 每个prompt重复生成多次
        3. 计算多次生成结果之间的相似度
        4. 对比两个模型的稳定性
        
        参数说明：
        test_prompts: 测试prompt列表
        
        返回：
        stability_stats: 包含稳定性统计信息的字典
        """
        logger.info("=== 开始稳定性评估 ===")
        
        # 选择部分prompt进行稳定性测试（稳定性测试计算量大）
        stability_prompts = random.sample(test_prompts, min(10, len(test_prompts)))
        
        original_stability_scores = []   # 原始模型的稳定性分数
        quantized_stability_scores = []  # 量化模型的稳定性分数
        
        # 对每个prompt进行稳定性测试
        for i, prompt in enumerate(stability_prompts):
            logger.info(f"稳定性测试 {i+1}/{len(stability_prompts)}")
            
            # 测试原始模型的稳定性
            original_responses = []
            for trial in range(self.config.num_consistency_trials):
                response = self.generate_response(self.original_model, self.original_tokenizer, prompt)
                original_responses.append(response)
            
            # 计算原始模型的自一致性分数
            original_stability = self._calculate_self_consistency(original_responses)
            original_stability_scores.append(original_stability)
            
            # 测试量化模型的稳定性
            quantized_responses = []
            for trial in range(self.config.num_consistency_trials):
                response = self.generate_response(self.quantized_model, self.quantized_tokenizer, prompt)
                quantized_responses.append(response)
            
            # 计算量化模型的自一致性分数
            quantized_stability = self._calculate_self_consistency(quantized_responses)
            quantized_stability_scores.append(quantized_stability)
            
            logger.info(f"原始模型稳定性: {original_stability:.3f}, 量化模型稳定性: {quantized_stability:.3f}")
        
        # 计算稳定性统计指标
        stability_stats = {
            "original_model": {
                "mean_stability": np.mean(original_stability_scores),  # 原始模型平均稳定性
                "std_stability": np.std(original_stability_scores),    # 原始模型稳定性标准差
                "scores": original_stability_scores                    # 所有稳定性分数
            },
            "quantized_model": {
                "mean_stability": np.mean(quantized_stability_scores), # 量化模型平均稳定性
                "std_stability": np.std(quantized_stability_scores),   # 量化模型稳定性标准差
                "scores": quantized_stability_scores                   # 所有稳定性分数
            },
            # 稳定性保持率：量化模型稳定性 / 原始模型稳定性
            "stability_retention": np.mean(quantized_stability_scores) / np.mean(original_stability_scores)
        }
        
        logger.info(f"稳定性评估完成，稳定性保持率: {stability_stats['stability_retention']:.3f}")
        return stability_stats
    
    def _calculate_self_consistency(self, responses: List[str]) -> float:
        """
        计算自一致性分数
        
        自一致性的定义：同一个prompt多次生成的结果之间的相似度
        
        计算方法：
        1. 对所有响应进行两两相似度计算
        2. 计算所有相似度的平均值
        3. 返回平均相似度作为自一致性分数
        
        参数说明：
        responses: 同一个prompt的多次响应列表
        
        返回：
        self_consistency: 自一致性分数，范围[0,1]
        """
        # 如果响应数量少于2，无法计算相似度
        if len(responses) < 2:
            return 1.0
        
        similarities = []  # 存储所有相似度分数
        
        # 计算所有响应之间的两两相似度
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self.calculate_semantic_similarity(responses[i], responses[j])
                similarities.append(sim)
        
        # 返回平均相似度
        return np.mean(similarities)
    
    def evaluate_performance(self, test_prompts: List[str]) -> Dict:
        """
        评估推理性能
        
        性能评估的目的：
        1. 测试量化模型相比原始模型的速度提升
        2. 评估量化带来的实际性能收益
        3. 为部署决策提供性能数据
        
        评估方法：
        1. 分别测试两个模型的推理时间
        2. 计算加速比
        3. 进行统计分析
        
        参数说明：
        test_prompts: 测试prompt列表
        
        返回：
        performance_stats: 包含性能统计信息的字典
        """
        logger.info("=== 开始性能评估 ===")
        
        # 选择部分prompt进行性能测试
        performance_prompts = random.sample(test_prompts, min(10, len(test_prompts)))
        
        # 分别测试每个模型，避免交替影响
        original_times = []   # 原始模型推理时间列表
        quantized_times = []  # 量化模型推理时间列表
        
        # 测试原始模型性能
        logger.info("测试原始模型性能...")
        for i, prompt in enumerate(performance_prompts):
            logger.info(f"原始模型测试 {i+1}/{len(performance_prompts)}")
            
            # GPU预热：避免冷启动影响计时准确性
            inputs = self.original_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.original_model.device)
            with torch.no_grad():
                for _ in range(self.config.warmup_rounds):
                    self.original_model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            # 准确计时：使用GPU同步确保计时准确
            torch.cuda.synchronize()  # 等待GPU操作完成
            start_time = time.time()
            _ = self.generate_response(self.original_model, self.original_tokenizer, prompt)
            torch.cuda.synchronize()  # 等待GPU操作完成
            original_time = time.time() - start_time
            original_times.append(original_time)
            
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
            # 测试量化模型性能
        logger.info("测试量化模型性能...")
        for i, prompt in enumerate(performance_prompts):
            logger.info(f"量化模型测试 {i+1}/{len(performance_prompts)}")
            
            # GPU预热
            inputs = self.quantized_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.quantized_model.device)
            with torch.no_grad():
                for _ in range(self.config.warmup_rounds):
                    self.quantized_model.generate(**inputs, max_new_tokens=50, do_sample=False)
            
            # 准确计时
            torch.cuda.synchronize()
            start_time = time.time()
            _ = self.generate_response(self.quantized_model, self.quantized_tokenizer, prompt)
            torch.cuda.synchronize()
            quantized_time = time.time() - start_time
            quantized_times.append(quantized_time)
        
        # 计算性能统计指标
        performance_stats = {
            "original_model": {
                "mean_time": np.mean(original_times),   # 原始模型平均时间
                "std_time": np.std(original_times),     # 原始模型时间标准差
                "times": original_times                 # 所有时间记录
            },
            "quantized_model": {
                "mean_time": np.mean(quantized_times),  # 量化模型平均时间
                "std_time": np.std(quantized_times),    # 量化模型时间标准差
                "times": quantized_times                # 所有时间记录
            },
            # 加速比：原始模型时间 / 量化模型时间
            "speedup": np.mean(original_times) / np.mean(quantized_times)
        }
        
        logger.info(f"原始模型平均时间: {performance_stats['original_model']['mean_time']:.2f}秒")
        logger.info(f"量化模型平均时间: {performance_stats['quantized_model']['mean_time']:.2f}秒")
        logger.info(f"加速比: {performance_stats['speedup']:.2f}x")
        
        return performance_stats
    
    def run_complete_evaluation(self) -> Dict:
        """
        运行完整的评估流程
        
        这是主要的评估方法，按顺序执行所有评估步骤：
        1. 加载模型
        2. 加载测试数据
        3. 执行三个维度的评估
        4. 生成综合报告
        5. 保存结果
        
        返回：
        results: 包含所有评估结果的字典
        """
        logger.info("=== 开始金融模型量化评估 ===")
        
        # 步骤1: 加载模型
        self.load_models()
        
        # 步骤2: 加载测试数据
        test_prompts = self.load_test_data()
        
        # 步骤3: 执行各项评估
        results = {
            "config": {
                "num_test_samples": len(test_prompts),
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "random_seed": self.config.random_seed
            },
            # 一致性评估：测试输出相似度
            "consistency_evaluation": self.evaluate_output_consistency(test_prompts),
            # 稳定性评估：测试输出稳定性
            "stability_evaluation": self.evaluate_stability(test_prompts),
            # 性能评估：测试推理速度
            "performance_evaluation": self.evaluate_performance(test_prompts)
        }
        
        # 步骤4: 保存结果
        self._save_results(results)
        
        # 步骤5: 生成评估报告
        self._generate_report(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """
        保存评估结果到JSON文件
        
        参数说明：
        results: 包含所有评估结果的字典
        """
        output_file = Path(self.config.output_dir) / "evaluation_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")
    
    def _generate_report(self, results: Dict):
        """
        生成评估报告
        
        根据评估结果生成人类可读的报告，包括：
        1. 基本配置信息
        2. 各项评估指标
        3. 综合评估结论
        4. 部署建议
        
        参数说明：
        results: 包含所有评估结果的字典
        """
        # 提取各项评估结果
        consistency = results["consistency_evaluation"]
        stability = results["stability_evaluation"]
        performance = results["performance_evaluation"]
        
        # 生成报告文本
        report = f"""
=== 金融模型量化评估报告 ===

评估配置:
- 测试样本数量: {results['config']['num_test_samples']}
- 生成token数: {results['config']['max_new_tokens']}
- 温度参数: {results['config']['temperature']}

输出一致性评估:
- 平均语义相似度: {consistency['mean_similarity']:.3f}
- 相似度标准差: {consistency['std_similarity']:.3f}
- 最低相似度: {consistency['min_similarity']:.3f}
- 90%分位数: {consistency['percentiles']['90th']:.3f}

稳定性评估:
- 原始模型自一致性: {stability['original_model']['mean_stability']:.3f}
- 量化模型自一致性: {stability['quantized_model']['mean_stability']:.3f}
- 稳定性保持率: {stability['stability_retention']:.3f}

性能评估:
- 原始模型平均推理时间: {performance['original_model']['mean_time']:.2f}秒
- 量化模型平均推理时间: {performance['quantized_model']['mean_time']:.2f}秒
- 推理速度提升: {performance['speedup']:.2f}倍

综合评估结论:
"""
        
        # 根据评估结果添加结论
        if consistency['mean_similarity'] > 0.85:
            report += "输出一致性: 优秀 (相似度 > 0.85)\n"
        elif consistency['mean_similarity'] > 0.8:
            report += "输出一致性: 良好 (相似度 > 0.8)\n"
        else:
            report += "输出一致性: 需要改进 (相似度 < 0.8)\n"
        
        if stability['stability_retention'] > 0.9:
            report += "稳定性保持: 优秀 (保持率 > 90%)\n"
        elif stability['stability_retention'] > 0.8:
            report += "稳定性保持: 良好 (保持率 > 80%)\n"
        else:
            report += "稳定性保持: 需要改进 (保持率 < 80%)\n"
        
        if performance['speedup'] > 2.0:
            report += "性能提升: 优秀 (加速比 > 2x)\n"
        elif performance['speedup'] > 1.5:
            report += "性能提升: 良好 (加速比 > 1.5x)\n"
        else:
            report += "性能提升: 不明显 (加速比 < 1.5x)\n"
        
        # 最终建议
        if (consistency['mean_similarity'] > 0.85 and 
            stability['stability_retention'] > 0.8 and 
            performance['speedup'] > 1.5):
            report += "\n建议: 量化效果良好，可以考虑部署到生产环境\n"
        else:
            report += "\n建议: 量化效果有待改进，建议调整量化参数或方法\n"
        
        print(report)
        
        # 保存报告到文件
        report_file = Path(self.config.output_dir) / "evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {report_file}")

def main():
    """
    主函数
    
    创建评估配置和评估器，执行完整的评估流程
    """
    # 创建评估配置
    config = EvaluationConfig(
        test_data_path="/shared/gptq_model/grpo_prompts_dataset_5k.jsonl",
        output_dir="./evaluation_results",
        num_test_samples=20,        # 可调整测试样本数
        warmup_rounds=5,            # 预热轮数
        random_seed=42
    )
    
    # 创建评估器
    evaluator = FinancialModelEvaluator(config)
    
    try:
        # 运行评估
        results = evaluator.run_complete_evaluation()
        logger.info("评估完成!")
        
    except Exception as e:
        logger.error(f"评估过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
