#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理器 - 简化版本
负责处理SFT数据和基础文章数据

【作用】
这个模块是整个数据生成流程的"数据管家"，负责：
1. 加载各种格式的数据文件（SFT数据、文章数据）
2. 从已有数据中提取有用信息（如问题）
3. 对数据进行采样和混合处理
4. 为其他模块提供标准化的数据格式

【在整体程序中的作用】
- 在问题生成阶段：加载SFT数据和文章数据，提取和生成问题
- 为问题生成器提供原始材料
- 统一数据格式，让其他模块易于使用
- 实现数据的随机采样和混合，确保数据多样性

【为什么需要数据处理器】
不同来源的数据格式可能不同，需要统一处理
而且需要从大量数据中选择合适的样本，不是所有数据都要用
"""

import logging
from typing import List, Dict, Any
from utils import load_jsonl_file, random_sample_with_seed, mix_data_evenly, validate_file_paths

class DataProcessor:
    """
    数据处理器类
    
    【核心功能】
    1. 数据文件的加载和验证
    2. SFT数据中问题的提取
    3. 文章数据的采样处理
    4. 不同来源数据的混合
    5. 数据格式的标准化
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化数据处理器
        
        【作用】
        设置数据处理器的基本参数，特别是随机种子
        
        【为什么要设置随机种子】
        随机种子确保每次运行程序时，随机采样的结果都相同
        这对于实验的可重现性非常重要
        
        【在整体程序中的作用】
        - 为整个数据处理流程做准备
        - 确保数据处理的一致性和可重现性
        
        Args:
            seed: 随机种子，确保结果可复现
        """
        self.seed = seed
        logging.info(f"数据处理器初始化，随机种子: {seed}")
    
    def load_sft_data(self, sft_file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        加载SFT数据文件
        
        【作用】
        SFT（Supervised Fine-Tuning）数据是已经训练好的对话数据，
        这个函数的任务是：
        1. 验证所有SFT文件是否存在
        2. 逐个加载每个文件的内容
        3. 将所有文件的数据合并成一个大列表
        4. 记录加载过程和统计信息
        
        【什么是SFT数据】
        SFT数据通常包含用户问题(input)和AI回答(output)的对话对
        我们主要用其中的问题部分来丰富我们的问题库
        
        【在整体程序中的作用】
        - 为问题生成阶段提供高质量的已有问题
        - 确保生成的数据集包含经过验证的问题类型
        - 平衡新生成问题和已有问题的比例
        
        Args:
            sft_file_paths: SFT数据文件路径列表
            
        Returns:
            合并后的SFT数据列表
        """
        # 验证文件路径
        # 【为什么要验证】避免程序运行到一半才发现文件不存在
        valid_paths = validate_file_paths(sft_file_paths)
        if not valid_paths:
            raise ValueError("没有找到有效的SFT数据文件")
        
        # 加载所有文件并合并
        all_sft_data = []
        for file_path in valid_paths:
            # 加载单个文件
            data = load_jsonl_file(file_path)
            all_sft_data.extend(data)
            logging.info(f"从 {file_path} 加载了 {len(data)} 条SFT数据")
        
        logging.info(f"总共加载了 {len(all_sft_data)} 条SFT数据")
        return all_sft_data
    
    def load_base_articles(self, base_articles_path: str) -> List[Dict[str, Any]]:
        """
        加载基础文章数据
        
        【作用】
        基础文章是用来生成新问题的原始材料，通常是金融新闻或分析文章
        这个函数负责：
        1. 验证文章文件是否存在
        2. 加载文章内容
        3. 为后续问题生成做准备
        
        【什么是基础文章】
        基础文章通常是英文的金融新闻，包含市场动态、公司分析、
        经济指标等信息，我们从中提取关键信息生成中文问题
        
        【在整体程序中的作用】
        - 为问题生成器提供丰富的素材
        - 确保生成的问题具有时效性和专业性
        - 扩展问题的多样性和覆盖面
        
        Args:
            base_articles_path: 基础文章文件路径
            
        Returns:
            基础文章数据列表
        """
        # 验证文件是否存在
        if not validate_file_paths([base_articles_path]):
            raise ValueError(f"基础文章文件不存在: {base_articles_path}")
        
        # 加载文章数据
        articles = load_jsonl_file(base_articles_path)
        logging.info(f"加载了 {len(articles)} 篇基础文章")
        return articles
    
    def extract_questions_from_sft(self, sft_data: List[Dict[str, Any]], sample_size: int) -> List[str]:
        """
        从SFT数据中提取问题（只提取input字段，不包含instruction）
        
        【作用】
        这是一个关键的数据处理函数，它的任务是：
        1. 从SFT数据中随机选择指定数量的样本
        2. 从每个样本中提取用户的问题部分
        3. 过滤掉无效或过短的问题
        4. 返回清洗后的问题列表
        
        【为什么只提取input字段】
        SFT数据的结构通常是：
        - input: 用户的实际问题
        - output: AI的回答
        - instruction: 系统指令
        我们只需要真实的用户问题，不需要系统指令
        
        【在整体程序中的作用】
        - 从已有的高质量对话数据中提取问题
        - 为问题库提供经过验证的问题类型
        - 确保生成的数据集包含多样化的问题
        
        Args:
            sft_data: SFT数据列表
            sample_size: 需要采样的问题数量
            
        Returns:
            问题列表
        """
        # 随机采样SFT数据
        # 【为什么要采样】通常SFT数据很多，我们只需要其中一部分
        # 采样可以控制数据量，也能保证随机性
        sampled_sft = random_sample_with_seed(sft_data, sample_size, self.seed)
        
        # 提取问题（只使用input字段）
        questions = []
        for item in sampled_sft:
            question = ""
            
            # 优先使用input字段，这是实际的问题内容
            # 【数据字段的优先级】
            # input > question > 其他字段
            if 'input' in item:
                question = item['input'].strip()
            elif 'question' in item:
                question = item['question'].strip()
            else:
                logging.warning(f"无法从SFT数据中提取问题: {item}")
                continue
            
            # 验证问题质量
            if question:
                questions.append(question)
            else:
                logging.warning(f"提取的问题为空: {item}")
        
        logging.info(f"从SFT数据中提取了 {len(questions)} 个问题")
        return questions
    
    def sample_articles_for_generation(self, articles: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """
        为问题生成采样基础文章
        
        【作用】
        从大量的基础文章中选择一部分用于问题生成：
        1. 根据需要生成的问题数量决定采样数量
        2. 随机选择文章，确保多样性
        3. 为每篇文章添加索引，方便追踪
        
        【为什么要采样】
        - 基础文章可能有成千上万篇，全部处理耗时太长
        - 采样可以控制处理时间和API成本
        - 随机采样保证了问题的多样性
        
        【采样策略】
        使用固定随机种子的采样，确保结果可重现
        这样每次运行程序都会选择相同的文章
        
        【在整体程序中的作用】
        - 为问题生成器提供合适数量的文章素材
        - 控制问题生成的工作量和成本
        - 确保文章选择的随机性和可重现性
        
        Args:
            articles: 基础文章列表
            sample_size: 采样数量
            
        Returns:
            采样后的文章列表
        """
        sampled_articles = random_sample_with_seed(articles, sample_size, self.seed)
        logging.info(f"为问题生成采样了 {len(sampled_articles)} 篇文章")
        return sampled_articles
    
    def combine_questions(self, sft_questions: List[str], generated_questions: List[str]) -> List[str]:
        """
        合并SFT问题和新生成的问题
        
        【作用】
        这是数据处理的最后一步，将两种来源的问题合并：
        1. 来自SFT数据的高质量问题
        2. 从文章新生成的问题
        3. 均匀混合，打乱顺序
        4. 返回最终的问题列表
        
        【为什么要混合】
        - SFT问题质量高但可能类型有限
        - 新生成的问题更有针对性但质量可能不稳定
        - 混合两者可以取长补短，提高数据集质量
        
        【混合策略】
        使用均匀混合算法，确保两种类型的问题分布均匀
        不是简单拼接，而是交替分布
        
        【在整体程序中的作用】
        - 完成问题生成阶段的最终整合
        - 为答案生成阶段提供完整的问题列表
        - 确保问题的多样性和质量平衡
        
        Args:
            sft_questions: 从SFT数据提取的问题
            generated_questions: 新生成的问题
            
        Returns:
            混合后的问题列表
        """
        # 使用工具函数进行均匀混合
        # 【mix_data_evenly函数的作用】
        # 1. 将两个列表合并
        # 2. 使用固定种子打乱顺序
        # 3. 确保结果可重现
        combined_questions = mix_data_evenly(sft_questions, generated_questions, self.seed)
        logging.info(f"合并了 {len(sft_questions)} + {len(generated_questions)} = {len(combined_questions)} 个问题")
        return combined_questions
