#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据拆分器 - 将问题-五答案数据拆分为训练/验证/测试集
并生成对应的偏好对格式数据

【作用】
这个模块是整个数据生成流程的最后一步，负责：
1. 将完整的问题-答案数据按比例分割成不同用途的数据集
2. 将数据转换为奖励模型训练需要的偏好对格式
3. 生成详细的统计报告，帮助用户了解数据分布
4. 为不同的训练阶段提供标准化的数据文件

【为什么要分割数据集】
机器学习的标准做法是将数据分为三部分：
- 训练集（80%）：用于训练奖励模型
- 验证集（10%）：用于调整模型参数
- 测试集（10%）：用于最终评估模型性能

【在整体程序中的作用】
- 完成整个数据生成流程的最后一环
- 为奖励模型训练提供标准格式的数据
- 确保训练数据的科学性和规范性
"""

import json
import random
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from utils import ensure_directory_exists, save_jsonl_file, load_jsonl_file

class DataSplitter:
    """
    数据拆分器类
    
    【核心功能】
    1. 加载完整的问题-答案数据
    2. 按比例随机分割数据集
    3. 生成偏好对格式的训练数据
    4. 保存不同格式的数据文件
    5. 生成详细的统计报告
    """
    
    def __init__(self, input_dir: str, output_dir: str, random_seed: int = 42):
        """
        初始化数据拆分器
        
        【作用】
        设置数据拆分器的基本参数和工作目录
        
        【为什么需要输入和输出目录】
        - input_dir: 存放完整数据的目录（来自答案生成阶段）
        - output_dir: 存放分割后数据的目录（用于后续训练）
        
        【在整体程序中的作用】
        - 衔接答案生成和模型训练两个阶段
        - 为数据分割过程做准备
        - 确保分割结果的可重现性
        
        Args:
            input_dir: 输入目录路径（包含答案数据文件）
            output_dir: 输出目录路径（保存分割后的数据）
            random_seed: 随机种子，确保结果可重现
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # 设置随机种子
        # 【确保可重现性】相同的种子会产生相同的分割结果
        random.seed(random_seed)
        
        print(f"数据拆分器初始化:")
        print(f"  输入目录: {self.input_dir}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  随机种子: {self.random_seed}")
    
    def load_all_answers_data(self) -> List[Dict[str, Any]]:
        """
        加载所有的answers_batch文件数据或合并文件数据
        
        【作用】
        从答案生成阶段的输出中加载完整数据：
        1. 优先查找合并后的完整文件
        2. 如果没有合并文件，则加载所有批次文件
        3. 验证数据的完整性和格式
        4. 返回统一格式的数据列表
        
        【为什么要支持两种加载方式】
        - 如果答案生成正常完成，会有合并文件，直接加载更快
        - 如果程序中断，可能只有批次文件，需要逐个加载
        
        【数据格式说明】
        每条数据包含：
        - question_id: 问题唯一标识
        - question: 问题内容
        - answers: 5个不同质量等级的答案列表
        
        【在整体程序中的作用】
        - 为数据分割提供输入数据
        - 确保数据加载的灵活性和鲁棒性
        - 验证上一阶段的输出质量
        
        Returns:
            合并后的所有问题-答案数据
        """
        all_data = []
        
        # 首先检查是否存在合并文件
        merged_file = self.input_dir / "complete_qa_dataset.jsonl"
        if merged_file.exists():
            print(f"找到合并文件: {merged_file.name}")
            data = load_jsonl_file(str(merged_file))
            all_data.extend(data)
            print(f"从合并文件加载了 {len(data)} 个问题-答案组合")
        else:
            # 查找所有answers_batch文件
            # 【批次文件的命名规则】answers_batch_1.jsonl, answers_batch_2.jsonl, ...
            answers_files = list(self.input_dir.glob("answers_batch_*.jsonl"))
            
            if not answers_files:
                raise FileNotFoundError(f"在 {self.input_dir} 中未找到 complete_qa_dataset.jsonl 或 answers_batch_*.jsonl 文件")
            
            print(f"找到 {len(answers_files)} 个answers_batch文件:")
            
            # 按文件名排序，确保加载顺序一致
            for file_path in sorted(answers_files):
                print(f"  - {file_path.name}")
                data = load_jsonl_file(str(file_path))
                all_data.extend(data)
        
        print(f"总共加载了 {len(all_data)} 个问题-答案组合")
        return all_data
    
    def split_data(self, data: List[Dict[str, Any]], 
                   train_ratio: float = 0.8, 
                   eval_ratio: float = 0.1, 
                   test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        将数据按比例拆分为训练/验证/测试集
        
        【作用】
        这是数据分割的核心方法：
        1. 验证分割比例的合理性
        2. 随机打乱数据顺序，避免偏差
        3. 按比例计算各个集合的大小
        4. 将数据分配到不同的集合中
        
        【为什么要随机打乱】
        原始数据可能有顺序偏差（比如简单问题在前，复杂问题在后）
        随机打乱确保每个数据集都包含各种类型的问题
        
        【分割比例的选择】
        8:1:1是机器学习中常用的分割比例：
        - 80%训练集：足够多的数据用于学习
        - 10%验证集：用于调整超参数
        - 10%测试集：用于最终评估
        
        【在整体程序中的作用】
        - 为奖励模型训练提供科学的数据分割
        - 确保模型评估的可靠性
        - 避免数据泄露和过拟合问题
        
        Args:
            data: 原始数据列表
            train_ratio: 训练集比例
            eval_ratio: 验证集比例  
            test_ratio: 测试集比例
            
        Returns:
            (训练集, 验证集, 测试集) 的元组
        """
        # 验证比例
        # 【为什么要验证比例】确保比例加起来等于1，避免数据丢失或重复
        total_ratio = train_ratio + eval_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:  # 允许微小的浮点误差
            raise ValueError(f"比例之和必须为1.0，当前为 {total_ratio}")
        
        # 打乱数据顺序
        # 【深拷贝的重要性】避免修改原始数据，保持数据完整性
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # 计算各集合的大小
        total_size = len(shuffled_data)
        train_size = int(total_size * train_ratio)
        eval_size = int(total_size * eval_ratio)
        # test_size自动为剩余部分，避免舍入误差
        
        # 拆分数据
        # 【索引分割法】使用数组切片进行分割，简单高效
        train_data = shuffled_data[:train_size]
        eval_data = shuffled_data[train_size:train_size + eval_size]
        test_data = shuffled_data[train_size + eval_size:]
        
        # 显示分割结果
        print(f"数据拆分结果:")
        print(f"  训练集: {len(train_data)} 个问题 ({len(train_data)/total_size*100:.1f}%)")
        print(f"  验证集: {len(eval_data)} 个问题 ({len(eval_data)/total_size*100:.1f}%)")
        print(f"  测试集: {len(test_data)} 个问题 ({len(test_data)/total_size*100:.1f}%)")
        
        return train_data, eval_data, test_data
    
    def generate_preference_pairs_from_answers(self, answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从5个答案生成所有可能的偏好对
        
        【作用】
        这是奖励模型训练数据生成的核心方法：
        1. 从5个不同质量的答案中生成所有两两对比的组合
        2. 确定每一对中哪个答案更好（质量等级更高）
        3. 创建"好答案vs差答案"的偏好对
        4. 为每个偏好对添加质量差距信息
        
        【什么是偏好对】
        偏好对是奖励模型训练的标准格式：
        - chosen: 更好的答案
        - rejected: 较差的答案
        - 模型学习给好答案打高分，给差答案打低分
        
        【为什么要生成所有组合】
        从5个答案可以生成C(5,2)=10个偏好对，这样可以：
        - 充分利用数据，增加训练样本
        - 让模型学习不同质量等级之间的差异
        - 提高训练效率和模型性能
        
        【在整体程序中的作用】
        - 将原始问题-答案数据转换为训练格式
        - 为奖励模型提供丰富的对比样本
        - 实现数据格式的标准化转换
        
        Args:
            answers: 5个不同质量等级的答案
            
        Returns:
            偏好对列表
        """
        preference_pairs = []
        
        # 生成所有可能的答案对组合 (C(5,2) = 10对)
        # 【双重循环的逻辑】
        # i从0到4，j从i+1到4，确保不重复且不自比较
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                answer_a = answers[i]
                answer_b = answers[j]
                
                # 跳过有错误的答案
                # 【错误处理】生成过程中可能有些答案生成失败
                if answer_a.get("error") or answer_b.get("error"):
                    continue
                
                # 确定哪个是更好的答案（质量等级更高的）
                # 【偏好判断的依据】完全基于预设的质量等级
                if answer_a["quality_level"] > answer_b["quality_level"]:
                    chosen = answer_a["content"]
                    rejected = answer_b["content"]
                    chosen_level = answer_a["quality_level"]
                    rejected_level = answer_b["quality_level"]
                else:
                    chosen = answer_b["content"]
                    rejected = answer_a["content"]
                    chosen_level = answer_b["quality_level"]
                    rejected_level = answer_a["quality_level"]
                
                # 创建偏好对记录
                preference_pairs.append({
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_level": chosen_level,
                    "rejected_level": rejected_level,
                    "quality_gap": chosen_level - rejected_level  # 质量差距，用于分析
                })
        
        return preference_pairs
    
    def convert_to_preference_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将问题-答案格式转换为偏好对格式
        
        【作用】
        这个方法实现了数据格式的批量转换：
        1. 遍历所有问题-答案数据
        2. 为每个问题生成偏好对
        3. 将偏好对与问题信息结合
        4. 创建最终的训练数据格式
        
        【输入格式】
        问题-答案格式：
        {
          "question_id": 1,
          "question": "问题内容",
          "answers": [答案1, 答案2, 答案3, 答案4, 答案5]
        }
        
        【输出格式】
        偏好对格式：
        {
          "question_id": 1,
          "question": "问题内容",
          "chosen": "更好的答案",
          "rejected": "较差的答案",
          "chosen_level": 4,
          "rejected_level": 2,
          "quality_gap": 2
        }
        
        【在整体程序中的作用】
        - 完成从原始数据到训练数据的转换
        - 大大增加训练样本的数量
        - 为奖励模型提供标准化的输入格式
        
        Args:
            data: 问题-答案格式的数据
            
        Returns:
            偏好对格式的数据
        """
        preference_data = []
        
        for item in data:
            question = item["question"]
            question_id = item["question_id"]
            answers = item["answers"]
            
            # 生成偏好对
            # 【调用前面定义的方法】复用代码，保持逻辑一致性
            preference_pairs = self.generate_preference_pairs_from_answers(answers)
            
            # 为每个偏好对添加问题信息
            # 【数据结构扩展】将偏好对与具体问题关联
            for pair in preference_pairs:
                preference_record = {
                    "question_id": question_id,
                    "question": question,
                    **pair  # 展开偏好对的所有字段
                }
                preference_data.append(preference_record)
        
        return preference_data
    
    def save_dataset(self, data: List[Dict[str, Any]], dataset_name: str, 
                     save_qa_format: bool = True, save_preference_format: bool = True):
        """
        保存数据集到指定目录
        
        【作用】
        这个方法负责将分割后的数据保存为不同格式的文件：
        1. 创建对应的目录结构
        2. 保存问题-答案格式（用于分析和调试）
        3. 保存偏好对格式（用于奖励模型训练）
        4. 确保文件命名的规范性
        
        【目录结构】
        output_dir/
        ├── train/
        │   ├── qa_dataset.jsonl          # 问题-答案格式
        │   └── preference_dataset.jsonl  # 偏好对格式
        ├── eval/
        │   ├── qa_dataset.jsonl
        │   └── preference_dataset.jsonl
        └── test/
        │   ├── qa_dataset.jsonl
        │   └── preference_dataset.jsonl
        
        【为什么要保存两种格式】
        - qa_dataset.jsonl: 便于人工检查和分析
        - preference_dataset.jsonl: 直接用于模型训练
        
        【在整体程序中的作用】
        - 为不同的使用场景提供合适的数据格式
        - 规范化数据文件的组织结构
        - 方便后续的模型训练和评估
        
        Args:
            data: 要保存的数据
            dataset_name: 数据集名称 (train/eval/test)
            save_qa_format: 是否保存问题-答案格式
            save_preference_format: 是否保存偏好对格式
        """
        # 创建目录
        dataset_dir = self.output_dir / dataset_name
        ensure_directory_exists(dataset_dir)
        
        print(f"保存 {dataset_name} 数据集到: {dataset_dir}")
        
        # 保存问题-答案格式
        if save_qa_format:
            qa_file = dataset_dir / "qa_dataset.jsonl"
            save_jsonl_file(data, str(qa_file))
            print(f"  ✅ QA格式: {qa_file.name} ({len(data)} 个问题)")
        
        # 保存偏好对格式
        if save_preference_format:
            preference_data = self.convert_to_preference_format(data)
            preference_file = dataset_dir / "preference_dataset.jsonl"
            save_jsonl_file(preference_data, str(preference_file))
            print(f"  ✅ 偏好对格式: {preference_file.name} ({len(preference_data)} 个偏好对)")

    def generate_summary_report(self, train_data: List[Dict], eval_data: List[Dict], test_data: List[Dict]):
        """
        生成数据拆分的统计报告
        
        【作用】
        创建详细的统计报告，帮助用户了解数据分割的结果：
        1. 基本统计信息（数量、比例）
        2. 质量分布分析
        3. 答案成功率统计
        4. 偏好对数量统计
        5. 数据完整性检查
        
        【报告的用途】
        - 验证数据分割的正确性
        - 分析数据质量分布
        - 为模型训练提供参考信息
        - 发现潜在的数据问题
        
        【统计维度】
        - 数量统计：每个集合的问题数和偏好对数
        - 质量统计：不同质量等级的答案分布
        - 成功率统计：答案生成的成功率
        - 完整性检查：是否有缺失或错误的数据
        
        【在整体程序中的作用】
        - 提供数据分割过程的完整总结
        - 为后续工作提供数据质量参考
        - 帮助发现和解决数据问题
        
        Args:
            train_data: 训练集数据
            eval_data: 验证集数据  
            test_data: 测试集数据
        """
        # 计算基本统计
        total_questions = len(train_data) + len(eval_data) + len(test_data)
        
        # 统计答案质量分布
        def analyze_quality_distribution(data, dataset_name):
            """分析单个数据集的质量分布"""
            quality_stats = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            total_answers = 0
            successful_answers = 0
            
            for item in data:
                answers = item.get("answers", [])
                for answer in answers:
                    total_answers += 1
                    if not answer.get("error", False):
                        successful_answers += 1
                        level = answer.get("quality_level", 0)
                        if level in quality_stats:
                            quality_stats[level] += 1
            
            return {
                "dataset_name": dataset_name,
                "questions": len(data),
                "total_answers": total_answers,
                "successful_answers": successful_answers,
                "success_rate": successful_answers / total_answers * 100 if total_answers > 0 else 0,
                "quality_distribution": quality_stats
            }
        
        # 分析各个数据集
        train_stats = analyze_quality_distribution(train_data, "训练集")
        eval_stats = analyze_quality_distribution(eval_data, "验证集")
        test_stats = analyze_quality_distribution(test_data, "测试集")
        
        # 计算偏好对数量
        def count_preference_pairs(data):
            """计算偏好对数量"""
            total_pairs = 0
            for item in data:
                answers = item.get("answers", [])
                successful_answers = [a for a in answers if not a.get("error", False)]
                # C(n,2) = n*(n-1)/2
                n = len(successful_answers)
                pairs = n * (n - 1) // 2 if n >= 2 else 0
                total_pairs += pairs
            return total_pairs
        
        train_pairs = count_preference_pairs(train_data)
        eval_pairs = count_preference_pairs(eval_data)
        test_pairs = count_preference_pairs(test_data)
        
        # 创建完整报告
        report = {
            "summary": {
                "total_questions": total_questions,
                "train_questions": len(train_data),
                "eval_questions": len(eval_data),
                "test_questions": len(test_data),
                "train_preference_pairs": train_pairs,
                "eval_preference_pairs": eval_pairs,
                "test_preference_pairs": test_pairs,
                "total_preference_pairs": train_pairs + eval_pairs + test_pairs
            },
            "dataset_statistics": [train_stats, eval_stats, test_stats],
            "split_ratios": {
                "train": len(train_data) / total_questions * 100,
                "eval": len(eval_data) / total_questions * 100,
                "test": len(test_data) / total_questions * 100
            },
            "random_seed": self.random_seed,
            "split_timestamp": json.dumps({"timestamp": __import__('time').time()})
        }
        
        # 保存报告
        report_file = self.output_dir / "split_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # 打印摘要信息
            print("\n📊 数据拆分统计报告:")
            print(f"  📋 总问题数: {total_questions}")
            print(f"  🎯 训练集: {len(train_data)} 问题, {train_pairs} 偏好对")
            print(f"  🔍 验证集: {len(eval_data)} 问题, {eval_pairs} 偏好对") 
            print(f"  🧪 测试集: {len(test_data)} 问题, {test_pairs} 偏好对")
            print(f"  📊 总偏好对: {train_pairs + eval_pairs + test_pairs}")
            print(f"  📄 详细报告: {report_file}")
            
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")

    def split_and_save(self, train_ratio: float = 0.8, eval_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        执行完整的数据拆分和保存流程
        
        【作用】
        这是数据拆分器的主要入口方法，执行完整的处理流程：
        1. 加载所有原始数据
        2. 按比例分割数据集
        3. 保存不同格式的数据文件
        4. 生成统计报告
        5. 清理和验证结果
        
        【流程概述】
        加载数据 → 随机分割 → 格式转换 → 保存文件 → 生成报告
        
        【异常处理】
        整个过程包含完善的错误处理：
        - 文件不存在的处理
        - 数据格式错误的处理
        - 磁盘空间不足的处理
        - 权限问题的处理
        
        【在整体程序中的作用】
        - 完成整个数据生成流程的最后一步
        - 为奖励模型训练提供标准化数据
        - 确保数据处理的科学性和规范性
        
        Args:
            train_ratio: 训练集比例
            eval_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        try:
            print("🚀 开始数据拆分流程...")
            
            # 第一步：加载所有数据
            print("📥 加载原始数据...")
            all_data = self.load_all_answers_data()
            
            if not all_data:
                raise ValueError("没有找到任何数据进行拆分")
            
            # 第二步：执行数据拆分
            print("✂️  执行数据拆分...")
            train_data, eval_data, test_data = self.split_data(
                all_data, train_ratio, eval_ratio, test_ratio
            )
            
            # 第三步：保存各个数据集
            print("💾 保存数据集文件...")
            self.save_dataset(train_data, "train")
            self.save_dataset(eval_data, "eval")  
            self.save_dataset(test_data, "test")
            
            # 第四步：生成统计报告
            print("📊 生成统计报告...")
            self.generate_summary_report(train_data, eval_data, test_data)
            
            print("✅ 数据拆分流程完成！")
            print(f"📁 输出目录: {self.output_dir.absolute()}")
            
        except Exception as e:
            print(f"❌ 数据拆分失败: {e}")
            raise

def main():
    """
    数据拆分器的命令行入口
    
    【作用】
    提供独立的命令行工具，允许用户单独运行数据拆分：
    1. 解析命令行参数
    2. 创建数据拆分器实例
    3. 执行拆分流程
    4. 处理命令行错误
    
    【使用场景】
    - 独立运行数据拆分（不执行完整流程）
    - 重新拆分已有数据
    - 使用不同的拆分比例
    - 调试和测试拆分功能
    
    【在整体程序中的作用】
    - 提供灵活的数据处理方式
    - 支持模块化的工作流程
    - 便于测试和调试
    """
    parser = argparse.ArgumentParser(description="数据拆分工具")
    parser.add_argument("--input-dir", required=True, help="输入目录（包含答案数据）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--random-seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    splitter = DataSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )
    
    splitter.split_and_save(
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio
    )

if __name__ == "__main__":
    main()
