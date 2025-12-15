#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励模型数据生成主程序
整合所有模块，执行完整的数据生成流程

【作用】
这是整个程序的入口点，就像乐队的指挥，负责：
1. 解析命令行参数
2. 初始化所有子模块
3. 协调各个模块按顺序工作
4. 处理全局错误
5. 记录整体运行情况

【流程概述】
配置解析 -> 数据处理 -> 问题生成 -> 答案生成 -> 训练集分割
    |            |           |           |           |
Config.py   DataProcessor  QuestionGen  AnswerGen  DataSplitter

【使用方法】
python main.py --stage all --sft_data_path ... --base_articles_path ...

详见 python main.py --help
"""

import os
import sys
import time
import logging
from pathlib import Path

# 添加当前目录到Python路径，确保能导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config import Config, parse_and_validate_args
from utils import setup_logging, ensure_directory_exists
from data_processor import DataProcessor
from question_generator import QuestionGenerator
from answer_generator import AnswerGenerator
from data_splitter import DataSplitter

def main():
    """主函数：程序的执行入口"""
    
    # 1. 解析参数
    args = parse_and_validate_args()
    
    # 2. 设置日志
    setup_logging(args.log_level)
    logging.info(f"开始执行任务，阶段: {args.stage}")
    
    # 3. 初始化各个模块
    # 就像组装机器一样，先把各个部件准备好
    try:
        # 数据处理器：负责搬运和清洗数据
        data_processor = DataProcessor(seed=args.random_seed)
        
        # 问题生成器：负责从文章造问题
        # 注意：这里我们传入concurrency_num作为文章生成的并发控制
        # 在问题生成阶段，concurrency_num控制同时处理多少篇文章
        question_generator = QuestionGenerator(
            Config, 
            output_dir=args.output_dir,
            concurrency_level=getattr(args, 'concurrency_num', Config.CONCURRENCY_NUM)
        )
        
        # 答案生成器：负责回答问题
        # 注意：在答案生成阶段，concurrency_num控制总并发量
        # 内部会自动分配给问题并发和答案并发
        answer_generator = AnswerGenerator(
            Config, 
            output_dir=args.output_dir,
            concurrency_level=getattr(args, 'concurrency_num', Config.CONCURRENCY_NUM)
        )
        
        # 数据拆分器：负责最后的分包
        data_splitter = DataSplitter(
            input_dir=os.path.join(args.output_dir, "answers"),
            output_dir=args.output_dir,
            random_seed=args.random_seed
        )
        
    except Exception as e:
        logging.error(f"模块初始化失败: {e}")
        return

    # 4. 根据选择的阶段执行任务
    
    # === 阶段1：数据采样和准备 ===
    # 如果用户选择 'sample' 或 'all' 阶段
    if args.stage in ['sample', 'all']:
        logging.info("=== 进入数据采样阶段 ===")
        try:
            # 1. 加载SFT数据
            sft_data = data_processor.load_sft_data(args.sft_data_path.split(','))
            
            # 2. 提取SFT问题
            # 根据sft_ratio计算需要多少个SFT问题
            sft_count = int(args.total_questions * args.sft_ratio)
            sft_questions = data_processor.extract_questions_from_sft(sft_data, sft_count)
            
            # 3. 保存采样后的SFT问题
            # 【重要】保存中间结果，方便调试和后续步骤使用
            from utils import save_jsonl_file
            save_jsonl_file(
                [{"question": q, "source": "sft"} for q in sft_questions], 
                Config.get_output_paths()["sampled_sft_questions"]
            )
            
            # 4. 加载和采样文章
            # 计算需要生成多少个新问题
            gen_count = args.total_questions - sft_count
            articles = data_processor.load_base_articles(args.base_articles_path)
            sampled_articles = data_processor.sample_articles_for_generation(articles, gen_count)
            
            # 5. 保存采样后的文章
            save_jsonl_file(sampled_articles, Config.get_output_paths()["sampled_articles"])
            
        except Exception as e:
            logging.error(f"数据采样阶段失败: {e}")
            if args.stage == 'sample': return
            sys.exit(1)  # 如果是all阶段，采样失败就必须停止

    # === 阶段2：问题生成 ===
    # 如果用户选择 'questions' 或 'all' 阶段
    if args.stage in ['questions', 'all']:
        logging.info("=== 进入问题生成阶段 ===")
        try:
            # 1. 读取采样后的文章
            # 如果是all阶段，可以直接用内存中的数据，但为了解耦，还是读取文件比较稳妥
            from utils import load_jsonl_file
            sampled_articles_path = Config.get_output_paths()["sampled_articles"]
            
            if not os.path.exists(sampled_articles_path):
                raise FileNotFoundError(f"找不到采样文章文件: {sampled_articles_path}，请先执行sample阶段")
            
            sampled_articles = load_jsonl_file(str(sampled_articles_path))
            
            # 2. 生成新问题
            generated_questions = question_generator.generate_questions_from_articles(sampled_articles)
            
            # 3. 合并所有问题（SFT + 生成）
            sft_questions_path = Config.get_output_paths()["sampled_sft_questions"]
            if os.path.exists(sft_questions_path):
                sft_data = load_jsonl_file(str(sft_questions_path))
                sft_questions = [item["question"] for item in sft_data]
            else:
                logging.warning("未找到SFT问题文件，将只使用生成的问题")
                sft_questions = []
            
            all_questions = data_processor.combine_questions(sft_questions, generated_questions)
            
            # 4. 保存混合后的问题列表
            # 这是下一阶段（答案生成）的输入
            mixed_questions_path = Config.get_output_paths()["mixed_questions"]
            # 转换为对象列表以便保存为jsonl
            questions_to_save = [{"id": i+1, "question": q} for i, q in enumerate(all_questions)]
            save_jsonl_file(questions_to_save, mixed_questions_path)
            
        except Exception as e:
            logging.error(f"问题生成阶段失败: {e}")
            if args.stage == 'questions': return
            sys.exit(1)

    # === 阶段3：答案生成 ===
    # 如果用户选择 'answers' 或 'all' 阶段
    if args.stage in ['answers', 'all']:
        logging.info("=== 进入答案生成阶段 ===")
        try:
            # 1. 读取混合后的问题列表
            mixed_questions_path = Config.get_output_paths()["mixed_questions"]
            
            if not os.path.exists(mixed_questions_path):
                raise FileNotFoundError(f"找不到问题文件: {mixed_questions_path}，请先执行questions阶段")
            
            from utils import load_jsonl_file
            questions = load_jsonl_file(str(mixed_questions_path))
            
            # 2. 生成答案
            # 这里的max_questions参数用于测试时限制生成数量
            max_q = getattr(args, 'max_questions', None)
            answer_generator.generate_preference_dataset(questions, max_questions=max_q)
            
            # 3. 执行数据拆分
            # 答案生成完成后，自动执行拆分，生成最终训练数据
            logging.info("=== 自动执行数据拆分 ===")
            data_splitter.split_and_save()
            
        except Exception as e:
            logging.error(f"答案生成阶段失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
