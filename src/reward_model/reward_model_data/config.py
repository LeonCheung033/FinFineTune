#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件：定义所有的配置参数和常量
这个文件包含了整个项目需要用到的所有配置信息，方便统一管理
"""

import os
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv  # 需要安装: pip install python-dotenv

# 加载环境变量文件
# 这样可以从.env文件中读取配置，避免在代码中硬编码敏感信息
load_dotenv()

class Config:
    """
    配置类：存储所有配置参数
    
    这个类就像一个设置面板，包含了程序运行需要的所有参数
    比如文件路径、API设置、生成参数等
    """
    
    # ==================== API相关配置 ====================
    # 这部分配置用于连接AI模型的API服务
    
    # DeepSeek API的密钥，优先从环境变量读取，确保安全性
    # 环境变量 > .env文件 > 默认值的优先级
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-de8e3482dwvyuybovuc22167d461b0d3cb")
    
    # DeepSeek API的基础URL地址
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    # 使用的AI模型名称
    # deepseek-chat是DeepSeek的主要对话模型
    MODEL_NAME: str = "deepseek-chat"
    
    # API调用的温度参数，控制生成内容的随机性
    # 0.7表示有一定随机性但不会太乱，数值越高越随机
    TEMPERATURE: float = 0.7
    
    # 每次API调用生成的最大字符数
    # 5000个token大概是3500-4000个中文字符
    MAX_TOKENS: int = 5000
    
    # API调用之间的等待时间（秒）
    # DeepSeek的频率限制相对宽松，1秒间隔比较安全
    API_CALL_INTERVAL: float = 1.0
    
    # 最大并发线程数
    # DeepSeek支持较高的并发，但建议不超过20个线程
    MAX_WORKERS: int = 20
    
    # ==================== 并发控制配置 ====================
    # 通用并发数：同时处理多少个任务
    CONCURRENCY_NUM: int = int(os.getenv("CONCURRENCY_NUM", "3"))
    
    # 答案级别的并发数：每个问题同时生成多少个答案（固定为5）
    ANSWER_CONCURRENCY: int = 5
    
    # ==================== 数据文件配置 ====================
    # 这部分配置定义了各种数据文件的路径和格式
    
    # 支持的数据文件格式列表
    # 程序会自动识别这些格式的文件
    SUPPORTED_FILE_FORMATS: List[str] = [".json", ".jsonl", ".txt", ".csv"]
    
    # SFT训练数据的文件路径列表
    # 可以是多个文件，程序会自动合并处理
    SFT_DATA_FILES: List[str] = []
    
    # 源文章数据的文件路径列表
    # 这些是用来生成新问题的基础文档
    SOURCE_ARTICLES_FILES: List[str] = []
    
    # 输出目录路径
    # 所有生成的文件都会保存在这个目录下
    OUTPUT_DIR: str = os.getenv("DEFAULT_OUTPUT_DIR", "./reward_model_data")
    
    # ==================== 数据生成配置 ====================
    # 这部分配置控制数据生成的数量和比例
    
    # 总共要生成的问题数量
    # 建议5000个问题，可以根据需要调整
    TOTAL_QUESTIONS: int = int(os.getenv("DEFAULT_TOTAL_QUESTIONS", "5000"))
    
    # 从SFT数据中提取问题的比例
    # 0.3表示30%的问题来自已有的SFT数据
    SFT_QUESTION_RATIO: float = float(os.getenv("DEFAULT_SFT_RATIO", "0.3"))
    
    # 每个问题生成的答案数量
    # 5个不同质量等级的答案，用于构建偏好对
    ANSWERS_PER_QUESTION: int = 5
    
    # 随机种子，用于确保结果可重现
    # 相同的种子会产生相同的随机结果
    RANDOM_SEED: int = int(os.getenv("DEFAULT_RANDOM_SEED", "42"))
    
    # 批处理大小，每处理这么多条数据就保存一次
    # 防止程序崩溃时丢失太多数据
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # ==================== 日志配置 ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ==================== 质量等级配置 ====================
    # 定义5个不同质量等级的答案要求
    
    QUALITY_LEVELS: Dict[int, Dict[str, str]] = {
        1: {
            "name": "低质量答案",
            "description": "逻辑混乱，信息不准确，分析浅显",
            "instruction": """请生成一个质量较低的回答，具体要求：
1. 逻辑不够清晰，分析比较浅显
2. 可能包含一些不够准确的信息（但不能完全错误）
3. 思考过程简单，缺乏深度分析
4. 但仍然要使用<think>标签包裹思考过程
5. 答案要有一定的相关性，不能完全偏题"""
        },
        2: {
            "name": "中低质量答案", 
            "description": "基本正确但分析不够深入",
            "instruction": """请生成一个中等偏低质量的回答，具体要求：
1. 基本信息正确但分析不够深入
2. 思考过程相对简单，缺少一些关键步骤
3. 缺乏一些重要的金融专业见解
4. 使用<think>标签包裹思考过程
5. 答案基本能解决问题但不够全面"""
        },
        3: {
            "name": "中等质量答案",
            "description": "正确但不够全面", 
            "instruction": """请生成一个中等质量的回答，具体要求：
1. 信息准确，逻辑基本清晰
2. 有一定的专业分析但覆盖面不够全面
3. 思考过程较为完整，有基本的分析步骤
4. 使用<think>标签包裹思考过程
5. 能够解决问题但可能遗漏一些重要方面"""
        },
        4: {
            "name": "中高质量答案",
            "description": "专业且全面",
            "instruction": """请生成一个中等偏高质量的回答，具体要求：
1. 信息准确，分析专业且较为全面
2. 逻辑清晰，有较深度的金融专业见解
3. 思考过程详细且有条理，考虑多个角度
4. 使用<think>标签包裹思考过程
5. 能够很好地解决问题，分析比较深入"""
        },
        5: {
            "name": "高质量答案",
            "description": "专业、全面、深入",
            "instruction": """请生成一个高质量的回答，具体要求：
1. 信息非常准确，分析专业且全面深入
2. 逻辑非常清晰，有独到的金融专业见解
3. 思考过程非常详细，考虑多个维度和可能性
4. 使用<think>标签包裹思考过程，展现完整的分析链条
5. 完美解决问题，提供深刻洞察和实用建议"""
        }
    }
    
    # ==================== 文件命名配置 ====================
    # 定义各种输出文件的名称
    
    # 采样后的SFT问题文件名
    SAMPLED_SFT_QUESTIONS_FILE: str = "sampled_sft_questions.jsonl"
    
    # 采样后的文章文件名
    SAMPLED_ARTICLES_FILE: str = "sampled_articles.jsonl"
    
    # 生成的新问题文件名
    GENERATED_QUESTIONS_FILE: str = "generated_questions.jsonl"
    
    # 混合问题文件名（第一阶段输出）
    MIXED_QUESTIONS_FILE: str = "all_questions_mixed.jsonl"
    
    # 生成答案文件名（第二阶段输出）
    GENERATED_ANSWERS_FILE: str = "generated_answers.jsonl"
    
    # 偏好对文件名（用于训练奖励模型）
    PREFERENCE_PAIRS_FILE: str = "preference_pairs.jsonl"
    
    # 统计报告文件名
    STATISTICS_REPORT_FILE: str = "generation_report.json"
    
    # 错误日志文件名
    ERROR_LOG_FILE: str = "error_log.txt"
    
    # ==================== 数据验证配置 ====================
    # 用于验证生成数据质量的配置
    
    # SFT数据必须包含的字段
    # 这些字段是SFT数据文件必须要有的，缺少会报错
    REQUIRED_SFT_FIELDS: List[str] = ["input"]  # SFT数据主要需要input字段作为问题
    
    # 源文章数据必须包含的字段
    # 这些字段是源文章文件必须要有的
    REQUIRED_ARTICLE_FIELDS: List[str] = ["Article"]  # 文章数据需要Article字段
    
    # 问题的最小长度（字符数）
    # 太短的问题质量通常不高
    MIN_QUESTION_LENGTH: int = 20
    
    # 答案的最小长度（字符数）
    # 太短的答案通常不够详细
    MIN_ANSWER_LENGTH: int = 100
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        验证配置是否正确
        
        这个方法会检查所有重要的配置项是否设置正确
        就像开车前检查油量、轮胎一样，确保程序能正常运行
        
        Returns:
            bool: 如果配置正确返回True，否则返回False
        """
        errors = []  # 用来收集所有的错误信息
        
        # 检查API密钥是否设置
        # API密钥就像身份证，没有它就无法调用AI服务
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY 未设置，请在.env文件中设置或通过环境变量提供")
        
        # 检查比例参数是否合理
        # 比例必须在0到1之间，超出范围就不合理
        if not 0 <= cls.SFT_QUESTION_RATIO <= 1:
            errors.append("SFT_QUESTION_RATIO 必须在0到1之间")
        
        # 检查数量参数是否为正数
        # 负数或零没有意义
        if cls.TOTAL_QUESTIONS <= 0:
            errors.append("TOTAL_QUESTIONS 必须大于0")
        
        if cls.ANSWERS_PER_QUESTION <= 0:
            errors.append("ANSWERS_PER_QUESTION 必须大于0")
        
        # 如果有错误，打印出来并返回False
        if errors:
            print("配置验证失败，发现以下错误：")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
            return False
        
        print("配置验证通过！")
        return True
    
    @classmethod
    def get_output_paths(cls) -> Dict[str, Path]:
        """
        获取所有输出文件的完整路径
        
        这个方法会返回一个字典，包含所有输出文件的路径
        方便其他模块使用，避免路径拼接错误
        
        Returns:
            Dict[str, Path]: 包含所有输出文件路径的字典
        """
        output_dir = Path(cls.OUTPUT_DIR)
        
        return {
            "output_dir": output_dir,
            "sampled_sft_questions": output_dir / cls.SAMPLED_SFT_QUESTIONS_FILE,
            "sampled_articles": output_dir / cls.SAMPLED_ARTICLES_FILE,
            "generated_questions": output_dir / cls.GENERATED_QUESTIONS_FILE,
            "mixed_questions": output_dir / cls.MIXED_QUESTIONS_FILE,
            "generated_answers": output_dir / cls.GENERATED_ANSWERS_FILE,
            "preference_pairs": output_dir / cls.PREFERENCE_PAIRS_FILE,
            "statistics_report": output_dir / cls.STATISTICS_REPORT_FILE,
            "error_log": output_dir / cls.ERROR_LOG_FILE
        }

    # ==================== 属性访问方法 ====================
    @property
    def api_key(self):
        """获取API密钥"""
        return self.DEEPSEEK_API_KEY
    
    @property
    def base_url(self):
        """获取API基础URL"""
        return self.DEEPSEEK_BASE_URL
    
    @property
    def model_name(self):
        """获取模型名称"""
        return self.MODEL_NAME
    
    @property
    def temperature(self):
        """获取温度参数"""
        return self.TEMPERATURE
    
    @property
    def max_tokens(self):
        """获取最大token数"""
        return self.MAX_TOKENS
    
    @property
    def request_interval(self):
        """获取请求间隔"""
        return self.API_CALL_INTERVAL
    
    @property
    def max_workers(self):
        """获取最大工作线程数"""
        return self.MAX_WORKERS
    
    @property
    def output_dir(self):
        """获取输出目录"""
        return self.OUTPUT_DIR
    
    @property
    def batch_size(self):
        """获取批处理大小"""
        return self.BATCH_SIZE

def create_argument_parser():
    """
    创建命令行参数解析器
    
    这个函数定义了所有可用的命令行参数
    让用户可以通过命令行灵活控制程序行为
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
            description='金融领域奖励模型数据生成工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:

1. 完整流程（推荐）:
   python main.py --stage all --sft_data_path ./sft_data.jsonl --base_articles_path ./articles.jsonl

2. 分阶段执行:
   # 第一步：数据采样
   python main.py --stage sample --sft_data_path ./sft_data.jsonl --base_articles_path ./articles.jsonl
   
   # 第二步：问题生成
   python main.py --stage questions
   
   # 第三步：答案生成
   python main.py --stage answers --max_questions 100 --concurrency_num 3

3. 自定义参数:
   python main.py --stage all --total_questions 3000 --sft_ratio 0.4 --output_dir ./my_data

注意:
- API密钥会自动从.env文件或环境变量中读取
- 默认使用DeepSeek API
- 所有生成的文件都会保存在输出目录中
        """
    )
    
    # === 执行阶段控制 ===
    # 这是最重要的参数，决定程序执行哪个阶段
    parser.add_argument(
        '--stage', 
            type=str, 
        choices=['sample', 'questions', 'answers', 'all'], 
        default='all',
        help="""执行阶段选择：
        sample - 仅执行数据采样（从原始数据中选择用于训练的子集）
        questions - 仅执行问题生成（从文章生成新问题）
        answers - 仅执行答案生成（为问题生成不同质量的答案）
        all - 执行完整流程（默认选项）"""
        )
        
        # === 数据路径配置 ===
    # 这些参数指定输入数据的位置
    data_group = parser.add_argument_group('数据路径配置', '指定输入数据文件的位置')
    data_group.add_argument(
            '--sft_data_path', 
            type=str, 
        help="""SFT训练数据文件路径（JSONL格式）
        文件应包含以下字段：
        - instruction: 任务指令（可选）
        - input: 问题内容（必需）
        - output: 期望答案（可选）
        
        示例：{"instruction": "分析以下情况", "input": "某公司股价下跌的原因", "output": "..."}"""
        )
    data_group.add_argument(
            '--base_articles_path', 
            type=str, 
        help="""基础文章数据文件路径（JSONL格式）
        文件应包含以下字段：
        - Article: 文章内容（必需）
        - Summary: 文章摘要（可选）
        
        示例：{"Article": "金融市场分析...", "Summary": "本文分析了..."}"""
        )
    data_group.add_argument(
            '--output_dir', 
            type=str, 
        default=Config.OUTPUT_DIR, 
        help=f'输出目录路径，所有生成的文件都会保存在这里（默认：{Config.OUTPUT_DIR}）'
        )
        
        # === 数据采样配置 ===
    # 这些参数控制数据采样的行为
    sample_group = parser.add_argument_group('数据采样配置', '控制如何从原始数据中选择训练样本')
    sample_group.add_argument(
        '--total_questions', 
        type=int, 
    default=Config.TOTAL_QUESTIONS, 
    help=f'总问题数量，程序会生成这么多个问题用于训练（默认：{Config.TOTAL_QUESTIONS}）'
    )
    sample_group.add_argument(
        '--sft_ratio', 
        type=float, 
    default=Config.SFT_QUESTION_RATIO, 
    help=f"""SFT数据占比，取值范围0-1
    例如0.3表示30%%的问题来自SFT数据，70%%来自新生成的问题
    （默认：{Config.SFT_QUESTION_RATIO}）"""
    )
    sample_group.add_argument(
        '--random_seed', 
        type=int, 
    default=Config.RANDOM_SEED, 
        help=f'随机种子，确保结果可重现。相同种子会产生相同的随机结果（默认：{Config.RANDOM_SEED}）'
    )
    
    # === 答案生成专用配置 ===
    answer_group = parser.add_argument_group('答案生成配置', '控制答案生成阶段的行为')
    answer_group.add_argument(
        '--max_questions', 
        type=int, 
        help='最大处理问题数（用于测试或限制处理量）'
    )
    answer_group.add_argument(
        '--concurrency_num', 
        type=int, 
        default=Config.CONCURRENCY_NUM,
        help=f'并发数量，同时处理多少个问题（默认：{Config.CONCURRENCY_NUM}）'
        )
        
        # === 生成配置 ===
    # 这些参数控制AI生成的行为
    gen_group = parser.add_argument_group('生成配置', '控制AI生成问题和答案的行为')
    gen_group.add_argument(
        '--max_workers', 
        type=int, 
    default=Config.MAX_WORKERS, 
    help=f"""最大并发线程数，控制同时进行的API调用数量
    数值越大生成越快，但可能触发API频率限制
    建议根据API服务商的限制调整（默认：{Config.MAX_WORKERS}）"""
    )
    gen_group.add_argument(
        '--request_interval', 
        type=float, 
    default=Config.API_CALL_INTERVAL, 
    help=f"""API请求间隔（秒），每次API调用之间的等待时间
    用于避免触发频率限制，数值越大越安全但生成越慢
    （默认：{Config.API_CALL_INTERVAL}秒）"""
    )
    gen_group.add_argument(
        '--batch_size', 
        type=int, 
        default=Config.BATCH_SIZE, 
        help=f"""批处理大小，每处理这么多条数据就保存一次
        用于防止程序崩溃时丢失数据，建议10-50之间
        （默认：{Config.BATCH_SIZE}）"""
    )
    
    # === 系统配置 ===
    system_group = parser.add_argument_group('系统配置', '控制程序运行行为的系统参数')
    system_group.add_argument(
        '--log_level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default=Config.LOG_LEVEL,
        help=f'日志级别（默认：{Config.LOG_LEVEL}）'
    )
    
    # === 高级选项 ===
    # 这些是一些高级用户可能需要的选项
    advanced_group = parser.add_argument_group('高级选项', '高级用户可能需要的特殊选项')
    advanced_group.add_argument(
        '--start_index', 
        type=int, 
        default=0, 
        help="""开始处理的索引位置，用于断点续传
        如果程序中途中断，可以从指定位置继续处理
        例如--start_index 1000表示从第1001个问题开始处理"""
    )
    advanced_group.add_argument(
        '--api_key_override', 
        type=str, 
        help="""临时覆盖API密钥，仅在特殊情况下使用
        正常情况下请在.env文件中设置DEEPSEEK_API_KEY
        注意：命令行中的密钥可能被其他用户看到，存在安全风险"""
    )
    advanced_group.add_argument(
        '--base_url_override', 
            type=str, 
        help=f"""临时覆盖API基础URL，用于使用自定义API端点
        （默认：{Config.DEEPSEEK_BASE_URL}）"""
    )
    advanced_group.add_argument(
        '--verbose', 
        action='store_true', 
        help='启用详细输出模式，显示更多调试信息'
    )
    
    return parser

def parse_and_validate_args():
    """
    解析并验证命令行参数
    
    这个函数会：
    1. 解析用户输入的命令行参数
    2. 验证参数的有效性
    3. 更新配置类的设置
    4. 返回验证后的参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    
    Raises:
        SystemExit: 如果参数验证失败
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 验证必需的参数
    # 某些阶段需要特定的输入文件
    if args.stage in ['sample', 'all']:
        if not args.sft_data_path:
            parser.error("执行sample或all阶段时，必须提供--sft_data_path参数")
        if not args.base_articles_path:
            parser.error("执行sample或all阶段时，必须提供--base_articles_path参数")
        
        # 检查文件是否存在
        sft_paths = [path.strip() for path in args.sft_data_path.split(',')]
        for sft_path in sft_paths:
            if not os.path.exists(sft_path):
                parser.error(f"SFT数据文件不存在: {sft_path}")
        
        if not os.path.exists(args.base_articles_path):
            parser.error(f"基础文章文件不存在: {args.base_articles_path}")
        
        # 验证参数范围
    if not 0 <= args.sft_ratio <= 1:
        parser.error("sft_ratio必须在0和1之间")
    if args.total_questions <= 0:
        parser.error("total_questions必须大于0")
    if args.sft_ratio <= 0:
        parser.error("sft_ratio必须大于0")
    if args.max_workers <= 0:
        parser.error("max_workers必须大于0")
    if args.request_interval < 0:
        parser.error("request_interval不能为负数")
    
    # 更新配置类
    # 将命令行参数应用到配置类中
    if args.api_key_override:
        Config.DEEPSEEK_API_KEY = args.api_key_override
        print("警告：使用命令行提供的API密钥，请注意安全风险")
    
    if args.base_url_override:
        Config.DEEPSEEK_BASE_URL = args.base_url_override
    
    Config.OUTPUT_DIR = args.output_dir
    Config.TOTAL_QUESTIONS = args.total_questions
    Config.SFT_QUESTION_RATIO = args.sft_ratio
    Config.RANDOM_SEED = args.random_seed
    Config.MAX_WORKERS = args.max_workers
    Config.API_CALL_INTERVAL = args.request_interval
    Config.BATCH_SIZE = args.batch_size
    Config.CONCURRENCY_NUM = getattr(args, 'concurrency_num', Config.CONCURRENCY_NUM)
    Config.LOG_LEVEL = args.log_level
        
        # 设置随机种子，确保结果可重现
    random.seed(args.random_seed)
        
        # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
        
    # 验证最终配置
    if not Config.validate_config():
        parser.error("配置验证失败，请检查上述错误信息")
    
    # 显示配置摘要
    print(f"\n=== 配置摘要 ===")
    print(f"执行阶段: {args.stage}")
    print(f"输出目录: {args.output_dir}")
    if hasattr(args, 'max_questions') and args.max_questions:
        print(f"限制问题数: {args.max_questions}")
    if hasattr(args, 'concurrency_num'):
        print(f"并发数量: {args.concurrency_num}")
    print(f"总问题数: {args.total_questions}")
    print(f"SFT数据比例: {args.sft_ratio:.1%}")
    print(f"随机种子: {args.random_seed}")
    print(f"日志级别: {args.log_level}")
    print(f"最大并发数: {args.max_workers}")
    print(f"API间隔: {args.request_interval}秒")
    if hasattr(args, 'sft_data_path') and args.sft_data_path:
        print(f"SFT数据文件: {args.sft_data_path}")
    if hasattr(args, 'base_articles_path') and args.base_articles_path:
        print(f"文章数据文件: {args.base_articles_path}")
    print(f"API服务: DeepSeek ({Config.DEEPSEEK_BASE_URL})")
    print("=" * 50)
        
    return args
