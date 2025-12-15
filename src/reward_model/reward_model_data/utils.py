#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块 - 简化版本
包含数据处理、文件操作等基础功能

【作用】
这个模块是整个项目的"工具箱"，提供各种通用的基础功能：
1. 文件操作：读取、保存、目录管理
2. 数据处理：采样、混合、格式转换
3. 日志管理：统一的日志配置
4. 路径验证：确保文件存在性

【为什么需要工具模块】
- 避免代码重复：多个模块都需要相同的基础功能
- 统一标准：确保所有模块使用相同的文件操作方式
- 便于维护：修改基础功能时只需改一个地方
- 提高可靠性：经过测试的通用函数更稳定

【在整体程序中的作用】
- 为所有其他模块提供基础服务
- 确保文件操作的一致性和安全性
- 提供数据处理的标准方法
- 统一错误处理和日志记录
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> None:
    """
    设置日志配置
    
    【作用】
    为整个程序配置统一的日志系统：
    1. 设置日志级别（控制显示哪些信息）
    2. 设置日志格式（时间、级别、消息）
    3. 确保所有模块使用相同的日志标准
    
    【日志级别说明】
    - DEBUG: 最详细，包含调试信息
    - INFO: 一般信息，程序正常运行状态
    - WARNING: 警告信息，可能有问题但不影响运行
    - ERROR: 错误信息，程序出现问题
    
    【为什么要统一日志】
    - 便于问题诊断：所有信息格式一致
    - 控制信息量：可以调整显示的详细程度
    - 便于监控：程序运行状态一目了然
    
    【在整体程序中的作用】
    - 程序启动时的第一步配置
    - 为所有模块提供统一的日志接口
    - 帮助用户了解程序运行状态
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),  # 将字符串转换为日志级别常量
        format='%(asctime)s - %(levelname)s - %(message)s',  # 格式：时间 - 级别 - 消息
        datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式：年-月-日 时:分:秒
    )

def ensure_directory_exists(directory_path) -> None:
    """
    确保目录存在，如果不存在则创建
    
    【作用】
    这是一个安全的目录创建函数：
    1. 检查目录是否已经存在
    2. 如果不存在，创建目录（包括所有父目录）
    3. 如果已存在，什么都不做
    4. 处理权限和其他创建错误
    
    【为什么需要这个函数】
    程序运行时经常需要保存文件到指定目录，
    但目录可能不存在，直接保存会出错。
    这个函数确保目录一定存在。
    
    【Path.mkdir()参数说明】
    - parents=True: 如果父目录不存在，也会创建
    - exist_ok=True: 如果目录已存在，不会报错
    
    【在整体程序中的作用】
    - 在保存任何文件之前确保目录存在
    - 避免因目录不存在导致的文件保存失败
    - 提供统一的目录管理方式
    
    Args:
        directory_path: 目录路径（可以是字符串或Path对象）
    """
    path = Path(directory_path)  # 转换为Path对象，统一处理
    path.mkdir(parents=True, exist_ok=True)  # 创建目录，包括父目录

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSONL格式文件
    
    【作用】
    JSONL（JSON Lines）是一种常用的数据格式，每行一个JSON对象。
    这个函数负责：
    1. 打开并读取JSONL文件
    2. 逐行解析JSON数据
    3. 处理格式错误和文件错误
    4. 返回所有数据的列表
    
    【什么是JSONL格式】
    每行都是一个完整的JSON对象，例如：
    {"id": 1, "question": "问题1"}
    {"id": 2, "question": "问题2"}
    {"id": 3, "question": "问题3"}
    
    【为什么使用JSONL】
    - 流式处理：可以逐行读取，不需要一次性加载整个文件
    - 容错性好：某行出错不影响其他行
    - 易于追加：可以直接在文件末尾添加新数据
    - 工具支持：很多数据处理工具支持JSONL格式
    
    【错误处理策略】
    - 文件不存在：抛出异常，程序停止
    - JSON格式错误：记录警告，跳过错误行，继续处理
    - 空行：自动跳过
    
    【在整体程序中的作用】
    - 为所有模块提供统一的文件读取接口
    - 确保数据加载的一致性和可靠性
    - 处理各种异常情况，提高程序稳定性
    
    Args:
        file_path: 文件路径
        
    Returns:
        包含所有数据行的列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # enumerate从1开始，方便报告行号
                line = line.strip()  # 去除首尾空白字符
                if line:  # 跳过空行
                    try:
                        # 解析JSON数据
                        json_data = json.loads(line)
                        data.append(json_data)
                    except json.JSONDecodeError as e:
                        # JSON格式错误时记录警告但继续处理
                        logging.warning(f"第{line_num}行JSON解析失败: {e}")
                        continue
        logging.info(f"成功读取 {len(data)} 条数据从 {file_path}")
    except FileNotFoundError:
        logging.error(f"文件不存在: {file_path}")
        raise  # 重新抛出异常，让调用者处理
    except Exception as e:
        logging.error(f"读取文件失败 {file_path}: {e}")
        raise
    
    return data

def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    保存数据到JSONL格式文件
    
    【作用】
    将Python数据结构保存为JSONL格式文件：
    1. 确保输出目录存在
    2. 将每个数据项转换为JSON字符串
    3. 每行写入一个JSON对象
    4. 处理编码和写入错误
    
    【保存格式】
    每个列表项会转换为一行JSON：
    [{"id": 1, "text": "数据1"}, {"id": 2, "text": "数据2"}]
    保存为：
    {"id": 1, "text": "数据1"}
    {"id": 2, "text": "数据2"}
    
    【编码说明】
    - encoding='utf-8': 支持中文字符
    - ensure_ascii=False: 保持中文字符不转义
    
    【错误处理】
    - 目录不存在：自动创建
    - 权限不足：抛出异常
    - 磁盘空间不足：抛出异常
    
    【在整体程序中的作用】
    - 为所有模块提供统一的文件保存接口
    - 确保保存格式的一致性
    - 处理文件保存的各种异常情况
    
    Args:
        data: 要保存的数据列表
        file_path: 输出文件路径
    """
    # 确保输出目录存在
    # 【预防性处理】在保存文件前确保目录存在
    ensure_directory_exists(Path(file_path).parent)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # 将每个数据项转换为JSON字符串并写入
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')  # 每行一个JSON对象
        logging.info(f"成功保存 {len(data)} 条数据到 {file_path}")
    except Exception as e:
        logging.error(f"保存文件失败 {file_path}: {e}")
        raise

def random_sample_with_seed(data: List[Any], sample_size: int, seed: int = 42) -> List[Any]:
    """
    使用固定随机种子进行随机采样
    
    【作用】
    从大量数据中随机选择指定数量的样本：
    1. 设置固定的随机种子，确保结果可重现
    2. 使用Python的random.sample进行无重复采样
    3. 处理采样数量超过数据总量的情况
    4. 返回采样后的数据列表
    
    【什么是随机采样】
    从N个数据中随机选择M个，每个数据被选中的概率相等，
    且选中的数据不会重复。
    
    【为什么要固定随机种子】
    - 可重现性：相同种子产生相同结果
    - 调试方便：每次运行结果一致，便于定位问题
    - 实验控制：确保对比实验的公平性
    
    【边界情况处理】
    如果需要采样的数量大于等于数据总量，直接返回所有数据的副本
    
    【在整体程序中的作用】
    - 为数据处理器提供标准的采样方法
    - 确保采样结果的可重现性
    - 控制数据处理的规模，避免处理过多数据
    
    Args:
        data: 原始数据列表
        sample_size: 采样数量
        seed: 随机种子
        
    Returns:
        采样后的数据列表
    """
    if sample_size >= len(data):
        logging.warning(f"采样数量({sample_size})大于等于数据总量({len(data)})，返回全部数据")
        return data.copy()  # 返回副本，避免修改原始数据
    
    # 设置随机种子确保结果可复现
    # 【临时设置种子】只影响这次采样，不影响其他随机操作
    random.seed(seed)
    sampled_data = random.sample(data, sample_size)
    
    logging.info(f"从 {len(data)} 条数据中采样了 {len(sampled_data)} 条")
    return sampled_data

def mix_data_evenly(data_list1: List[Any], data_list2: List[Any], seed: int = 42) -> List[Any]:
    """
    将两个数据列表均匀混合
    
    【作用】
    将两个不同来源的数据列表合并并打乱：
    1. 将两个列表合并成一个
    2. 使用固定种子随机打乱顺序
    3. 确保两种数据类型分布均匀
    4. 返回混合后的数据列表
    
    【为什么要均匀混合】
    如果两个列表分别代表不同类型的数据（如SFT问题和生成问题），
    简单拼接会导致数据分布不均匀，影响后续处理效果。
    随机混合确保数据分布更加均匀。
    
    【混合策略】
    1. 先合并：[A数据] + [B数据] = [所有数据]
    2. 再打乱：随机重新排列顺序
    3. 使用固定种子确保结果可重现
    
    【使用场景】
    - 合并SFT问题和新生成的问题
    - 混合不同来源的训练数据
    - 打乱数据顺序，避免顺序偏差
    
    【在整体程序中的作用】
    - 为数据处理器提供数据混合功能
    - 确保最终数据集的多样性
    - 避免数据来源的顺序偏差
    
    Args:
        data_list1: 第一个数据列表
        data_list2: 第二个数据列表
        seed: 随机种子
        
    Returns:
        混合后的数据列表
    """
    # 合并两个列表
    # 【列表拼接】使用+操作符合并列表
    combined_data = data_list1 + data_list2
    
    # 使用固定种子打乱顺序
    # 【就地打乱】shuffle会直接修改列表顺序
    random.seed(seed)
    random.shuffle(combined_data)
    
    logging.info(f"混合了 {len(data_list1)} + {len(data_list2)} = {len(combined_data)} 条数据")
    return combined_data

def validate_file_paths(file_paths: List[str]) -> List[str]:
    """
    验证文件路径是否存在
    
    【作用】
    批量检查文件是否存在，过滤出有效的文件路径：
    1. 遍历所有提供的文件路径
    2. 检查每个文件是否真实存在
    3. 记录检查结果（存在/不存在）
    4. 返回所有存在的文件路径列表
    
    【为什么要验证文件路径】
    - 早期发现问题：在程序开始处理前就发现文件缺失
    - 避免运行时错误：防止程序运行到一半才发现文件不存在
    - 提供清晰反馈：告诉用户哪些文件存在，哪些不存在
    
    【处理策略】
    - 存在的文件：记录信息，加入有效列表
    - 不存在的文件：记录警告，跳过该文件
    - 返回所有有效文件的列表
    
    【使用场景】
    - 程序启动时验证输入文件
    - 批量处理前的文件检查
    - 配置验证和错误预防
    
    【在整体程序中的作用】
    - 为所有模块提供文件存在性检查
    - 提高程序的健壮性和用户体验
    - 避免因文件不存在导致的程序崩溃
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        存在的文件路径列表
    """
    valid_paths = []
    for path in file_paths:
        if Path(path).exists():  # 使用Path对象检查文件存在性
            valid_paths.append(path)
            logging.info(f"文件存在: {path}")
        else:
            logging.warning(f"文件不存在: {path}")
    
    return valid_paths
