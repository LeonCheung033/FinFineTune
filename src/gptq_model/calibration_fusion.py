#!/usr/bin/env python3
"""
量化校准数据融合工具 - JSON格式输出

这个工具的主要作用是：
1. 从不同格式的训练数据文件中提取文本
2. 将这些文本按指定比例融合成一个统一的校准数据集
3. 用于模型量化时的校准过程

支持的数据格式：
- SFT格式：包含instruction、input、output字段
- GRPO格式：包含prompt字段
"""

# 导入必要的库
import json       # 用于处理JSON格式的数据
import random     # 用于随机采样和打乱数据
import argparse   # 用于处理命令行参数
from pathlib import Path      # 用于处理文件路径
from typing import List, Dict # 用于类型注解，提高代码可读性

class CalibrationDataFusion:
    """
    校准数据融合类
    
    这个类的主要功能是将多个不同格式的训练数据文件
    融合成一个统一的校准数据集，用于模型量化
    """
    
    def __init__(self, random_seed: int = 42):
        """
        初始化方法
        
        参数说明：
        - random_seed: 随机种子，用于保证结果的可重复性
                      相同的种子会产生相同的随机结果
        """
        self.random_seed = random_seed  # 保存随机种子
        random.seed(random_seed)        # 设置随机种子，确保结果可重复
    
    def extract_sft_texts(self, sft_file: str) -> List[Dict]:
        """
        从SFT格式数据文件中提取文本
        
        SFT格式说明：
        - instruction: 指令文本
        - input: 输入文本  
        - output: 输出文本（可能包含<think>标签）
        
        参数说明：
        - sft_file: SFT格式数据文件的路径
        
        返回值：
        - 返回提取的文本列表，每个元素包含text、source、file字段
        """
        texts = []  # 用于存储提取的文本
        
        # 打开文件并逐行读取
        with open(sft_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 将每行JSON数据解析为Python字典
                    data = json.loads(line.strip())
                    
                    # 第一步：提取instruction和input，合并成prompt
                    instruction = data.get('instruction', '').strip()  # 获取指令，如果没有则为空字符串
                    input_text = data.get('input', '').strip()         # 获取输入，如果没有则为空字符串
                    
                    # 如果instruction和input都存在，则合并它们
                    if instruction and input_text:
                        prompt = f"{instruction}\n{input_text}"  # 用换行符连接
                        
                        # 只保留长度大于20的文本，过滤掉太短的无效文本
                        if len(prompt) > 20:
                            texts.append({
                                "text": prompt,           # 实际的文本内容
                                "source": "sft_prompt",   # 标记来源类型
                                "file": Path(sft_file).name  # 记录来源文件名
                            })
                    
                    # 第二步：提取output作为回复文本
                    output = data.get('output', '').strip()  # 获取输出文本
                    
                    # 如果output存在且长度足够，则保存
                    if output and len(output) > 20:
                        texts.append({
                            "text": output,              # 实际的文本内容
                            "source": "sft_response",    # 标记来源类型
                            "file": Path(sft_file).name  # 记录来源文件名
                        })
                        
                except Exception:
                    # 如果某行数据解析失败，跳过继续处理下一行
                    continue
        
        return texts  # 返回提取的所有文本
    
    def extract_grpo_texts(self, grpo_file: str) -> List[Dict]:
        """
        从GRPO格式数据文件中提取文本
        
        GRPO格式说明：
        - prompt: 提示文本，用于强化学习训练
        
        参数说明：
        - grpo_file: GRPO格式数据文件的路径
        
        返回值：
        - 返回提取的文本列表，每个元素包含text、source、file字段
        """
        texts = []  # 用于存储提取的文本
        
        # 打开文件并逐行读取
        with open(grpo_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 将每行JSON数据解析为Python字典
                    data = json.loads(line.strip())
                    
                    # 提取prompt字段
                    prompt = data.get('prompt', '').strip()  # 获取提示文本
                    
                    # 只保留长度大于20的文本，过滤掉太短的无效文本
                    if prompt and len(prompt) > 20:
                        texts.append({
                            "text": prompt,              # 实际的文本内容
                            "source": "grpo_prompt",     # 标记来源类型
                            "file": Path(grpo_file).name # 记录来源文件名
                        })
                        
                except Exception:
                    # 如果某行数据解析失败，跳过继续处理下一行
                    continue
        
        return texts  # 返回提取的所有文本
    
    def detect_file_type(self, file_path: str) -> str:
        """
        自动检测文件的数据格式类型
        
        检测逻辑：
        1. 优先通过文件内容的字段来判断
        2. 如果内容检测失败，则通过文件名来判断
        
        参数说明：
        - file_path: 要检测的文件路径
        
        返回值：
        - 'sft': SFT格式文件
        - 'grpo': GRPO格式文件  
        - 'unknown': 未知格式文件
        """
        # 方法1：通过文件内容检测
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 解析JSON数据
                    data = json.loads(line.strip())
                    
                    # 如果包含instruction和output字段，判断为SFT格式
                    if 'instruction' in data and 'output' in data:
                        return 'sft'
                    # 如果包含prompt字段，判断为GRPO格式
                    elif 'prompt' in data:
                        return 'grpo'
                except:
                    # 如果解析失败，继续检查下一行
                    continue
        
        # 方法2：通过文件名检测
        file_name = Path(file_path).name.lower()  # 获取文件名并转为小写
        
        if 'sft' in file_name:
            return 'sft'
        elif 'grpo' in file_name:
            return 'grpo'
        else:
            return 'unknown'  # 无法识别的格式
    
    def fuse_data(self, input_files: List[str], output_file: str, 
                  total_samples: int, ratios: List[float]) -> None:
        """
        融合多个数据文件的核心方法
        
        工作流程：
        1. 检测每个文件的格式类型
        2. 根据格式提取文本数据
        3. 按指定比例从每个文件中采样
        4. 将所有文本合并并随机打乱
        5. 保存为统一的JSON格式文件
        
        参数说明：
        - input_files: 输入文件路径列表
        - output_file: 输出文件路径
        - total_samples: 最终要生成的总样本数
        - ratios: 每个文件的采样比例列表
        """
        print(f"融合 {len(input_files)} 个文件，目标样本数: {total_samples}")
        
        # 第一步：归一化比例
        # 确保所有比例加起来等于1.0
        if abs(sum(ratios) - 1.0) > 0.01:  # 如果比例和不等于1
            total_ratio = sum(ratios)      # 计算当前比例总和
            ratios = [r / total_ratio for r in ratios]  # 重新计算比例
        
        all_texts = []  # 用于存储所有提取的文本
        
        # 第二步：处理每个输入文件
        for i, file_path in enumerate(input_files):
            print(f"处理文件: {Path(file_path).name}")
            
            # 检测文件类型
            file_type = self.detect_file_type(file_path)
            
            # 根据文件类型选择相应的提取方法
            if file_type == 'sft':
                texts = self.extract_sft_texts(file_path)
            elif file_type == 'grpo':
                texts = self.extract_grpo_texts(file_path)
            else:
                print(f"跳过未知格式文件: {file_path}")
                continue  # 跳过无法识别的文件
            
            # 第三步：按比例采样
            target_samples = int(total_samples * ratios[i])  # 计算当前文件应该提供的样本数
            
            if len(texts) > target_samples:
                # 如果文本数量超过目标，随机采样
                selected_texts = random.sample(texts, target_samples)
            else:
                # 如果文本数量不足，全部使用
                selected_texts = texts
                if len(texts) < target_samples:
                    print(f"警告：{file_path} 样本不足，实际获得 {len(texts)} 个")
            
            # 将选中的文本添加到总列表中
            all_texts.extend(selected_texts)
            print(f"已添加 {len(selected_texts)} 个样本")
        
        # 第四步：随机打乱并截取
        random.shuffle(all_texts)  # 随机打乱所有文本
        
        # 如果总数超过目标数量，截取到目标数量
        if len(all_texts) > total_samples:
            all_texts = all_texts[:total_samples]
        
        # 第五步：保存为JSON格式文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_texts:
                # 每行写入一个JSON对象
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 输出完成信息
        print(f"融合完成！最终样本数: {len(all_texts)}")
        print(f"输出文件: {output_file}")

def main():
    """
    主函数 - 处理命令行参数并执行数据融合
    
    支持的命令行参数：
    - --input_files: 输入文件路径列表
    - --output_file: 输出文件路径
    - --total_samples: 总样本数
    - --ratios: 各文件的比例列表
    - --random_seed: 随机种子
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='量化校准数据融合工具')
    
    # 添加各种命令行参数
    parser.add_argument('--input_files', '-i', nargs='+', required=True,
                        help='输入文件路径列表')
    parser.add_argument('--output_file', '-o', required=True,
                        help='输出文件路径')
    parser.add_argument('--total_samples', '-n', type=int, default=512,
                        help='总样本数 (默认: 512)')
    parser.add_argument('--ratios', '-r', nargs='+', type=float,
                        help='各文件的比例列表 (例如: 0.7 0.3)')
    parser.add_argument('--random_seed', '-s', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置默认比例
    if args.ratios is None:
        if len(args.input_files) == 2:
            # 如果有2个文件，默认比例为7:3
            args.ratios = [0.7, 0.3]
        else:
            # 如果有多个文件，平均分配比例
            ratio = 1.0 / len(args.input_files)
            args.ratios = [ratio] * len(args.input_files)
    
    # 检查参数有效性
    if len(args.ratios) != len(args.input_files):
        print("错误：比例数量必须与输入文件数量相同")
        return
    
    # 执行数据融合
    fusion = CalibrationDataFusion(random_seed=args.random_seed)
    fusion.fuse_data(args.input_files, args.output_file, args.total_samples, args.ratios)

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    main()
