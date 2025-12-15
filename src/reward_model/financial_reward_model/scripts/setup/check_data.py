#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import sys

def find_data_files(data_base_path):
    """查找数据文件"""
    train_dir = os.path.join(data_base_path, "train")
    eval_dir = os.path.join(data_base_path, "eval")
    
    print(f"查找数据文件...")
    print(f"训练数据目录: {train_dir}")
    print(f"评估数据目录: {eval_dir}")
    
    # 查找训练数据文件
    train_files = []
    if os.path.exists(train_dir):
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.jsonl')]
        print(f"训练目录中的jsonl文件: {train_files}")
    
    # 查找评估数据文件
    eval_files = []
    if os.path.exists(eval_dir):
        eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.jsonl')]
        print(f"评估目录中的jsonl文件: {eval_files}")
    
    return train_files, eval_files

# ============================================================================
# 🔍 偏好数据检查函数：验证preference数据的结构和内容
# 
# 🎯 这个函数在项目中的作用：
# 1. 【数据验证】：确保训练数据存在且格式正确
# 2. 【问题预防】：在训练开始前发现数据问题
# 3. 【格式检查】：验证JSON结构是否符合期望
# 4. 【统计报告】：提供数据集的基本统计信息
# 
# ⚠️ 注意：这个函数在项目中【很少被调用】
# - 主要用于调试和问题排查
# - 不是训练流程的必需部分
# ============================================================================
def check_preference_data_structure():
    """
    检查preference数据结构和内容的主要函数
    
    这个函数会检查以下内容：
    1. 数据目录结构是否正确
    2. 数据文件是否存在
    3. JSON格式是否正确
    4. 必需字段是否完整
    5. 数据内容是否合理
    
    返回：
    - 如果检查通过：(True, train_file, eval_file)
    - 如果检查失败：(False, None, None)
    """
    
    # 第1步：设置数据路径（硬编码路径）
    data_base_path = "../reward_model_data/reward_data"
    
    print("=== 数据结构检查 ===")
    print(f"数据基础路径: {data_base_path}")
    
    # 第2步：检查基础路径是否存在
    if not os.path.exists(data_base_path):
        print(f"❌ 数据基础路径不存在: {data_base_path}")
        return False
    
    # 第3步：查找数据文件
    train_files, eval_files = find_data_files(data_base_path)
    
    # 第4步：选择数据文件（优先使用preference_dataset.jsonl）
    train_file = "preference_dataset.jsonl" if "preference_dataset.jsonl" in train_files else (train_files[0] if train_files else None)
    eval_file = "preference_dataset.jsonl" if "preference_dataset.jsonl" in eval_files else (eval_files[0] if eval_files else None)
    
    # 第5步：验证数据文件存在性
    if not train_file:
        print(f"❌ 训练目录中没有找到jsonl文件")
        return False
    
    if not eval_file:
        print(f"❌ 评估目录中没有找到jsonl文件")
        return False
    
    # 第6步：构建完整文件路径
    train_path = os.path.join(data_base_path, "train", train_file)
    eval_path = os.path.join(data_base_path, "eval", eval_file)
    
    print(f"✅ 使用训练文件: {train_file}")
    print(f"✅ 使用评估文件: {eval_file}")
    
    # ========================================================================
    # 📊 数据内容检查子函数：验证单个数据文件的内容
    # 
    # 🎯 这个内部函数的作用：
    # 1. 【格式验证】：检查每行是否是有效的JSON
    # 2. 【字段检查】：验证必需字段是否存在
    # 3. 【内容分析】：分析数据内容的合理性
    # 4. 【统计报告】：提供数据集的详细统计
    # ========================================================================
    def check_file_content(file_path, split_name):
        """
        检查单个数据文件的内容
        
        参数说明：
        - file_path: 要检查的文件路径
        - split_name: 数据集名称（"训练" 或 "评估"）
        
        返回：
        - True: 检查通过
        - False: 检查失败
        """
        print(f"\n--- {split_name} 数据检查 ---")
        try:
            # 读取所有行
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"数据行数: {len(lines)}")
            
            # 检查是否为空
            if len(lines) == 0:
                print(f"❌ {split_name} 数据为空")
                return False
            
            # 检查前几行数据格式（避免检查所有数据，节省时间）
            for i, line in enumerate(lines[:2]):  # 只检查前2行作为样本
                try:
                    data = json.loads(line.strip())  # 解析JSON
                    
                    print(f"样本 {i+1}:")
                    print(f"  字段: {list(data.keys())}")
                    
                    # 检查question字段
                    if "question" in data:
                        print(f"  ✅ 包含question字段")
                        print(f"  问题长度: {len(data['question'])}")
                    else:
                        print(f"  ❌ 缺少question字段")
                        return False
                    
                    # 检查answers字段（用于preference训练）
                    if "answers" in data:
                        print(f"  ✅ 包含answers字段")
                        if isinstance(data["answers"], list):
                            print(f"  答案数量: {len(data['answers'])}")
                            if len(data["answers"]) >= 2:
                                print(f"  ✅ 至少有2个答案，可用于preference训练")
                            else:
                                print(f"  ⚠️  只有{len(data['answers'])}个答案，可能不足以进行preference训练")
                        else:
                            print(f"  ⚠️  answers不是列表格式")
                    
                    # 检查其他字段并统计信息
                    for key, value in data.items():
                        if key not in ["question", "answers"]:
                            if isinstance(value, str):
                                print(f"  {key}: {len(value)} 字符")
                            else:
                                print(f"  {key}: {type(value).__name__}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ 第{i+1}行JSON格式错误: {e}")
                    return False
            
            print(f"✅ {split_name} 数据格式基本正确")
            return True
            
        except Exception as e:
            print(f"❌ 读取{split_name}数据时出错: {e}")
            return False
    
    # 第7步：检查训练和评估数据
    train_ok = check_file_content(train_path, "训练")
    eval_ok = check_file_content(eval_path, "评估")
    
    # 第8步：汇总检查结果
    if train_ok and eval_ok:
        print("\n🎉 数据检查通过！")
        print(f"📝 数据文件信息:")
        print(f"  训练文件: {train_file}")
        print(f"  评估文件: {eval_file}")
        print(f"\n⚠️  注意: 需要根据实际数据格式修改数据集类")
        return True, train_file, eval_file
    else:
        print("\n❌ 数据检查失败，请修复数据问题后再试。")
        return False, None, None


# ============================================================================
# 🤖 模型路径检查函数：验证预训练模型文件是否完整
# 
# 🎯 这个函数在项目中的作用：
# 1. 【模型验证】：确保预训练模型文件存在且完整
# 2. 【路径检查】：验证模型路径配置是否正确
# 3. 【文件完整性】：检查关键模型文件是否齐全
# 4. 【格式支持】：检查模型格式（safetensors/pytorch）
# 
# ⚠️ 注意：这个函数在项目中【很少被调用】
# - 主要用于环境配置验证
# - 硬编码了模型路径，可能需要根据实际情况调整
# ============================================================================
def check_model_path():
    """
    检查预训练模型路径和文件完整性
    
    这个函数会检查以下内容：
    1. 模型目录是否存在
    2. 关键配置文件是否存在（config.json, tokenizer.json）
    3. 模型权重文件是否存在（safetensors或pytorch格式）
    4. 文件格式是否支持
    
    返回：
    - True: 模型检查通过
    - False: 模型检查失败
    """
    # 硬编码的模型路径（⚠️ 可能需要根据实际情况调整）
    model_path = "/shared/QRM-Llama3.1-8B-v2"
    
    print(f"\n=== 模型路径检查 ===")
    print(f"模型路径: {model_path}")
    
    # 第1步：检查模型目录是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 第2步：检查关键配置文件
    required_files = ["config.json", "tokenizer.json"]  # 必需的配置文件
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if os.path.exists(file_path):
            print(f"✅ {file_name} 存在")
        else:
            missing_files.append(file_name)
    
    # 第3步：检查模型权重文件
    model_files_found = False
    
    # 获取目录中的所有文件
    all_files = os.listdir(model_path)
    
    # 检查safetensors格式（推荐格式）
    safetensors_files = [f for f in all_files if f.endswith(".safetensors")]
    if safetensors_files:
        print(f"✅ 找到safetensors模型文件: {len(safetensors_files)} 个")
        model_files_found = True
    
    # 检查pytorch格式（传统格式）
    pytorch_files = [f for f in all_files if f.endswith(".bin") and "pytorch_model" in f]
    if pytorch_files:
        print(f"✅ 找到pytorch模型文件: {len(pytorch_files)} 个")
        model_files_found = True
    
    # 第4步：检查是否找到模型权重文件
    if not model_files_found:
        print(f"❌ 未找到模型权重文件")
        missing_files.append("model_weights")
    
    # 第5步：汇总检查结果
    if missing_files:
        print(f"❌ 缺少关键文件: {missing_files}")
        return False
    
    print("✅ 模型文件检查通过")
    return True


# ============================================================================
# 🚀 主执行函数：脚本的入口点，协调所有检查流程
# 
# 🎯 这个部分在项目中的作用：
# 1. 【流程协调】：协调数据检查和模型检查
# 2. 【结果汇总】：汇总所有检查结果
# 3. 【状态返回】：通过退出码告知调用者检查结果
# 4. 【用户友好】：提供清晰的成功/失败信息
# 
# ⚠️ 注意：这个脚本在项目中【很少被实际使用】
# - 被run_training.sh引用，但run_training.sh本身也未被使用
# - 主要用于手动调试和环境验证
# ============================================================================
if __name__ == "__main__":
    """
    脚本主入口：当直接运行此脚本时执行
    
    执行流程：
    1. 执行数据结构和内容检查
    2. 执行模型路径和文件检查
    3. 汇总所有检查结果
    4. 根据检查结果设置退出码
    
    退出码说明：
    - 0: 所有检查通过
    - 1: 检查失败
    """
    print("开始环境和数据检查...\n")
    
    # 第1步：执行数据检查
    data_result = check_preference_data_structure()
    if isinstance(data_result, tuple):
        data_ok, train_file, eval_file = data_result  # 解包返回值
    else:
        data_ok = data_result
        train_file = eval_file = None
    
    # 第2步：执行模型检查
    model_ok = check_model_path()
    
    # 第3步：汇总检查结果
    print(f"\n=== 检查结果汇总 ===")
    print(f"数据检查: {'✅ 通过' if data_ok else '❌ 失败'}")
    print(f"模型检查: {'✅ 通过' if model_ok else '❌ 失败'}")
    
    # 第4步：根据检查结果提供相应的信息和退出码
    if data_ok and model_ok:
        print("\n🎉 所有检查通过！环境准备就绪。")
        if train_file and eval_file:
            print(f"\n📝 建议使用的数据文件:")
            print(f"   训练文件: {train_file}")
            print(f"   评估文件: {eval_file}")
        sys.exit(0)  # 成功退出
    else:
        print("\n❌ 检查失败，请修复问题后再试。")
        sys.exit(1)  # 失败退出