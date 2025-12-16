#!/usr/bin/env python3
"""
GPTQModel INT4量化脚本 - 针对金融GRPO模型

这个脚本的主要作用：
1. 将原始的大语言模型从FP16/BF16精度压缩到INT4精度
2. 使用GPTQ算法进行量化，保持模型性能的同时显著减少模型大小
3. 量化后的模型可以在相同硬件上运行更大的模型或获得更快的推理速度

核心概念：
- 量化：将模型权重从32位/16位浮点数转换为4位整数，大幅减少存储空间
- 校准数据：用于统计模型激活分布的样本数据，帮助确定最佳量化参数
- GPTQ：一种先进的量化算法，通过二阶信息优化量化误差
"""

# 导入必要的库
import torch                                         # PyTorch深度学习框架
import json                                          # JSON数据处理
from transformers import AutoTokenizer               # 用于处理文本tokenization
from gptqmodel import GPTQModel, QuantizeConfig     # GPTQModel量化框架
from pathlib import Path                             # 文件路径处理
import time                                          # 时间计算

class FinanceModelQuantizer:
    """
    金融模型量化器类
    
    主要功能：
    1. 加载原始模型和校准数据
    2. 配置量化参数
    3. 执行量化过程
    4. 保存量化后的模型
    5. 对比模型大小变化
    """
    
    def __init__(self, model_path: str, calibration_data_path: str):
        """
        初始化量化器
        
        参数说明：
        model_path: 原始模型的路径，包含模型权重和配置文件
        calibration_data_path: 校准数据文件路径，用于量化过程中的统计信息收集
        """
        self.model_path = model_path                     # 存储原始模型路径
        self.calibration_data_path = calibration_data_path  # 存储校准数据路径
        self.tokenizer = None                            # 用于文本处理的分词器，初始化为空
        self.model = None                                # 用于存储加载的模型，初始化为空
        
    def load_calibration_data(self) -> list:
        """
        加载校准数据
        
        校准数据的作用：
        1. 量化过程中，需要通过校准数据来统计模型各层的激活分布
        2. 根据激活分布来确定最佳的量化参数（如缩放因子、零点等）
        3. 校准数据应该代表模型的实际使用场景，这样量化后的模型才能保持性能
        
        数据格式要求：
        - JSONL格式，每行一个JSON对象
        - 每个对象必须包含'text'字段
        - 文本长度建议在10-2048个字符之间
        
        返回值：
        calibration_data: 校准文本列表，每个元素是一个用于校准的文本字符串
        
        异常处理：
        - 跳过格式错误的行
        - 过滤掉过短的文本
        - 如果没有有效数据会返回空列表
        """
        print("正在加载校准数据...")
        calibration_data = []  # 存储校准文本的列表
        
        # 逐行读取校准数据文件
        with open(self.calibration_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 解析JSON格式的每一行
                    data = json.loads(line.strip())
                    # 提取文本内容，去除首尾空白字符
                    text = data.get('text', '').strip()
                    # 过滤掉太短的文本，确保校准数据质量
                    # 长度阈值设为10，避免无意义的短文本影响量化效果
                    if text and len(text) > 10:
                        calibration_data.append(text)
                except:
                    # 如果某行数据格式错误，跳过该行继续处理
                    continue
        
        print(f"成功加载了 {len(calibration_data)} 条校准数据")
        return calibration_data
    
    def setup_quantization_config(self) -> QuantizeConfig:
        """
        设置量化配置
        
        量化配置参数详解：
        - bits: 量化位数，4表示将权重量化为4位整数
        - group_size: 分组大小，128表示每128个权重为一组共享量化参数
                     较小的值(64)精度更高但速度慢，较大的值(256)速度快但精度略低
        - damp_percent: 阻尼系数，0.01表示在优化过程中的正则化强度
                       防止量化过程中的数值不稳定，通常设为0.01
        - desc_act: 是否启用描述符激活，False表示不启用（提升速度）
                   启用会提高精度但降低推理速度，金融场景通常关闭
        - static_groups: 是否使用静态分组，False表示动态分组
                        动态分组可以获得更好的量化效果
        - sym: 是否使用对称量化，True表示使用对称量化（更简单）
               对称量化计算更快，非对称量化精度略高
        - true_sequential: 是否使用真正的顺序量化，True表示按层顺序量化
                          确保量化过程的稳定性和可重复性
        
        返回值：
        quantize_config: 配置好的量化参数对象
        """
        print("正在设置量化配置...")
        
        quantize_config = QuantizeConfig(
            bits=4,                    # 量化位数：4位整数，相比16位浮点数大幅减少存储
            group_size=128,            # 分组大小：每128个权重共享一个量化参数，平衡精度和压缩率
            damp_percent=0.01,         # 阻尼系数：较小的值(0.01)减少量化过程中的数值震荡
            desc_act=False,            # 描述符激活：关闭以提升推理速度，对精度影响较小
            static_groups=False,       # 静态分组：关闭以允许动态优化分组策略
            sym=True,                  # 对称量化：启用对称量化，简化计算过程
            true_sequential=True,      # 顺序量化：按模型层的顺序进行量化，确保稳定性
        )
        
        print("量化配置设置完成")
        return quantize_config
    
    def quantize_model(self, output_path: str):
        """
        执行模型量化的主要流程
        
        参数说明：
        output_path: 量化后模型的保存路径
                    会自动创建目录结构，保存模型权重和配置文件
        
        量化流程：
        1. 加载校准数据 -> 2. 设置量化配置 -> 3. 加载原始模型 
        -> 4. 执行量化计算 -> 5. 保存量化模型 -> 6. 对比模型大小
        
        内存管理：
        - 使用torch.float16减少内存占用
        - 采用device_map="auto"自动分配GPU/CPU
        - 量化过程中可能需要大量内存，建议至少32GB
        
        时间预估：
        - 7B模型通常需要10-30分钟
        - 13B模型通常需要30-60分钟
        - 具体时间取决于硬件配置和校准数据量
        """
        print(f"开始量化模型: {self.model_path}")
        
        # 步骤1: 加载校准数据
        # 校准数据用于统计模型在实际数据上的激活分布
        calibration_data = self.load_calibration_data()
        
        # 步骤2: 设置量化配置
        # 配置量化算法的各种参数
        quantize_config = self.setup_quantization_config()
        
        # 步骤3: 加载分词器
        # 分词器用于将文本转换为模型可理解的token序列
        print("正在加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 步骤4: 加载原始模型
        # 使用GPTQModel框架加载模型，准备进行量化
        print("正在加载原始模型...")
        self.model = GPTQModel.from_pretrained(
            self.model_path,                    # 模型路径
            quantize_config=quantize_config,    # 量化配置
            device_map="auto",                  # 自动分配设备（CPU/GPU）
            torch_dtype=torch.float16,          # 使用float16精度以节省内存
            trust_remote_code=True              # 允许执行模型中的自定义代码
        )
        
        # 步骤5: 执行量化过程
        # 这是最耗时的步骤，会计算量化参数并转换模型权重
        print("正在进行量化计算...")
        start_time = time.time()  # 记录开始时间
        
        # 核心量化步骤：使用校准数据来优化量化参数
        # 这个过程会：
        # 1. 将校准数据输入模型获取激活值
        # 2. 统计每层的激活分布
        # 3. 计算最优的量化参数（缩放因子、零点）
        # 4. 将FP16权重转换为INT4权重
        self.model.quantize(calibration_data)
        
        end_time = time.time()  # 记录结束时间
        print(f"量化计算完成！耗时: {end_time - start_time:.2f} 秒")
        
        # 步骤6: 保存量化模型
        # 将量化后的模型和分词器保存到指定路径
        print(f"正在保存量化模型到: {output_path}")
        
        # 创建输出目录（如果不存在）
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        self.model.save_quantized(output_path)      # 保存量化后的模型权重
        self.tokenizer.save_pretrained(output_path)  # 保存分词器配置
        
        # 步骤7: 对比模型大小
        # 展示量化前后的模型大小变化
        self.compare_model_sizes(output_path)
        
        print("模型量化流程全部完成！")
    
    def compare_model_sizes(self, quantized_path: str):
        """
        对比原始模型和量化模型的大小
        支持 .bin 和 .safetensors 格式
        
        参数说明：
        quantized_path: 量化模型的保存路径
        
        支持格式：
        - .bin: 传统的PyTorch模型格式
        - .safetensors: 新的安全张量格式，加载更快更安全
        
        计算指标：
        - 绝对大小（GB）
        - 压缩比（倍数）
        - 节省空间（GB）
        - 压缩率（百分比）
        """
        try:
            # 支持多种模型文件格式
            def calculate_model_size(path):
                """
                计算模型文件总大小
                
                参数说明：
                path: 模型目录路径
                
                返回值：
                total_size: 总大小（字节）
                bin_count: .bin文件数量
                safetensors_count: .safetensors文件数量
                """
                total_size = 0
                path_obj = Path(path)
                
                # 查找 .bin 文件（传统PyTorch格式）
                bin_files = list(path_obj.rglob('*.bin'))
                total_size += sum(f.stat().st_size for f in bin_files if f.is_file())
                
                # 查找 .safetensors 文件（新格式，更安全）
                safetensors_files = list(path_obj.rglob('*.safetensors'))
                total_size += sum(f.stat().st_size for f in safetensors_files if f.is_file())
                
                return total_size, len(bin_files), len(safetensors_files)
            
            # 计算原始模型大小
            original_size, orig_bin_count, orig_safetensors_count = calculate_model_size(self.model_path)
            
            # 计算量化模型大小
            quantized_size, quant_bin_count, quant_safetensors_count = calculate_model_size(quantized_path)
            
            print(f"\n文件格式检测:")
            print(f"原始模型: {orig_bin_count} 个 .bin 文件, {orig_safetensors_count} 个 .safetensors 文件")
            print(f"量化模型: {quant_bin_count} 个 .bin 文件, {quant_safetensors_count} 个 .safetensors 文件")
            
            # 检查是否找到了模型文件
            if original_size == 0:
                print("警告: 未找到原始模型文件")
                print("请检查模型路径是否正确")
                return
            
            if quantized_size == 0:
                print("警告: 未找到量化模型文件")
                print("量化过程可能未成功完成")
                return
            
            # 转换为GB单位，方便阅读
            original_gb = original_size / (1024**3)        # 将字节转换为GB
            quantized_gb = quantized_size / (1024**3)      # 将字节转换为GB
            compression_ratio = original_size / quantized_size  # 计算压缩比
            
            # 显示详细的对比结果
            print(f"\n模型大小对比:")
            print(f"原始模型: {original_gb:.2f} GB")
            print(f"量化模型: {quantized_gb:.2f} GB")
            print(f"压缩比: {compression_ratio:.2f}倍")
            print(f"节省空间: {(original_gb - quantized_gb):.2f} GB")
            print(f"压缩率: {((original_size - quantized_size) / original_size * 100):.2f}%")
            
        except Exception as e:
            # 如果计算过程中出现错误，显示详细的警告信息
            print(f"无法计算模型大小对比: {e}")

def main():
    """
    主函数：配置路径并执行量化流程
    
    使用说明：
    1. 修改model_path为你的原始模型路径
    2. 修改calibration_data_path为你的校准数据路径
    3. 修改output_path为你想要保存量化模型的路径
    4. 运行脚本即可开始量化过程
    """
    # 配置路径
    model_path = "/shared/grpo_financial_tuning/output/best_model"  # 原始模型路径
    calibration_data_path = "calibration_data.jsonl"  # 校准数据路径
    output_path = "./output/best_model_gptq_int4"     # 量化模型输出路径
    
    # 检查路径是否存在
    if not Path(model_path).exists():
        print(f"错误：模型路径不存在: {model_path}")
        return
    
    if not Path(calibration_data_path).exists():
        print(f"错误：校准数据路径不存在: {calibration_data_path}")
        return
    
    # 创建量化器并执行量化
    quantizer = FinanceModelQuantizer(model_path, calibration_data_path)
    quantizer.quantize_model(output_path)

if __name__ == "__main__":
    main()
