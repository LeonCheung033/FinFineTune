import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from rouge import Rouge

# 定义命令行参数解析器
# 这个工具允许用户在运行脚本时通过命令行传入不同的参数，而不需要修改代码
parser = argparse.ArgumentParser(description='评估微调后的模型效果')
parser.add_argument('--model_path', type=str, default="/root/autodl-tmp/finetune_output/fianl_complete_model", 
                    help='微调后的完整模型路径')  # 指定要评估的模型位置
parser.add_argument('--test_file', type=str, default="/root/autodl-tmp/data/sft/deepspeek_test_sft_dataset_0_2k.jsonl", 
                    help='测试数据集路径')  # 指定测试数据的位置
parser.add_argument('--batch_size', type=int, default=16, help='批次大小')  # 一次处理多少个样本
parser.add_argument('--max_length', type=int, default=1024, help='最大序列长度')  # 输入文本的最大长度限制
parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备，可以是cuda:0或auto等')  # 指定使用GPU还是CPU
parser.add_argument('--sample_size', type=int, default=200, help='评估样本数量，设为-1使用全部')  # 评估多少个样本
parser.add_argument('--max_new_tokens', type=int, default=2048, 
                    help='生成的最大新token数量')  # 模型生成回答时的最大长度
args = parser.parse_args()

# 启用TensorFloat32优化
# 这是一种GPU计算优化技术，可以提高计算速度而不显著影响精度
torch.set_float32_matmul_precision('high')

# 自定义数据集类，用于加载和预处理标准SFT格式的测试数据
# Dataset是PyTorch提供的数据集基类，我们继承它来创建自己的数据集
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        """
        初始化数据集
        参数:
            data_path: 数据文件路径 - 测试数据存放的位置
            tokenizer: 分词器 - 将文本转换为模型能理解的数字序列
            max_length: 最大序列长度 - 防止文本过长导致内存不足
        """
        self.tokenizer = tokenizer  # 保存分词器，用于后续文本处理
        self.max_length = max_length  # 保存最大长度限制
        self.instructions = []  # 存储指令 - 告诉模型要做什么的部分
        self.inputs = []        # 存储输入 - 具体的问题或背景信息
        self.outputs = []       # 存储期望输出 - 标准答案，用于对比评估
        self.prompts = []       # 存储完整提示 - 指令+输入组合成的完整问题
        
        total_lines = 0      # 统计文件总行数
        processed_lines = 0  # 统计成功处理的行数
        
        try:
            # 打开测试数据文件，每行是一个JSON格式的样本
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    total_lines += 1
                    try:
                        # 解析JSON格式的数据
                        item = json.loads(line)
                        
                        # 提取标准SFT格式的字段
                        # instruction: 指令，告诉模型要执行什么任务
                        # input: 输入，具体的问题或数据
                        # output: 输出，期望的答案
                        instruction = item.get("instruction", "")
                        input_text = item.get("input", "")
                        output = item.get("output", "")
                        
                        # 确保有输出 - 没有标准答案就无法评估
                        if output:
                            self.instructions.append(instruction)
                            self.inputs.append(input_text)
                            self.outputs.append(output)
                            
                            # 构建提示模板 - 将指令和输入组合成完整的问题
                            # 这个格式要与训练时使用的格式保持一致
                            if input_text:
                                prompt = f"{instruction}\n\n{input_text}\n"
                            else:
                                prompt = f"{instruction}\n"
                                
                            self.prompts.append(prompt)
                            processed_lines += 1
                    except Exception as e:
                        print(f"处理第{total_lines}行时出错: {e}")
        except Exception as e:
            print(f"打开文件时出错: {e}")
        
        # 打印数据加载统计信息，帮助用户了解数据质量
        print(f"数据集{data_path} - 总行数: {total_lines}, 成功处理: {processed_lines}, 最终样本数: {len(self.prompts)}")
    
    def __len__(self):
        """返回数据集大小 - PyTorch要求实现这个方法"""
        return len(self.prompts)
    
    def __getitem__(self, idx):
        """获取指定索引的样本 - PyTorch要求实现这个方法"""
        return {
            "prompt": self.prompts[idx],        # 完整的问题
            "output": self.outputs[idx],        # 标准答案
            "instruction": self.instructions[idx],  # 指令部分
            "input": self.inputs[idx]           # 输入部分
        }

def compute_perplexity(model, tokenizer, dataset, device, batch_size=16):
    """
    计算模型在测试集上的困惑度
    
    困惑度(Perplexity)是评估语言模型的重要指标：
    - 困惑度越低，说明模型对文本的预测越准确
    - 可以理解为模型在预测下一个词时的"困惑程度"
    - 数学上等于损失函数的指数：perplexity = exp(loss)
    
    参数:
        model: 待评估的模型
        tokenizer: 分词器
        dataset: 测试数据集
        device: 计算设备(GPU或CPU)
        batch_size: 批次大小，一次处理多少个样本
    
    返回:
        perplexity: 困惑度值，越低表示模型效果越好
    """
    model.eval()  # 设置为评估模式，关闭dropout等训练时的随机性
    dataloader = DataLoader(dataset, batch_size=batch_size)  # 创建数据加载器，支持批量处理
    
    total_loss = 0  # 累积总损失
    total_tokens = 0  # 累积总token数，用于计算平均损失
    
    with torch.no_grad():  # 不计算梯度，节省内存和计算时间
        for batch in tqdm(dataloader, desc="计算困惑度"):  # tqdm显示进度条
            prompts = batch["prompt"]    # 获取问题
            outputs = batch["output"]    # 获取标准答案
            
            # 构建输入序列 - 提示+输出
            # 这样模型需要预测整个回答，我们可以计算预测的准确性
            texts = [p + o for p, o in zip(prompts, outputs)]
            
            # 批量编码 - 将文本转换为模型能理解的数字序列
            encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
            input_ids = encodings["input_ids"].to(device)        # 输入序列
            attention_mask = encodings["attention_mask"].to(device)  # 注意力掩码，告诉模型哪些位置是有效的
            
            # 创建标签并找到提示结束位置
            # 标签用于计算损失，我们只想计算回答部分的损失，不包括问题部分
            labels = input_ids.clone()
            
            # 批量处理标签
            for i, p in enumerate(prompts):
                # 计算提示token数量
                prompt_tokens = len(tokenizer.encode(p, add_special_tokens=False))
                # 将提示部分的标签设为-100，这样计算损失时会忽略这部分
                # 只计算模型生成回答部分的损失
                labels[i, :prompt_tokens] = -100
            
            # 计算损失 - 模型预测与真实答案的差距
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 累加损失和有效token数
            # 只统计非忽略的token（即回答部分）
            non_ignored = (labels != -100).sum().item()
            total_loss += loss.item() * non_ignored
            total_tokens += non_ignored
            
            # 清理GPU内存，防止内存溢出
            del input_ids, attention_mask, labels, outputs
            torch.cuda.empty_cache()
    
    # 计算困惑度 = exp(平均损失)
    # 平均损失越小，困惑度越小，模型效果越好
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def generate_answers(model, tokenizer, dataset, device, batch_size=32, max_new_tokens=2048):
    """
    使用模型批量生成回答 - 优化版，防止输出不完整
    
    这个函数让模型根据问题生成回答，然后与标准答案对比
    这是评估模型实际应用效果的重要方法
    """
    model.eval()  # 设置为评估模式
    generated_answers = []  # 存储模型生成的回答
    reference_answers = []  # 存储标准答案
    
    # 创建批次 - 将数据分批处理，提高效率
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(dataloader)
    print(f"开始生成回答，共{total_batches}个批次...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="生成回答")):
        prompts = batch["prompt"]      # 获取问题
        batch_outputs = batch["output"]  # 获取标准答案
        
        try:
            # 批量编码输入 - 将问题转换为模型能理解的格式
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # 增强生成参数，确保输出完整
            # 这些参数控制模型如何生成回答
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,      # 最大生成长度，确保回答足够完整
                    min_new_tokens=100,       # 最小生成长度，强制生成一定长度的回答
                    do_sample=True,           # 使用采样而非贪心搜索，增加回答多样性
                    temperature=0.8,          # 温度参数，控制生成的随机性，0.8是较好的平衡
                    top_p=0.95,               # 核采样参数，只考虑概率最高的95%的词汇
                    num_beams=4,              # 束搜索数量，平衡质量和速度
                    repetition_penalty=1.2,   # 重复惩罚，避免生成重复内容
                    length_penalty=1.0,       # 长度惩罚，1.0表示不惩罚长度
                    no_repeat_ngram_size=3,   # 避免重复的n-gram大小
                    pad_token_id=tokenizer.eos_token_id,  # 填充token
                    # 注意：在生成完整测试时可以临时禁用EOS token
                    # eos_token_id=None,      # 如果想要强制生成更多内容，可以禁用EOS
                )
            
            # 解码生成的文本并提取回答
            batch_generated = []
            for i, output in enumerate(outputs):
                # 将数字序列转换回文本
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                # 提取回答部分（去掉问题部分）
                generated_answer = generated_text[len(prompts[i]):]
                batch_generated.append(generated_answer)
                generated_answers.append(generated_answer)
                reference_answers.append(batch_outputs[i])
            
            # 检查生成内容的长度 - 用于诊断生成质量
            if batch_idx == 0:  # 只在第一批显示详细信息
                token_lengths = [len(tokenizer.encode(ans)) for ans in batch_generated]
                avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
                print(f"第一批样本的平均生成长度: {avg_length:.1f} tokens")
                print(f"样本长度分布: 最短 {min(token_lengths) if token_lengths else 0}，最长 {max(token_lengths) if token_lengths else 0} tokens")
                
                # 打印第一个样本详情 - 帮助用户了解生成效果
                print(f"指令示例: {batch['instruction'][0]}")
                if batch['input'][0]:
                    print(f"输入示例: {batch['input'][0]}")
                print(f"生成回答示例(前200字符): {batch_generated[0][:200]}...")
                print(f"参考回答示例(前200字符): {batch_outputs[0][:200]}...")
                print(f"生成回答token数: {token_lengths[0] if token_lengths else 0}")
            
            # 按需清理GPU内存 - 防止内存溢出
            if (batch_idx + 1) % 5 == 0:
                del input_ids, attention_mask, outputs
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"生成回答时出错 (批次 {batch_idx}/{total_batches}): {e}")
            continue
    
    # 最终清理内存
    torch.cuda.empty_cache()
    
    # 生成完成后分析生成长度 - 评估生成质量
    if generated_answers:
        token_lengths = [len(tokenizer.encode(ans)) for ans in generated_answers]
        avg_length = sum(token_lengths) / len(token_lengths)
        print(f"\n生成完成，共生成{len(generated_answers)}个回答")
        print(f"生成回答平均长度: {avg_length:.1f} tokens")
        print(f"生成回答长度分布: 最短 {min(token_lengths)}，最长 {max(token_lengths)} tokens")
        
        # 检查生成是否被截断 - 过短的回答可能质量不佳
        short_responses = sum(1 for l in token_lengths if l < 100)
        if short_responses > 0:
            print(f"警告: 有{short_responses}个回答长度小于100个token，占比{short_responses/len(token_lengths)*100:.1f}%")
    
    return generated_answers, reference_answers

def evaluate_rouge(generated_answers, reference_answers):
    """
    计算生成回答与参考答案的ROUGE分数
    
    ROUGE(Recall-Oriented Understudy for Gisting Evaluation)是评估文本生成质量的重要指标：
    - ROUGE-1: 基于单词重叠的评估，衡量词汇覆盖度
    - ROUGE-2: 基于双词组重叠的评估，衡量语法连贯性
    - ROUGE-L: 基于最长公共子序列的评估，衡量整体结构相似性
    
    分数范围0-1，越高表示生成的回答与标准答案越相似
    
    参数:
        generated_answers: 生成的回答列表
        reference_answers: 参考答案列表
    
    返回:
        rouge_scores: 包含ROUGE-1/2/L分数的字典
    """
    rouge = Rouge()  # 初始化ROUGE评估器
    
    # 确保所有文本不为空，空文本会导致ROUGE计算错误
    valid_pairs = [(g, r) for g, r in zip(generated_answers, reference_answers) 
                 if len(g.strip()) > 0 and len(r.strip()) > 0]
    
    if not valid_pairs:
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    
    gen_valid, ref_valid = zip(*valid_pairs)
    
    try:
        # 计算ROUGE分数
        scores = rouge.get_scores(gen_valid, ref_valid, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],  # ROUGE-1 F1分数
            "rouge-2": scores["rouge-2"]["f"],  # ROUGE-2 F1分数
            "rouge-l": scores["rouge-l"]["f"]   # ROUGE-L F1分数
        }
    except Exception as e:
        print(f"计算ROUGE得分时出错: {e}")
        return {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}

def main():
    """
    主函数，执行整个评估流程
    
    评估流程：
    1. 加载模型和分词器
    2. 加载测试数据
    3. 计算困惑度（衡量模型对语言的理解能力）
    4. 生成回答（测试模型的实际应用能力）
    5. 计算ROUGE分数（衡量生成质量）
    6. 保存评估结果
    """
    print(f"加载模型: {args.model_path}")
    # 加载分词器 - 负责将文本转换为模型能理解的数字序列
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 确保有填充token，用于批量处理
    # 设置默认padding侧为右侧 - 影响批量处理的对齐方式
    tokenizer.padding_side = "left"
    
    # 处理device_map参数，使用更高效的自动分配
    # 自动分配可以让模型在多个GPU上分布，提高处理速度
    if args.device == "auto" or "," in args.device:
        device_map = "auto"
        print("使用自动设备映射进行多GPU分配")
    else:
        device_map = args.device
    
    # 加载模型，使用BF16格式提高效率和显存利用率
    # BF16是一种半精度浮点格式，可以节省显存并加速计算
    print("使用BF16格式加载模型以提高显存利用率")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # 尝试使用bfloat16以获得更好的精度和显存利用率
            device_map=device_map,       # 设备映射，支持多GPU
            trust_remote_code=True,      # 允许执行自定义代码
            max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1024**3)}GiB" 
                     for i in range(torch.cuda.device_count())}  # 最大限度利用显存，使用85%的可用显存
        )
    except Exception as e:
        print(f"使用BF16加载失败，回退到FP16: {e}")
        # 如果BF16不支持，回退到FP16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
    
    # 尝试使用torch.compile加速
    # torch.compile是PyTorch 2.0的新特性，可以显著加速模型推理
    if torch.__version__ >= "2.0.0" and torch.cuda.is_available():
        try:
            print("使用torch.compile()加速模型...")
            model = torch.compile(model)
        except Exception as e:
            print(f"模型编译失败，将使用原始模型: {e}")
    
    # 加载测试数据集
    test_dataset = SFTDataset(args.test_file, tokenizer, max_length=args.max_length)
    print(f"加载了{len(test_dataset)}个测试样例")
    
    # 设置计算设备
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device.split(",")[0] if "," in args.device else args.device)
    
    # 计算困惑度 - 增加batch_size以充分利用显存
    print("计算模型困惑度...")
    # 优化batch_size，根据可用GPU数量调整
    # 更多GPU意味着可以处理更大的批次
    adjusted_batch_size = args.batch_size * max(1, torch.cuda.device_count())
    print(f"自动调整批处理大小为: {adjusted_batch_size}")
    
    # 计算困惑度 - 这是评估语言模型最基础的指标
    perplexity = compute_perplexity(model, tokenizer, test_dataset, device, batch_size=adjusted_batch_size)
    print(f"模型困惑度: {perplexity:.4f}")
    
    # 使用全部样本或采样
    # 如果测试集很大，可以只评估一部分样本以节省时间
    if args.sample_size > 0 and args.sample_size < len(test_dataset):
        sampled_indices = np.random.choice(len(test_dataset), args.sample_size, replace=False)
        sampled_dataset = torch.utils.data.Subset(test_dataset, sampled_indices)
        print(f"从{len(test_dataset)}个样例中随机采样{args.sample_size}个进行评估")
        dataset_for_generation = sampled_dataset
    else:
        dataset_for_generation = test_dataset
        print(f"使用全部{len(test_dataset)}个样例进行评估")
    
    # 生成回答，使用调整后的batch_size
    # 这是测试模型实际应用能力的关键步骤
    generated_answers, reference_answers = generate_answers(model, tokenizer, dataset_for_generation, device, batch_size=adjusted_batch_size, max_new_tokens=args.max_new_tokens)
    
    # 计算ROUGE得分
    # ROUGE分数衡量生成的回答与标准答案的相似程度
    print("计算ROUGE得分...")
    rouge_scores = evaluate_rouge(generated_answers, reference_answers)
    print(f"ROUGE-1: {rouge_scores['rouge-1']:.4f}")  # 词汇重叠度
    print(f"ROUGE-2: {rouge_scores['rouge-2']:.4f}")  # 双词组重叠度
    print(f"ROUGE-L: {rouge_scores['rouge-l']:.4f}")  # 最长公共子序列相似度
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(args.model_path), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估指标
    metrics = {
        "perplexity": perplexity,    # 困惑度
        "rouge": rouge_scores        # ROUGE分数
    }
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 在保存前添加检查
    print(f"生成的回答数量: {len(generated_answers)}")
    print(f"参考答案数量: {len(reference_answers)}")
    if len(generated_answers) > 0:
        print(f"第一个生成的回答样例: {generated_answers[0][:100]}...")

    # 保存生成的回答与参考答案对比
    # 这些样例可以用于人工检查模型的生成质量
    with open(os.path.join(results_dir, "generation_samples.jsonl"), "w", encoding="utf-8") as f:
        for i, idx in enumerate(range(len(generated_answers))):
            if idx < len(generated_answers) and idx < len(reference_answers):
                # 获取原始样本索引（如果使用了采样）
                orig_idx = idx
                if isinstance(dataset_for_generation, torch.utils.data.Subset):
                    orig_idx = dataset_for_generation.indices[idx]
                
                # 构建结果样本
                sample = {
                    "instruction": test_dataset.instructions[orig_idx] if orig_idx < len(test_dataset.instructions) else "",
                    "input": test_dataset.inputs[orig_idx] if orig_idx < len(test_dataset.inputs) else "",
                    "reference_output": reference_answers[idx],  # 标准答案
                    "generated_output": generated_answers[idx]   # 模型生成的答案
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # 在保存生成结果前添加长度分析
    # 分析生成回答的长度分布，帮助判断模型是否生成了合适长度的回答
    token_length_analysis = {}
    if len(generated_answers) > 0:
        token_lengths = [len(tokenizer.encode(ans)) for ans in generated_answers]
        breaks = [0, 100, 200, 500, 1000, 2000, 5000]  # 长度区间
        for i in range(len(breaks)-1):
            count = sum(1 for l in token_lengths if breaks[i] <= l < breaks[i+1])
            token_length_analysis[f"{breaks[i]}-{breaks[i+1]-1}"] = {
                "count": count,
                "percentage": f"{count/len(token_lengths)*100:.1f}%"
            }
        
        # 打印分析结果
        print("\n生成回答长度分布:")
        for range_name, stats in token_length_analysis.items():
            print(f"  {range_name} tokens: {stats['count']} 个回答 ({stats['percentage']})")
        
        # 将分析结果添加到metrics中
        metrics["token_length_analysis"] = token_length_analysis
    
    print(f"评估结果已保存到 {results_dir}")

if __name__ == "__main__":
    main()
