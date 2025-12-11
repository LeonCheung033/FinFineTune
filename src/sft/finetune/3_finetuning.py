# # 创建一个新环境
# conda create -n torch25 python=3.11 -y
# conda activate torch25

# # 安装旧版本的PyTorch (2.5.1)
# pip install torch==2.5.1 torchvision==0.16.1

# # 安装其他依赖
# pip install transformers deepspeed peft


# 基础库导入
import os                   # 操作系统接口，用于文件路径操作（创建文件夹、检查文件是否存在等）
import json                 # JSON数据处理库，用于读取和解析训练数据文件
import torch                # PyTorch深度学习框架核心库，提供神经网络训练的基础功能
from torch.utils.data import Dataset  # PyTorch数据集基类，用于创建自定义数据加载器
import glob                 # 文件路径匹配工具，可以批量查找符合条件的文件

# DeepSpeed相关导入（解决PyTorch 2.6加载兼容性问题）
# DeepSpeed是微软开发的深度学习优化库，可以大幅减少显存占用
import deepspeed.runtime.fp16.loss_scaler    # DeepSpeed的半精度浮点数损失缩放器
import deepspeed.runtime.zero.config         # DeepSpeed的ZeRO内存优化配置
from torch.serialization import add_safe_globals  # PyTorch安全序列化功能，防止加载恶意代码

# Transformers库核心组件
# Transformers是Hugging Face开发的预训练模型库，包含GPT、BERT等各种AI模型
from transformers import (
    AutoTokenizer,          # 自动分词器：将文本转换为模型能理解的数字序列（token）
    AutoModelForCausalLM,   # 自动因果语言模型：专门用于文本生成任务的模型类型
    Trainer,                # 训练器：封装了完整训练流程的高级接口，简化训练代码
    TrainingArguments,      # 训练参数配置类：包含学习率、批大小、训练轮数等所有训练设置
    DataCollatorForLanguageModeling,  # 语言模型数据整理器：将多个文本样本组合成训练批次
    EarlyStoppingCallback,  # 早停回调：防止过拟合，当模型性能不再提升时自动停止训练
    TrainerCallback         # 自定义回调基类：用于在训练过程中执行自定义操作
)

# PEFT库组件（用于高效参数微调）
# PEFT (Parameter-Efficient Fine-Tuning) 是一种只训练少量参数的微调技术
# 相比传统全参数微调，PEFT可以用更少的显存和时间达到相似效果
from peft import (
    get_peft_model,         # 获取PEFT模型：将普通模型转换为参数高效微调模型
    LoraConfig,             # LoRA配置：LoRA是最流行的PEFT方法，通过低秩矩阵分解减少参数
    TaskType,               # 任务类型枚举：指定模型要执行的任务（文本生成、分类等）
    PeftModel               # PEFT模型基类：用于加载和操作参数高效微调模型
)

import deepspeed            # DeepSpeed分布式训练框架：提供内存优化和多GPU训练功能

# 定义模型路径和数据路径
# 这些路径告诉程序在哪里找到预训练模型、训练数据，以及把结果保存到哪里
MODEL_PATH = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"  # 基础预训练模型的存储路径
# 更新为SFT数据路径
# 训练数据路径列表：可以同时使用多个数据文件，程序会自动合并
TRAIN_DATA_PATHS = ["/root/autodl-tmp/data/sft/deepspeek_sft_dataset_1w.jsonl","/root/autodl-tmp/data/sft/deepspeek_sft_dataset_3k.jsonl"]
VAL_DATA_PATH = "/root/autodl-tmp/data/sft/deepspeek_sft_dataset_1_1k.jsonl"  # 验证数据集：用于评估模型训练效果
OUTPUT_DIR = "/root/autodl-tmp/finetune_output"              # 输出目录：训练过程中的模型、日志等文件保存位置

# 创建输出目录(如果不存在)
# exist_ok=True 表示如果目录已经存在也不会报错
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载分词器
# 分词器的作用是将人类可读的文本转换为模型可以处理的数字序列
# trust_remote_code=True 允许加载包含自定义代码的模型（某些模型需要特殊的分词逻辑）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 处理没有pad_token的分词器
# pad_token用于将不同长度的文本填充到相同长度，这样可以批量处理
# 如果分词器没有专门的填充标记，就使用文本结束标记代替
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为填充标记

# 自定义数据集类，用于处理SFT格式数据
# SFT (Supervised Fine-Tuning) 是监督微调的意思，使用问答对数据训练模型
class SFTDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=1024):
        """
        初始化SFT数据集
        
        这个函数的作用：
        1. 读取JSONL格式的训练数据文件（每行一个JSON对象）
        2. 将每条数据转换为模型训练所需的格式
        3. 处理文本分词和长度限制
        
        参数解释：
        data_paths: 数据文件路径，可以是单个文件路径（字符串）或多个文件路径（列表）
        tokenizer: 分词器对象，用于将文本转换为token ID
        max_length: 文本序列的最大长度，超过这个长度的文本会被截断（1024是常用值）
        """
        self.tokenizer = tokenizer      # 保存分词器，后续处理数据时使用
        self.max_length = max_length    # 保存最大长度限制
        self.examples = []              # 用于存储所有处理后的训练样本

        # 确保data_paths是列表格式，方便统一处理
        # 如果传入的是单个字符串路径，转换为包含一个元素的列表
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        # 初始化计数器，用于统计数据加载情况
        total_lines = 0      # 文件总行数
        processed_lines = 0  # 成功处理的行数

        # 逐个处理每个数据文件
        for data_path in data_paths:
            try:
                # 以UTF-8编码打开文件，确保中文字符正确显示
                with open(data_path, 'r', encoding='utf-8') as f:
                    # 逐行读取文件内容（JSONL格式：每行一个JSON对象）
                    for line in f:
                        total_lines += 1
                        try:
                            # 解析JSON格式的数据
                            item = json.loads(line)
                            # 处理SFT格式数据
                            # SFT数据通常包含三个字段：instruction（指令）、input（输入）、output（期望输出）
                            instruction = item.get("instruction", "")  # 获取任务指令，如果没有则为空字符串
                            input_text = item.get("input", "")         # 获取输入内容
                            output = item.get("output", "")            # 获取期望的输出内容

                            # 构建提示模板
                            # 这个模板告诉模型如何理解和回应用户的问题
                            if input_text:
                                # 如果有输入内容，将指令和输入组合，用换行符分隔
                                prompt = f"{instruction}\n\n{input_text}\n\n"
                            else:
                                # 如果没有输入内容，只使用指令
                                prompt = f"{instruction}\n\n"

                            # 构建完整的训练文本：提示 + 期望输出 + 结束标记
                            # 结束标记告诉模型这里是文本的结尾
                            full_text = prompt + output + tokenizer.eos_token

                            # 将处理后的数据添加到样本列表
                            self.examples.append({
                                "prompt": prompt,        # 提示部分（模型不需要学习生成这部分）
                                "full_text": full_text   # 完整文本（用于训练）
                            })
                            processed_lines += 1
                        except Exception as e:
                            # 如果某行数据解析失败，打印错误信息但继续处理其他行
                            print(f"处理第{total_lines}行时出错: {e}")
            except Exception as e:
                # 如果文件打开失败，打印错误信息
                print(f"打开文件{data_path}时出错: {e}")

        # 打印数据加载统计信息，让用户了解数据处理情况
        print(f"数据集加载 - 总行数: {total_lines}, 成功处理: {processed_lines}, 最终样本数: {len(self.examples)}")

    def __len__(self):
        """返回数据集中样本的总数，PyTorch需要这个信息来组织训练"""
        return len(self.examples)

    def __getitem__(self, idx):
        """
        获取指定索引的训练样本
        
        这个方法会被PyTorch的DataLoader自动调用，用于获取批次数据
        每次训练时，PyTorch会调用这个方法来获取一个样本
        
        参数：
        idx: 样本索引（第几个样本）
        
        返回：
        包含input_ids、attention_mask、labels的字典，这是模型训练需要的格式
        """
        # 获取指定索引的样本
        example = self.examples[idx]
        full_text = example["full_text"]  # 完整的训练文本
        prompt = example["prompt"]        # 提示部分

        # 使用分词器将文本转换为token ID
        # token ID是模型能理解的数字序列，每个数字代表一个词或字符
        encodings = self.tokenizer(
            full_text,                    # 要编码的文本
            truncation=True,              # 启用截断：如果文本太长会被截断到max_length
            max_length=self.max_length,   # 最大长度限制
            padding="max_length",         # 填充到最大长度：短文本会用pad_token填充
            return_tensors="pt"           # 返回PyTorch张量格式
        )

        # 提取编码结果
        input_ids = encodings["input_ids"][0]        # token ID序列：文本转换成的数字序列
        attention_mask = encodings["attention_mask"][0]  # 注意力掩码：告诉模型哪些是真实内容，哪些是填充

        # 创建训练标签，用于计算损失
        # 在因果语言模型中，标签就是输入序列（模型要学会预测下一个词）
        labels = input_ids.clone()

        # 计算提示部分的长度
        # 我们不希望模型学习如何生成提示，只学习如何生成回答
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_tokens)

        # 将提示部分的标签设为-100，这样在计算损失时会被忽略
        # -100是PyTorch中的特殊值，表示在计算损失时忽略这些位置
        # 这样模型只会学习生成回答部分，不会学习生成提示
        labels[:prompt_len] = -100

        # 返回训练所需的所有数据
        return {
            "input_ids": input_ids,           # 输入的token ID序列
            "attention_mask": attention_mask, # 注意力掩码（区分真实内容和填充）
            "labels": labels                  # 训练标签（告诉模型应该输出什么）
        }

# 加载训练和验证数据集
# 创建数据集实例，这些对象会自动处理数据加载和预处理
train_dataset = SFTDataset(TRAIN_DATA_PATHS, tokenizer)  # 使用列表形式的数据路径创建训练集
eval_dataset = SFTDataset(VAL_DATA_PATH, tokenizer)      # 创建验证集，用于评估模型性能


# 定义LoRA配置
# LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法
# 它不修改原始模型权重，而是添加小的"适配器"层，大幅减少需要训练的参数
lora_config = LoraConfig(
    # target_modules: 指定要应用LoRA的模块名称
    # 这些是Transformer模型中的关键组件：注意力机制和前馈网络
    # q_proj, k_proj, v_proj, o_proj 是注意力机制的四个投影层
    # gate_proj, down_proj, up_proj 是前馈网络的三个层
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型（用于文本生成，模型只能看到前面的内容）
    r=32,                   # LoRA的秩（rank）：控制适配器的大小
                           # 数值越大，适配器越大，表达能力越强，但参数也越多
                           # 32是一个在性能和效率之间平衡的常用值
    lora_alpha=64,          # LoRA的缩放因子：控制LoRA权重对最终结果的影响程度
                           # 通常设置为r的2倍，这里64 = 32 * 2
    lora_dropout=0.15,      # LoRA层的dropout率：随机丢弃15%的连接，防止过拟合
    bias="none",            # 偏置参数处理方式："none"表示不训练偏置参数
                           # 这进一步减少了需要训练的参数数量
)

# 修改训练参数配置
# TrainingArguments包含了训练过程中的所有重要设置，类似于训练的"配方"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                  # 输出目录：模型检查点、日志等文件的保存位置
    per_device_train_batch_size=3,          # 每个GPU的训练批大小：一次处理3个样本
                                            # 较小的值减少显存占用，但可能影响训练稳定性
    per_device_eval_batch_size=3,           # 每个GPU的评估批大小：评估时一次处理3个样本
    eval_accumulation_steps=3,              # 评估时的梯度累积步数：累积3个批次再计算指标
    gradient_accumulation_steps=16,          # 训练时的梯度累积步数：累积16个小批次模拟大批次
                                            # 实际批大小 = per_device_train_batch_size × gradient_accumulation_steps = 3×16=48
    num_train_epochs=70,                    # 训练轮数：整个数据集被训练70次
                                            # 这是一个相对较多的设置，适合小数据集的充分训练
    #max_steps=3000,                         # 最大训练步数（被注释掉）
                                            # 如果设置了max_steps，会覆盖num_train_epochs
    learning_rate=5e-6,                     # 初始学习率：控制参数更新的步长
                                            # 5e-6 (0.000005) 是一个较小的学习率，适合微调预训练模型
    warmup_ratio=0.1,                        # 预热阶段占比：前10%的训练步数用于学习率预热
                                            # 预热期间学习率从0逐渐增加到设定值，有助于训练稳定
    fp16=True,                              # 启用FP16混合精度训练：使用16位浮点数代替32位
                                            # 可以减少一半显存占用并加速训练，但可能略微影响精度
    eval_strategy="steps",                  # 评估策略：按训练步数进行评估（而不是按轮数）
    eval_steps=300,                         # 每300步评估一次模型性能
    do_eval=True,                           # 启用评估：在训练过程中定期评估模型性能
    save_strategy="steps",                  # 保存策略：按训练步数保存检查点
    save_steps=300,                         # 每300步保存一次模型检查点
    logging_steps=10,                       # 每10步记录一次训练日志（损失、学习率等）
    save_total_limit=10,                    # 最多保存10个检查点
                                            # 超过限制时会删除最旧的检查点以节省磁盘空间
    remove_unused_columns=False,            # 保留数据中的所有列，不自动删除未使用的列
    load_best_model_at_end=True,            # 训练结束时自动加载性能最佳的模型
    metric_for_best_model="eval_loss",      # 使用验证损失作为最佳模型的评判标准
    greater_is_better=False,                # 损失越小越好（False表示指标越小越好）
    deepspeed="ds_config.json",             # DeepSpeed配置文件路径
                                            # DeepSpeed提供内存优化和分布式训练功能
    report_to=["tensorboard"],              # 使用TensorBoard记录训练过程，可以可视化训练曲线
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),  # TensorBoard日志保存路径
    max_grad_norm=0.5,                      # 梯度裁剪阈值：防止梯度爆炸
                                            # 当梯度范数超过0.5时会被缩放到0.5
    weight_decay=0.03,                      # 权重衰减：L2正则化系数，防止过拟合
                                            # 0.03表示对权重施加适度的衰减惩罚
    lr_scheduler_type="cosine",             # 学习率调度器类型：余弦退火
                                            # 学习率会按余弦函数逐渐降低，有助于模型收敛
)

# 创建数据整理器，用于批处理和填充
# DataCollator负责将多个训练样本组合成批次，并处理长度不一致的问题
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,    # 使用前面定义的分词器
    mlm=False               # mlm=False表示使用因果语言模型模式（用于文本生成）
                           # 因果语言模型只能看到当前位置之前的内容，适合生成任务
                           # mlm=True是掩码语言模型模式（如BERT），用于理解任务
)

# 修改后的模型加载和准备流程
def create_and_prepare_model():
    """
    创建并准备用于训练的模型
    
    这个函数执行以下步骤：
    1. 从指定路径加载预训练的基础模型
    2. 配置模型以减少显存占用（启用梯度检查点）
    3. 应用LoRA配置，将普通模型转换为参数高效微调模型
    
    返回：
    配置好的PEFT模型，可以直接用于训练
    """
    # 加载模型到CPU，之后DeepSpeed会处理GPU分配
    # 不直接加载到GPU是为了让DeepSpeed更好地管理显存分配
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,                 # 预训练模型的路径
        torch_dtype=torch.float16,  # 使用FP16数据类型减少显存占用
                                   # FP16使用16位浮点数，相比FP32可以节省一半显存
        device_map=None,            # 不使用Hugging Face的自动设备映射
                                   # 让DeepSpeed来处理设备分配，避免冲突
        low_cpu_mem_usage=True,     # 启用低CPU内存使用模式
                                   # 在加载大模型时减少CPU内存占用
        trust_remote_code=True      # 允许执行模型中的自定义代码
                                   # 某些模型（如DeepSeek）可能包含自定义的模型架构代码
    )

    # 启用梯度检查点以节省内存（牺牲少量计算性能换取更多内存）
    # 梯度检查点是一种内存优化技术：不保存所有中间激活值，而是在反向传播时重新计算
    # 这样可以大幅减少显存占用，但会增加一些计算时间
    model.gradient_checkpointing_enable()

    # 应用LoRA配置，将模型转换为PEFT模型
    # 这会在原始模型的基础上添加LoRA适配器层，只有这些层的参数会被训练
    # 原始模型的参数保持冻结，大幅减少训练参数数量
    model = get_peft_model(model, lora_config)
    return model

# 创建模型
model = create_and_prepare_model()
# 打印可训练参数占比，确认LoRA设置生效
# 这会显示总参数数量、可训练参数数量和占比，帮助确认LoRA是否正确应用
model.print_trainable_parameters()

# 内存优化回调，在训练关键点主动释放内存
# 这个回调类在训练过程中的关键时刻清理显存，防止显存不足导致训练中断
class AggressiveMemoryOptimizationCallback(TrainerCallback):
    """
    激进的内存优化回调
    
    在以下时机进行显存清理：
    1. 评估开始前：清理训练阶段积累的显存碎片
    2. 评估结束后：清理评估阶段的显存占用
    3. 每次日志输出后：定期清理显存碎片
    """
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """在评估开始前执行内存优化"""
        import gc  # Python垃圾回收模块
        # 手动触发垃圾收集，释放Python中不再使用的对象
        gc.collect()
        # 清空PyTorch的CUDA缓存，释放GPU显存中的未使用内存
        torch.cuda.empty_cache()
        # 临时将模型设置为eval模式，这是评估的标准做法
        model.eval()
        # 禁用梯度计算，在评估时不需要计算梯度，可以节省显存
        model.config.use_cache = False
        return control

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        """在评估结束后执行内存清理"""
        import gc
        # 再次清理内存，确保评估阶段的显存占用被释放
        gc.collect()
        torch.cuda.empty_cache()
        # 恢复模型设置，准备继续训练
        model.train()  # 切换回训练模式
        model.config.use_cache = True  # 重新启用缓存
        return control

    def on_log(self, args, state, control, **kwargs):
        """每次日志输出后清理内存"""
        # 每次日志输出后清理显存，防止内存碎片积累
        # 这是一个轻量级的清理，不会显著影响训练速度
        torch.cuda.empty_cache()
        return control

# 追踪最佳模型并在训练结束时保存最终模型和最佳模型信息
class SaveBestAndLastModelCallback(TrainerCallback):
    """
    最佳模型跟踪和保存回调
    
    功能：
    1. 在训练过程中跟踪性能最佳的模型
    2. 在训练结束时保存最终轮次的模型
    3. 自动合并LoRA权重到基础模型，生成可直接使用的完整模型
    4. 保存模型信息到文件，方便后续查找和使用
    """
    def __init__(self):
        """初始化回调，设置跟踪变量"""
        self.best_metric = None           # 记录最佳指标值（如最低的验证损失）
        self.best_model_checkpoint = None # 记录最佳模型的检查点路径

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """在每次评估后检查是否出现了新的最佳模型"""
        # 在每次评估后检查是否是新的最佳模型
        # 只在主进程（rank 0）执行，避免多进程环境下的重复操作
        if state.is_world_process_zero and metrics is not None:  # 仅在主进程执行
            # 获取用于判断最佳模型的指标名称（在training_args中设置）
            metric_to_check = args.metric_for_best_model
            if metric_to_check in metrics:
                current_metric = metrics[metric_to_check]  # 当前评估的指标值
                # 检查是否为更好的模型
                # 判断逻辑：如果是第一次评估，或者当前指标比历史最佳更好
                if self.best_metric is None or (
                    args.greater_is_better and current_metric > self.best_metric  # 指标越大越好的情况
                ) or (
                    not args.greater_is_better and current_metric < self.best_metric  # 指标越小越好的情况（如损失）
                ):
                    # 更新最佳模型记录
                    self.best_metric = current_metric
                    self.best_model_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                    print(f"\n发现新的最佳模型! {metric_to_check}: {self.best_metric}, 保存在 {self.best_model_checkpoint}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时的处理逻辑"""
        # 训练结束时保存最终轮次的模型
        # 只在主进程执行，避免多进程重复保存
        if state.is_world_process_zero:  # 仅在主进程执行
            # 1. 保存最终轮次的LoRA模型
            # LoRA模型只包含训练的适配器权重，文件较小
            final_lora_path = os.path.join(args.output_dir, "final_lora_model")
            os.makedirs(final_lora_path, exist_ok=True)
            # 保存当前状态（最后一轮）的模型
            if "model" in kwargs:
                kwargs["model"].save_pretrained(final_lora_path)
                print(f"最终轮次的LoRA模型已保存至: {final_lora_path}")

            # 2. 合并最终轮次的完整模型
            # 将LoRA权重合并到基础模型中，生成可以直接使用的完整模型
            print("合并并保存最终轮次的完整模型...")
            # 重新加载干净的基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            # 加载LoRA权重并合并
            final_lora_model = PeftModel.from_pretrained(base_model, final_lora_path)
            final_merged_model = final_lora_model.merge_and_unload()  # 合并权重并卸载LoRA层

            # 保存合并后的完整模型
            final_complete_path = os.path.join(args.output_dir, "final_complete_model")
            os.makedirs(final_complete_path, exist_ok=True)
            final_merged_model.save_pretrained(final_complete_path)

            # 保存分词器，确保模型可以直接使用
            # 完整模型需要配套的分词器才能正常工作
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            tokenizer.save_pretrained(final_complete_path)
            print(f"最终轮次的完整模型已保存至: {final_complete_path}")

            # 3. 保存最佳模型信息到文件，方便后续查看和使用
            # 创建一个文本文件记录最佳模型的详细信息
            with open(os.path.join(args.output_dir, "best_model_info.txt"), "w") as f:
                f.write(f"最佳模型checkpoint路径: {self.best_model_checkpoint}\n")
                f.write(f"最佳{args.metric_for_best_model}: {self.best_metric}")
            print(f"最佳模型信息已保存至: {os.path.join(args.output_dir, 'best_model_info.txt')}")

# 创建自定义回调实例
# 实例化我们定义的回调类，准备在训练过程中使用
best_and_last_callback = SaveBestAndLastModelCallback()

# 创建Trainer实例
# Trainer是Hugging Face提供的高级训练接口，封装了完整的训练流程
trainer = Trainer(
    model=model,                    # 要训练的模型（已应用LoRA的PEFT模型）
    args=training_args,             # 训练参数配置（学习率、批大小等）
    train_dataset=train_dataset,    # 训练数据集
    eval_dataset=eval_dataset,      # 验证数据集（用于评估模型性能）
    data_collator=data_collator,    # 数据整理器（处理批次组装和填充）
    callbacks=[
        # 早停回调：防止过拟合的重要机制
        # 如果连续3次评估验证损失没有改善超过0.005，就自动停止训练
        # early_stopping_patience=3: 容忍3次没有改善
        # early_stopping_threshold=0.005: 改善的最小阈值
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005),
        # 内存优化回调：在关键时刻清理显存，防止显存不足
        AggressiveMemoryOptimizationCallback(),
        # 最佳模型跟踪回调：自动保存最佳模型和最终模型
        best_and_last_callback
    ]
)

# 开始训练 - 优先尝试从最新checkpoint恢复
# 支持断点续训功能，如果训练中断可以从上次保存的地方继续
last_checkpoint_dir = "/root/autodl-tmp/finetune_output/checkpoint-3000"  # 检查点路径
if os.path.exists(last_checkpoint_dir):
    # 如果找到检查点文件，从断点继续训练
    # 这样可以节省时间，不需要从头开始训练
    # 从上次中断的地方继续
    print(f"从checkpoint {last_checkpoint_dir} 恢复训练...")
    # resume_from_checkpoint: 指定要恢复的检查点路径
    # ignore_keys_for_eval: 在评估时忽略某些键，避免版本兼容性问题
    trainer.train(resume_from_checkpoint=last_checkpoint_dir, ignore_keys_for_eval=["*"])
else:
    # 如果没有找到检查点，从头开始训练
    # 这是全新训练的情况
    # 如果没有检查点，从头开始训练
    print("从头开始训练...")
    trainer.train()

# 保存最佳LoRA模型 - 利用load_best_model_at_end=True特性
# 由于设置了load_best_model_at_end=True，此时trainer.model已经是训练过程中性能最佳的模型
best_lora_model_path = os.path.join(OUTPUT_DIR, "best_lora_model")
os.makedirs(best_lora_model_path, exist_ok=True)
# 保存最佳LoRA权重，这个文件比较小，只包含训练的适配器参数
trainer.model.save_pretrained(best_lora_model_path)
print(f"最佳LoRA模型已保存至: {best_lora_model_path}")

# 保存完整的最佳模型（合并LoRA权重到基础模型）
# 完整模型可以直接使用，不需要额外的LoRA配置
print("开始合并LoRA权重与基础模型，并保存最佳完整模型...")
# 获取基础模型
# 重新加载一个干净的基础模型，用于合并LoRA权重
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 加载最佳LoRA模型与基础模型合并
# 将保存的LoRA权重加载到基础模型上
best_lora_model = PeftModel.from_pretrained(base_model, best_lora_model_path)
# 合并LoRA权重到基础模型
# merge_and_unload()会将LoRA的权重合并到原始模型参数中，并移除LoRA层
best_merged_model = best_lora_model.merge_and_unload()

# 保存完整最佳模型
# 这个模型包含了所有参数，可以像普通的预训练模型一样直接使用
best_complete_model_path = os.path.join(OUTPUT_DIR, "best_complete_model")
os.makedirs(best_complete_model_path, exist_ok=True)
best_merged_model.save_pretrained(best_complete_model_path)
# 同时保存分词器，确保模型可以完整使用
tokenizer.save_pretrained(best_complete_model_path)

print(f"最佳完整模型已保存至: {best_complete_model_path}")
