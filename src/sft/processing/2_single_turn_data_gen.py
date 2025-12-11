# 导入必要的库
import os                    # 操作系统相关功能，如文件路径操作
import json                  # 处理JSON格式数据
import time                  # 时间相关功能，如延迟等待
import random                # 随机数生成和随机采样
import openai                # OpenAI API客户端，这里用于调用DeepSeek API
from tqdm import tqdm        # 进度条显示库
import concurrent.futures    # 并发处理库，用于多线程
import threading             # 线程相关功能，用于线程锁

# ==================== 配置参数 ====================
API_KEY = "sk-6uvkiohp89g6798e9eda"  # DeepSeek API的密钥，用于身份验证
INPUT_FILE = "/root/autodl-tmp/data/filtered_financial_news_5k.jsonl"    # 输入文件路径：过滤后的金融新闻数据
OUTPUT_FILE = "/root/autodl-tmp/data/sft/deepspeek_sft_dataset_5k.jsonl" # 输出文件路径：生成的SFT训练数据
SAMPLE_COUNT = 5000          # 需要采样的记录数量：从输入数据中选择5000条进行处理
MAX_WORKERS = 80             # 并行处理的最大线程数：同时运行80个线程来加速处理
REQUEST_INTERVAL = 1         # 请求间隔（秒）：避免API调用过于频繁触发限制
RANDOM_SEED = 57             # 随机种子：确保每次运行程序时随机采样的结果都相同

# ==================== 随机种子设置 ====================
# 设置随机种子，确保程序的可重现性
# 随机种子的作用：让"随机"变得可预测和可重复
# 例如：每次运行程序时，random.sample()会选择相同的数据
random.seed(RANDOM_SEED)

# ==================== 线程安全锁 ====================
# 在多线程环境中，多个线程可能同时访问共享资源，导致数据混乱
# 使用锁来确保同一时间只有一个线程能访问特定资源
print_lock = threading.Lock()   # 用于保护打印输出，避免多个线程同时打印导致输出混乱
output_lock = threading.Lock()  # 用于保护文件写入，避免多个线程同时写入同一文件导致数据损坏

# ==================== API客户端初始化 ====================
# 初始化OpenAI客户端，配置为使用DeepSeek的API服务
client = openai.OpenAI(
    api_key=API_KEY,                        # 使用上面定义的API密钥
    base_url="https://api.deepseek.com"     # DeepSeek的API服务地址
)

def truncate_text(text, max_length=5000):
    """
    截断文本以满足API长度限制
    
    参数:
        text: 需要截断的文本
        max_length: 最大允许长度，默认5000字符
    
    返回:
        截断后的文本
    
    作用: API通常对输入文本长度有限制，过长的文本会导致请求失败
    """
    if len(text) <= max_length:
        return text  # 如果文本长度在限制内，直接返回原文本
    return text[:max_length]  # 如果超出限制，只返回前max_length个字符

def generate_question(item):
    """
    第一阶段：基于英文金融文章生成中文金融问题
    
    参数:
        item: 包含文章内容的字典，应该有'Article'字段
    
    返回:
        包含生成问题的字典，格式：{"full_question": "问题内容"}
        如果失败返回None
    
    这个函数的特色：
    1. 将英文金融文章转换为中文问题
    2. 确保问题包含足够的背景信息，让回答者无需看原文就能回答
    3. 避免直接引用文章，而是将信息融入问题中
    """
    # 从输入数据中提取文章内容
    article = item.get('Article', '')  # 使用get方法安全获取，如果没有'Article'字段则返回空字符串
    
    # 截断文章以满足API长度限制
    article_truncated = truncate_text(article)
    
    # 构建发送给AI的提示词（prompt）
    # 这个提示词告诉AI如何处理文章并生成问题
    prompt = f"""
请基于以下英文金融文章，创建一个详细的中文金融问题。

英文文章:
{article_truncated}

请完成以下任务:
1. 从文章中提取关键信息、数据、事实和核心观点
2. 创建一个针对这些关键信息的专业金融领域问题
3. 问题必须包含足够详细的背景信息和事实，确保仅凭问题本身就能够推导出合理的回答

输出格式必须是有效的JSON，结构如下:
{{
  "full_question": "这里是包含详细背景信息的专业金融问题"
}}

要求:
- 问题必须具体且深入，能够引导出金融专业领域的分析
- 必须包含足够丰富的事实信息，使第三方仅通过阅读问题就能回答
- 禁止出现"本文"、"文章"、"整体基调"、"情绪"等字样
- 禁止对文章本身进行评价或总结
- 直接以陈述事实的方式提供背景信息
- 问题应以客观的方式呈现数据和事实，避免主观评价
- 问题要以自然、符合实际提问习惯的方式表达
- 问题内容要特别详细，包含文章中所有能够支持回答问题的关键信息
"""

    try:
        # 调用DeepSeek API生成问题
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用DeepSeek的聊天模型
            messages=[
                # 系统消息：定义AI的角色和任务
                {"role": "system", "content": "你是一个专业的金融数据分析助手，精通英文金融文章翻译和问题构建。你的任务是创建包含充分背景信息的专业金融问题。"},
                # 用户消息：具体的任务指令
                {"role": "user", "content": prompt}
            ],
            stream=False,      # 不使用流式输出，一次性获取完整回复
            temperature=0.7    # 控制输出的随机性，0.7表示适中的创造性
        )
        # 获取AI的回复内容
        result = response.choices[0].message.content
        
        # 从AI回复中提取JSON格式的数据
        # AI的回复可能包含其他文本，需要找到JSON部分
        json_start = result.find('{')        # 找到第一个'{'的位置
        json_end = result.rfind('}') + 1     # 找到最后一个'}'的位置
        if json_start != -1 and json_end != -1:
            # 提取JSON字符串
            json_str = result[json_start:json_end]
            # 解析JSON
            data = json.loads(json_str)
            
            # 检查JSON是否包含必要的字段
            if "full_question" in data:
                return {"full_question": data["full_question"]}
            else:
                print("警告: 返回的JSON缺少必要字段")
                return None
        else:
            print("无法在响应中找到有效的JSON")
            return None
            
    except Exception as e:
        # 如果出现任何错误（网络错误、API错误、JSON解析错误等），打印错误信息
        print(f"生成问题时出错: {e}")
        return None

def generate_answer(question_data):
    """
    第二阶段：基于生成的问题创建带有思考过程的专业答案
    
    参数:
        question_data: 包含问题的字典，应该有'full_question'字段
    
    返回:
        包含答案的字典，格式：{"answer": "完整答案内容"}
        如果失败返回None
    
    这个函数的特色：
    1. 生成包含思考过程的答案，模拟人类专家的分析思路
    2. 答案分为两部分：<think>标签内的思考过程 + 最终结论
    3. 这种格式有助于训练模型学会逐步推理
    """
    # 从输入数据中提取问题内容
    full_question = question_data.get("full_question", "")
    
    # 检查问题是否为空
    if not full_question:
        print("错误: 问题内容为空")
        return None
    
    # 构建用于生成答案的提示词
    prompt = f"""
请针对以下金融问题提供专业、全面的分析和回答。

问题:
{full_question}

请仅基于问题中提供的信息进行回答，不要引入外部知识。你的回答必须包含两部分：
1. 使用<think>标签包围的详细思考过程
2. 最终的专业回答

首先，使用<think>和</think>标签包围你的详细思考过程：
<think>
在这里，你需要进行非常详细的分析，包括以下几方面：
1. 问题背景分析：分析问题中提供的关键信息和数据
2. 数据解读：对问题中的数字、百分比等数据进行专业解读
3. 原因探究：分析可能的原因和影响因素
4. 多角度思考：从多个维度考虑问题
5. 推理过程：清晰展示你的推理步骤和逻辑
</think>

然后，不使用任何标签，直接提供你的最终专业回答。

回答要求:
- 使用专业的金融术语和表达方式
- 提供有价值的见解和结论
- 清晰解释原因和影响
- 仅基于问题中提供的信息进行回答，不要编造事实
- 回答必须使用中文

严格注意：
- 思考过程必须详细，至少包含300字以上的分析
- 最终回答必须放在</think>标签之后，不得包含在思考标签内
- 思考过程和最终回答必须严格分开
- 禁止在回答中再次使用<think>或</think>标签
"""

    try:
        # 调用DeepSeek API生成答案
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                # 系统消息：定义AI作为专业金融分析师的角色
                {"role": "system", "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"},
                # 用户消息：具体的分析任务
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7
        )
        # 获取AI生成的答案
        answer = response.choices[0].message.content
        
        # 检查答案是否包含必要的思考过程标签
        if "<think>" in answer and "</think>" in answer:
            return {"answer": answer}
        else:
            # 如果AI没有按要求使用标签，尝试自动添加标签
            # 假设前面部分是思考过程，后面部分是结论
            thinking_end = answer.find("\n\n")  # 寻找段落分隔符
            if thinking_end != -1:
                thinking = answer[:thinking_end]      # 前面部分作为思考过程
                conclusion = answer[thinking_end+2:]  # 后面部分作为最终答案
                # 重新格式化答案
                formatted_answer = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
                return {"answer": formatted_answer}
            else:
                print("警告: 无法在回答中识别思考过程")
                return None
            
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return None

def create_sft_data(item, index, total):
    """
    两阶段处理：先生成问题，再生成答案，最后合并为SFT格式
    
    参数:
        item: 原始文章数据
        index: 当前处理的文章索引（用于显示进度）
        total: 总文章数量
    
    返回:
        SFT格式的训练数据字典，包含instruction、input、output三个字段
        如果失败返回None
    
    SFT (Supervised Fine-Tuning) 格式说明：
    - instruction: 任务指令，告诉模型要做什么
    - input: 具体的输入内容（这里是金融问题）
    - output: 期望的输出内容（这里是带思考过程的答案）
    """
    # 使用线程锁确保打印输出不会混乱
    with print_lock:
        print(f"处理文章 {index+1}/{total}")
    
    # 第一阶段：生成问题
    question_data = generate_question(item)
    if not question_data:
        with print_lock:
            print(f"文章 {index+1}/{total}: 问题生成失败")
        return None
    
    # 添加随机延迟，避免API请求过于集中
    # 这样可以避免触发API的频率限制
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    
    # 第二阶段：生成答案
    answer_data = generate_answer(question_data)
    if not answer_data:
        with print_lock:
            print(f"文章 {index+1}/{total}: 答案生成失败")
        return None
    
    # 合并为SFT训练数据格式
    sft_data = {
        "instruction": "以下是一个关于金融领域的问题，请提供详细的分析和见解。",  # 统一的任务指令
        "input": question_data["full_question"],    # 生成的金融问题
        "output": answer_data["answer"]             # 生成的带思考过程的答案
    }
    
    return sft_data

def process_article(args):
    """
    处理单篇文章的包装函数，专门用于并行处理
    
    参数:
        args: 包含(item, index, total, output_file)的元组
    
    返回:
        True表示处理成功，False表示处理失败
    
    这个函数的作用：
    1. 解包参数
    2. 调用create_sft_data处理文章
    3. 将结果写入文件（使用线程锁保证安全）
    """
    # 解包参数
    item, index, total, output_file = args
    
    # 添加随机延迟，避免所有线程同时发起API请求
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    
    # 处理文章生成SFT数据
    result = create_sft_data(item, index, total)
    if result:
        # 使用文件写入锁，确保多个线程不会同时写入同一文件
        with output_lock:
            # 以追加模式打开文件，将结果写入
            with open(output_file, 'a', encoding='utf-8') as out_f:
                # 将结果转换为JSON格式并写入文件，每行一个JSON对象
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return True  # 返回成功标志
    return False     # 返回失败标志

def sample_evenly(data_list, sample_count):
    """
    均匀采样指定数量的数据
    
    参数:
        data_list: 原始数据列表
        sample_count: 需要采样的数量
    
    返回:
        采样后的数据列表
    
    均匀采样的原理：
    - 将数据按固定间隔选择，确保选择的数据在原始数据中分布均匀
    - 例如：有1000条数据，要选100条，就每隔10条选一条
    """
    total_count = len(data_list)
    
    # 如果原始数据不够，返回全部数据
    if total_count <= sample_count:
        print(f"数据总量({total_count})小于等于需要采样的数量({sample_count})，返回全部数据")
        return data_list
    
    # 计算采样间隔
    step = total_count / sample_count
    # 生成采样索引列表
    indices = [int(i * step) for i in range(sample_count)]
    
    # 确保最后一个元素是最后一个索引，保证覆盖到数据的末尾
    if indices[-1] != total_count - 1:
        indices[-1] = total_count - 1
        
    # 根据索引返回采样数据
    return [data_list[i] for i in indices]

def random_sample(data_list, sample_count):
    """
    随机采样指定数量的数据，使用固定随机种子确保可复现
    
    参数:
        data_list: 原始数据列表
        sample_count: 需要采样的数量
    
    返回:
        随机采样后的数据列表
    
    随机采样的特点：
    - 每条数据被选中的概率相等
    - 由于使用了固定随机种子，每次运行结果相同
    """
    if len(data_list) <= sample_count:
        print(f"数据总量({len(data_list)})小于等于需要采样的数量({sample_count})，返回全部数据")
        return data_list
    
    # 由于在程序开始时已设置了随机种子，这里的随机采样将是可复现的
    # 即：每次运行程序，random.sample会选择相同的数据
    sampled_items = random.sample(data_list, sample_count)
    return sampled_items

def stratified_random_sample(data_list, sample_count):
    """
    分层随机采样 - 结合均匀采样的覆盖性和随机采样的随机性
    
    参数:
        data_list: 原始数据列表
        sample_count: 需要采样的数量
    
    返回:
        分层随机采样后的数据列表
    
    分层随机采样的原理：
    1. 将原始数据分成若干个区间（层）
    2. 从每个区间中随机选择一定数量的数据
    3. 这样既保证了数据的代表性（覆盖所有区间），又保持了随机性
    
    优势：
    - 比纯随机采样更有代表性
    - 比均匀采样更有随机性
    - 适合大规模数据集的采样
    """
    total_count = len(data_list)
    
    if total_count <= sample_count:
        print(f"数据总量({total_count})小于等于需要采样的数量({sample_count})，返回全部数据")
        return data_list
    
    # 计算需要划分的区间数量
    # 如果样本数量太小，至少划分为样本数量的区间
    num_strata = min(sample_count, total_count // 10 + 1)
    
    # 计算每个区间的大小
    stratum_size = total_count // num_strata
    
    sampled_items = []  # 存储采样结果
    for i in range(num_strata):
        # 计算当前区间的起止索引
        start_idx = i * stratum_size
        end_idx = start_idx + stratum_size if i < num_strata - 1 else total_count
        
        # 当前区间的数据
        stratum_data = data_list[start_idx:end_idx]
        
        # 计算当前区间需要采样的数量
        # 按比例分配采样数量
        stratum_sample_count = max(1, int((end_idx - start_idx) / total_count * sample_count))
        
        # 确保总采样数不超过要求
        if len(sampled_items) + stratum_sample_count > sample_count:
            stratum_sample_count = sample_count - len(sampled_items)
        
        # 如果区间样本数少于要求采样数，全部选择
        if len(stratum_data) <= stratum_sample_count:
            sampled_items.extend(stratum_data)
        else:
            # 从当前区间随机采样
            stratum_samples = random.sample(stratum_data, stratum_sample_count)
            sampled_items.extend(stratum_samples)
        
        # 如果已经达到所需样本数，结束循环
        if len(sampled_items) >= sample_count:
            break
    
    # 处理边界情况：最终样本数小于要求数
    if len(sampled_items) < sample_count:
        # 从未选择的数据中随机补充
        remaining = [item for item in data_list if item not in sampled_items]
        additional = random.sample(remaining, sample_count - len(sampled_items))
        sampled_items.extend(additional)
    
    return sampled_items

def main():
    """
    主函数：协调整个数据处理流程
    
    主要步骤：
    1. 检查输入文件
    2. 读取和解析数据
    3. 采样数据
    4. 并行处理生成SFT数据
    5. 统计处理结果
    """
    # 确保输出目录存在
    # os.path.dirname()获取文件路径的目录部分
    # exist_ok=True表示如果目录已存在不会报错
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件 '{INPUT_FILE}' 不存在!")
        return
    
    # 检查文件大小，确保文件不为空
    file_size = os.path.getsize(INPUT_FILE)
    print(f"输入文件大小: {file_size} 字节")
    if file_size == 0:
        print("错误: 输入文件为空!")
        return
    
    print(f"使用随机种子: {RANDOM_SEED} 确保可复现性")
    
    # ==================== 读取和解析输入文件 ====================
    # 读取输入文件，保留完整的JSON对象
    items = []          # 存储有效的数据记录
    line_count = 0      # 总行数计数器
    error_count = 0     # 错误行数计数器
    valid_count = 0     # 有效记录计数器
    
    try:
        # 逐行读取JSONL文件（每行一个JSON对象）
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # enumerate从1开始计数
                line_count += 1
                line = line.strip()  # 去除行首行尾的空白字符
                if not line:         # 跳过空行
                    continue
                
                try:
                    # 尝试解析JSON
                    item = json.loads(line)
                    
                    # 检查必要的字段是否存在
                    # 支持两种字段名格式：'Article'/'article' 和 'Summary'/'summary'
                    if 'Article' in item:
                        if 'Summary' in item:
                            items.append(item)
                            valid_count += 1
                        else:
                            print(f"警告: 第 {line_num} 行没有'Summary'字段")
                    elif 'article' in item:
                        if 'summary' in item:
                            # 标准化字段名，统一使用大写开头
                            item['Article'] = item['article']
                            item['Summary'] = item['summary']
                            items.append(item)
                            valid_count += 1
                        else:
                            print(f"警告: 第 {line_num} 行没有'summary'字段")
                    else:
                        print(f"警告: 第 {line_num} 行没有'Article'或'article'字段")
                        print(f"可用字段: {list(item.keys())}")
                        
                except json.JSONDecodeError as e:
                    # JSON解析失败
                    error_count += 1
                    print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
                    print(f"问题行内容: {line[:100]}...")  # 只显示前100个字符
                    
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    # 打印文件读取统计信息
    print(f"文件共有 {line_count} 行")
    print(f"解析错误: {error_count} 行")
    print(f"成功读取: {valid_count} 条有效记录")
    print(f"最终收集: {len(items)} 条记录")
    
    # 如果没有读取到任何有效数据，显示调试信息
    if len(items) == 0:
        print("\n尝试显示文件前5行内容进行调试:")
        try:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # 只显示前5行
                        break
                    print(f"第 {i+1} 行: {line.strip()[:200]}...")  # 显示前200个字符
                    try:
                        data = json.loads(line.strip())
                        print(f"JSON键: {list(data.keys())}")  # 显示JSON的所有键
                    except:
                        pass  # 忽略JSON解析错误
        except Exception as e:
            print(f"显示文件内容时发生错误: {e}")
        return  # 没有数据就退出程序
    
    # ==================== 数据采样 ====================
    # 使用分层随机采样替代简单随机采样
    # 这样可以获得更有代表性的数据子集
    sampled_items = stratified_random_sample(items, SAMPLE_COUNT)
    print(f"分层随机采样了 {len(sampled_items)}/{len(items)} 条记录")
    
    # 创建或清空输出文件
    # 'w'模式会清空文件内容，确保输出文件是干净的
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass  # 只是创建/清空文件，不写入任何内容
    
    # ==================== 并行处理 ====================
    print(f"开始并行处理，最大线程数: {MAX_WORKERS}")
    
    # 准备传递给每个线程的参数
    # 每个元组包含：(文章数据, 索引, 总数, 输出文件路径)
    args_list = [(item, i, len(sampled_items), OUTPUT_FILE) for i, item in enumerate(sampled_items)]
    
    # 使用线程池并行处理所有文章
    success_count = 0  # 成功处理的文章数量
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map会将args_list中的每个元素传递给process_article函数
        # 并在多个线程中并行执行
        results = list(executor.map(process_article, args_list))
        # 统计成功处理的数量（process_article返回True表示成功）
        success_count = sum(1 for r in results if r)
    
    # 打印最终处理结果
    print(f"处理完成，成功处理 {success_count}/{len(sampled_items)} 条记录")
    print(f"结果已保存至 {OUTPUT_FILE}")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()  # 运行主函数