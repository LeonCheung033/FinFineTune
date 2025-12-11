
import os
import json
import time
import random
import openai
from tqdm import tqdm
import concurrent.futures
import threading

# ==================== Configuration ====================
API_KEY = "sk-6uvkiohp89g6798e9eda"
INPUT_FILE = "/root/autodl-tmp/data/filtered_financial_news_5k.jsonl"
OUTPUT_FILE = "/root/autodl-tmp/data/sft/deepspeek_multi_turn_dataset.jsonl"
SAMPLE_COUNT = 10
MAX_WORKERS = 10
REQUEST_INTERVAL = 1
RANDOM_SEED = 57
MAX_TURNS = 3

random.seed(RANDOM_SEED)

print_lock = threading.Lock()
output_lock = threading.Lock()

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

def truncate_text(text, max_length=5000):
    if len(text) <= max_length:
        return text
    return text[:max_length]

def generate_first_question(item):
    article = item.get('Article', '')
    article_truncated = truncate_text(article)
    
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
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的金融数据分析助手，精通英文金融文章翻译和问题构建。你的任务是创建包含充分背景信息的专业金融问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7
        )
        result = response.choices[0].message.content
        
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            if "full_question" in data:
                return {"full_question": data["full_question"]}
            else:
                print("警告: 返回的JSON缺少必要字段")
                return None
        else:
            print("无法在响应中找到有效的JSON")
            return None
            
    except Exception as e:
        print(f"生成问题时出错: {e}")
        return None

def generate_first_answer(question_data):
    full_question = question_data.get("full_question", "")
    if not full_question:
        print("错误: 问题内容为空")
        return None
    
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
- 确保回答中保留1-2个可能的后续问题点，以便用户能够继续提问

严格注意：
- 思考过程必须详细，至少包含300字以上的分析
- 最终回答必须放在</think>标签之后，不得包含在思考标签内
- 思考过程和最终回答必须严格分开
- 禁止在回答中再次使用<think>或</think>标签
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        
        if "<think>" in answer and "</think>" in answer:
            return {"answer": answer}
        else:
            thinking_end = answer.find("\n\n")
            if thinking_end != -1:
                thinking = answer[:thinking_end]
                conclusion = answer[thinking_end+2:]
                formatted_answer = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
                return {"answer": formatted_answer}
            else:
                print("警告: 无法在回答中识别思考过程")
                return None
            
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return None

def extract_final_answer(answer_with_thinking):
    if "<think>" in answer_with_thinking and "</think>" in answer_with_thinking:
        end_of_thinking = answer_with_thinking.rfind("</think>") + 8
        final_answer = answer_with_thinking[end_of_thinking:].strip()
        return final_answer
    return answer_with_thinking

def generate_follow_up_question(initial_question, previous_answer, turn_num):
    prompt = f"""
你需要基于下面的初始问题和前一轮对话回答，生成一个自然且相关的后续问题。这是第{turn_num}轮对话。

初始问题:
{initial_question}

前一轮回答:
{extract_final_answer(previous_answer)}

请生成一个合理的后续问题，满足以下要求:
1. 问题必须是对前一轮回答中提到内容的深入探讨
2. 应该选择前一轮回答中的某个观点或信息点进行追问
3. 不要引入与话题无关的内容
4. 问题必须具体且专业，避免过于宽泛的提问
5. 问题应该自然，就像是真实用户看到上一轮回答后会提出的疑问

输出格式必须是有效的JSON，结构如下:
{{
  "follow_up_question": "这里是后续提问的内容"
}}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的金融数据分析助手，你的任务是根据对话上下文生成自然的后续问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.75
        )
        result = response.choices[0].message.content
        
        json_start = result.find('{')
        json_end = result.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = result[json_start:json_end]
            data = json.loads(json_str)
            
            if "follow_up_question" in data:
                return {"follow_up_question": data["follow_up_question"]}
            else:
                print("警告: 返回的JSON缺少必要字段")
                return None
        else:
            print("无法在响应中找到有效的JSON")
            return None
            
    except Exception as e:
        print(f"生成后续问题时出错: {e}")
        return None

def generate_follow_up_answer(conversation_history, current_question, turn_num):
    conversation_context = ""
    for i, (q, a) in enumerate(conversation_history):
        conversation_context += f"第{i+1}轮问题: {q}\n"
        conversation_context += f"第{i+1}轮回答: {extract_final_answer(a)}\n\n"
    
    prompt = f"""
请针对以下多轮对话中的最新问题提供专业、全面的金融分析和回答。这是第{turn_num}轮对话。

对话历史:
{conversation_context}

当前问题 (第{turn_num}轮):
{current_question}

请仅基于对话历史和问题中提供的信息进行回答，不要引入外部知识。你的回答必须包含两部分：
1. 使用<think>标签包围的详细思考过程
2. 最终的专业回答

首先，使用<think>和</think>标签包围你的详细思考过程：
<think>
在这里，你需要进行非常详细的分析，包括以下几方面：
1. 对话历史分析：理解之前的问答内容和上下文
2. 当前问题分析：分析当前问题的关键点和需求
3. 数据解读：对涉及的数字、百分比等数据进行专业解读
4. 原因探究：分析可能的原因和影响因素
5. 多角度思考：从多个维度考虑问题
6. 推理过程：清晰展示你的推理步骤和逻辑
</think>

然后，不使用任何标签，直接提供你的最终专业回答。

回答要求:
- 使用专业的金融术语和表达方式
- 提供有价值的见解和结论
- 清晰解释原因和影响
- 仅基于对话历史和当前问题中提供的信息进行回答，不要编造事实
- 回答必须使用中文
- 确保回答是对当前问题的直接回应，同时也考虑对话的连贯性

严格注意：
- 思考过程必须详细，至少包含300字以上的分析
- 最终回答必须放在</think>标签之后，不得包含在思考标签内
- 思考过程和最终回答必须严格分开
- 禁止在回答中再次使用<think>或</think>标签
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.7
        )
        answer = response.choices[0].message.content
        
        if "<think>" in answer and "</think>" in answer:
            return {"answer": answer}
        else:
            thinking_end = answer.find("\n\n")
            if thinking_end != -1:
                thinking = answer[:thinking_end]
                conclusion = answer[thinking_end+2:]
                formatted_answer = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
                return {"answer": formatted_answer}
            else:
                print("警告: 无法在回答中识别思考过程")
                return None
            
    except Exception as e:
        print(f"生成回答时出错: {e}")
        return None

def create_multi_turn_sft_data(item, index, total):
    with print_lock:
        print(f"处理文章 {index+1}/{total}")
    
    question_data = generate_first_question(item)
    if not question_data:
        with print_lock:
            print(f"文章 {index+1}/{total}: 初始问题生成失败")
        return None
    
    first_question = question_data["full_question"]
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    
    answer_data = generate_first_answer(question_data)
    if not answer_data:
        with print_lock:
            print(f"文章 {index+1}/{total}: 第一轮回答生成失败")
        return None
    
    first_answer = answer_data["answer"]
    
    actual_turns = random.randint(1, MAX_TURNS)
    conversation = [(first_question, first_answer)]
    
    current_turn = 1
    while current_turn < actual_turns:
        current_turn += 1
        time.sleep(random.uniform(0, REQUEST_INTERVAL))
        
        follow_up_question_data = generate_follow_up_question(
            first_question,
            conversation[-1][1],
            current_turn
        )
        
        if not follow_up_question_data:
            with print_lock:
                print(f"文章 {index+1}/{total}: 第{current_turn}轮问题生成失败")
            break
        
        follow_up_question = follow_up_question_data["follow_up_question"]
        time.sleep(random.uniform(0, REQUEST_INTERVAL))
        
        follow_up_answer_data = generate_follow_up_answer(
            conversation,
            follow_up_question,
            current_turn
        )
        
        if not follow_up_answer_data:
            with print_lock:
                print(f"文章 {index+1}/{total}: 第{current_turn}轮回答生成失败")
            break
        
        follow_up_answer = follow_up_answer_data["answer"]
        conversation.append((follow_up_question, follow_up_answer))
    
    messages = []
    messages.append({
        "role": "system",
        "content": "你是一个专业的金融分析师，擅长提供深入、全面的金融分析。你的回答必须包含详细的思考过程和最终结论。"
    })
    
    for question, answer in conversation:
        messages.append({
            "role": "user",
            "content": question
        })
        messages.append({
            "role": "assistant",
            "content": answer
        })
    
    sft_data = {
        "messages": messages,
        "turns": len(conversation)
    }
    
    return sft_data

def process_article(args):
    item, index, total, output_file = args
    time.sleep(random.uniform(0, REQUEST_INTERVAL))
    result = create_multi_turn_sft_data(item, index, total)
    if result:
        with output_lock:
            with open(output_file, 'a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return True
    return False

def stratified_random_sample(data_list, sample_count):
    total_count = len(data_list)
    if total_count <= sample_count:
        print(f"数据总量({total_count})小于等于需要采样的数量({sample_count})，返回全部数据")
        return data_list
    
    num_strata = min(sample_count, total_count // 10 + 1)
    stratum_size = total_count // num_strata
    
    sampled_items = []
    for i in range(num_strata):
        start_idx = i * stratum_size
        end_idx = start_idx + stratum_size if i < num_strata - 1 else total_count
        stratum_data = data_list[start_idx:end_idx]
        stratum_sample_count = max(1, int((end_idx - start_idx) / total_count * sample_count))
        
        if len(sampled_items) + stratum_sample_count > sample_count:
            stratum_sample_count = sample_count - len(sampled_items)
        
        if len(stratum_data) <= stratum_sample_count:
            sampled_items.extend(stratum_data)
        else:
            stratum_samples = random.sample(stratum_data, stratum_sample_count)
            sampled_items.extend(stratum_samples)
        
        if len(sampled_items) >= sample_count:
            break
    
    if len(sampled_items) < sample_count:
        remaining = [item for item in data_list if item not in sampled_items]
        additional = random.sample(remaining, sample_count - len(sampled_items))
        sampled_items.extend(additional)
    
    return sampled_items

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件 '{INPUT_FILE}' 不存在!")
        return
    
    file_size = os.path.getsize(INPUT_FILE)
    print(f"输入文件大小: {file_size} 字节")
    if file_size == 0:
        print("错误: 输入文件为空!")
        return
    
    print(f"使用随机种子: {RANDOM_SEED} 确保可复现性")
    print(f"最大对话轮数: {MAX_TURNS}")
    
    items = []
    line_count = 0
    error_count = 0
    valid_count = 0
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if 'Article' in item:
                        if 'Summary' in item:
                            items.append(item)
                            valid_count += 1
                        else:
                            print(f"警告: 第 {line_num} 行没有'Summary'字段")
                    elif 'article' in item:
                        if 'summary' in item:
                            item['Article'] = item['article']
                            item['Summary'] = item['summary']
                            items.append(item)
                            valid_count += 1
                        else:
                            print(f"警告: 第 {line_num} 行没有'summary'字段")
                    else:
                        print(f"警告: 第 {line_num} 行没有'Article'或'article'字段")
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"错误: 第 {line_num} 行JSON解析失败: {e}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    print(f"文件共有 {line_count} 行")
    print(f"解析错误: {error_count} 行")
    print(f"成功读取: {valid_count} 条有效记录")
    print(f"最终收集: {len(items)} 条记录")
    
    if len(items) == 0:
        return
    
    sampled_items = stratified_random_sample(items, SAMPLE_COUNT)
    print(f"分层随机采样了 {len(sampled_items)}/{len(items)} 条记录")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass
    
    print(f"开始并行处理，最大线程数: {MAX_WORKERS}")
    args_list = [(item, i, len(sampled_items), OUTPUT_FILE) for i, item in enumerate(sampled_items)]
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_article, args_list))
        success_count = sum(1 for r in results if r)
    
    print(f"处理完成，成功处理 {success_count}/{len(sampled_items)} 条记录")
    print(f"结果已保存至 {OUTPUT_FILE}")
    
    try:
        turn_distribution = {1: 0, 2: 0, 3: 0}
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    turns = data.get("turns", 0)
                    if turns in turn_distribution:
                        turn_distribution[turns] += 1
                    else:
                        turn_distribution[turns] = 1
                except:
                    pass
        print("对话轮数分布:")
        for turns, count in sorted(turn_distribution.items()):
            percentage = count/success_count*100 if success_count > 0 else 0
            print(f"{turns}轮对话: {count}条 ({percentage:.2f}%)")
    except Exception as e:
        print(f"统计对话轮数分布时出错: {e}")

if __name__ == "__main__":
    main()
