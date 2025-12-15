#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题生成模块
负责从基础文章生成金融领域的专业问题
"""
import json
import time
import random
import openai
import threading
import concurrent.futures
import os
from typing import List, Dict
from utils import save_jsonl_file

class QuestionGenerator:
    """问题生成器类，负责从文章生成问题"""
    
    def __init__(self, config, output_dir: str = None, concurrency_level: int = None):
        """
        初始化问题生成器
        
        参数:
            config: 配置对象
            output_dir: 输出目录路径，如果不指定则使用配置中的默认值
            concurrency_level: 并发级别，控制同时处理的文章数量
        """
        self.config = config
        self.output_dir = output_dir or config.output_dir
        
        # 并发控制参数 - 统一使用concurrency_num
        self.concurrency_level = concurrency_level or getattr(config, 'CONCURRENCY_NUM', 3)
        
        # 初始化OpenAI客户端（用于调用DeepSeek API）
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # 线程锁，用于多线程环境下的安全操作
        self.print_lock = threading.Lock()    # 保护打印输出
        self.output_lock = threading.Lock()   # 保护文件写入
    
    def truncate_text(self, text: str, max_length: int = 5000) -> str:
        """
        截断文本以满足API长度限制
        
        API通常对输入文本长度有限制，过长的文本会导致请求失败
        
        参数:
            text: 需要截断的文本
            max_length: 最大允许长度
        
        返回:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length]
    
    def generate_question_from_article(self, article_data: Dict) -> Dict:
        """
        从单篇文章生成问题
        
        这个函数的核心作用：
        1. 分析英文金融文章的内容
        2. 提取关键信息和数据
        3. 生成包含详细背景信息的中文问题
        4. 确保问题包含足够信息支持后续回答
        
        参数:
            article_data: 文章数据字典，包含article字段
        
        返回:
            生成的问题数据，如果失败返回None
        """
        # 提取文章内容
        article_content = article_data.get('Article', '')
        if not article_content:
            return None
        
        # 截断文章以满足API限制
        article_truncated = self.truncate_text(article_content)
        
        # 构建提示词
        # 这个提示词的设计非常重要，它决定了生成问题的质量
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
  "question": "这里是包含详细背景信息的专业金融问题"
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
- 问题应该能够支撑生成带有思考过程的详细回答
"""

        try:
            # 调用API生成问题
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的金融数据分析助手，精通英文金融文章翻译和问题构建。你的任务是创建包含充分背景信息的专业金融问题，这些问题需要能够支撑后续生成详细的分析回答。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                stream=False,
                temperature=self.config.temperature
            )
            
            # 解析API响应
            result = response.choices[0].message.content
            
            # 提取JSON格式的数据
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = result[json_start:json_end]
                data = json.loads(json_str)
                
                # 验证返回的数据
                if "question" in data and data["question"].strip():
                    return {
                        "question": data["question"],
                        "source": "generated_from_article",
                        "original_article_index": article_data.get('original_index', -1)
                    }
                else:
                    print("警告: 返回的JSON缺少question字段或内容为空")
                    return None
            else:
                print("无法在响应中找到有效的JSON")
                return None
                
        except Exception as e:
            print(f"生成问题时出错: {e}")
            return None
    
    def process_single_article(self, args):
        """
        处理单篇文章的包装函数
        
        这个函数专门用于并行处理，它：
        1. 解包参数
        2. 调用问题生成函数
        3. 将结果写入文件
        4. 返回处理状态
        
        参数:
            args: 包含(article_data, index, total, output_file)的元组
        
        返回:
            True表示处理成功，False表示失败
        """
        # 解包参数
        article_data, index, total, output_file = args
        
        # 显示处理进度（使用线程锁确保输出不混乱）
        with self.print_lock:
            print(f"处理文章 {index+1}/{total}")
        
        # 添加随机延迟，避免API请求过于集中
        time.sleep(random.uniform(0, self.config.request_interval))
        
        # 生成问题
        question_result = self.generate_question_from_article(article_data)
        
        if question_result:
            # 使用文件写入锁，确保多线程写入安全
            with self.output_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(question_result, ensure_ascii=False) + '\n')
            return True
        else:
            with self.print_lock:
                print(f"文章 {index+1}/{total}: 问题生成失败")
            return False
    
    def generate_questions_from_articles(self, articles: List[Dict]) -> List[str]:
        """
        从文章列表批量生成问题
        
        这个函数使用多线程并行处理，大大提高生成效率
        
        参数:
            articles: 文章数据列表
        
        返回:
            生成的问题列表
        """
        print(f"=== 开始从 {len(articles)} 篇文章生成问题 ===")
        print(f"📊 并发设置: 文章级别并发数={self.concurrency_level}")
        
        # 准备输出文件
        output_file = os.path.join(self.output_dir, 'generated_questions.jsonl')
        
        # 清空输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
        
        # 准备并行处理的参数
        args_list = [
            (article, i, len(articles), output_file) 
            for i, article in enumerate(articles)
        ]
        
        # 使用线程池并行处理 - 使用统一的并发控制
        success_count = 0
        print(f"开始并行处理，并发线程数: {self.concurrency_level}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency_level) as executor:
            # 并行执行所有任务
            results = list(executor.map(self.process_single_article, args_list))
            # 统计成功数量
            success_count = sum(1 for r in results if r)
        
        # 加载生成的问题
        generated_questions = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        question_data = json.loads(line.strip())
                        generated_questions.append(question_data['question'])
                    except json.JSONDecodeError:
                        continue
        
        print(f"=== 问题生成完成 ===")
        print(f"处理结果:")
        print(f"  - 成功生成: {success_count}/{len(articles)} 个问题")
        print(f"  - 实际保存: {len(generated_questions)} 个问题")
        print(f"  - 保存位置: {output_file}")
        
        return generated_questions
