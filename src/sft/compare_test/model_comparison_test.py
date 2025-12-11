#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比测试脚本
修复了API端点问题，支持多种API格式，并自动获取正确的模型名称

这个脚本的作用：
就像组织一场"AI模型考试"，让不同的模型回答相同的问题，
然后对比它们的回答质量，帮助我们了解哪个模型表现更好。

比喻：
- 这个脚本 = 考试监考老师
- 不同的模型 = 不同的学生
- 测试数据 = 考试题目
- 模型回答 = 学生答案
- 对比结果 = 成绩单

主要功能：
1. 自动发现正在运行的模型服务
2. 向所有模型发送相同的问题
3. 收集和对比不同模型的回答
4. 生成详细的对比报告
"""

import json         # 用于处理JSON格式的数据（配置文件、测试数据等）
import random       # 用于随机选择测试样本（从大量数据中随机抽取一部分进行测试）
import asyncio      # 用于异步编程，可以同时向多个模型发送请求（提高效率）
import aiohttp      # 用于发送异步HTTP请求（与模型API通信）
import time         # 用于时间相关操作（记录时间戳、控制请求频率）
import argparse     # 用于处理命令行参数（让用户可以自定义测试参数）
from pathlib import Path  # 用于处理文件路径（跨平台兼容）
from typing import List, Dict, Any  # 用于类型提示，让代码更清晰易懂

class ModelComparisonTester:
    """
    模型对比测试器类
    
    这个类就像一个"智能考试系统"，它的主要职责是：
    1. 管理多个模型服务的连接信息（知道有哪些"考生"参加考试）
    2. 向模型发送问题并获取回答（组织考试过程）
    3. 对比不同模型的回答效果（评分和排名）
    4. 支持多种API格式（适配不同"考生"的答题方式）
    5. 自动获取正确的模型名称（确认"考生"身份）
    
    比喻：就像一个全自动的考试系统，能够：
    - 自动识别参加考试的学生
    - 同时给所有学生发放相同的试卷
    - 收集所有学生的答案
    - 对比分析不同学生的表现
    """
    
    def __init__(self, service_config_file="running_services.json"):
        """
        初始化对比测试器
        
        参数说明：
        service_config_file: 服务配置文件路径，包含运行中的模型服务信息
        
        比喻：就像考试系统启动时，先读取"考生名单"，
        了解有哪些学生要参加考试，他们的"座位号"（端口）是多少
        """
        self.services = {}  # 存储模型服务的连接信息，格式：{模型名: 连接信息}
        self.load_service_config(service_config_file)  # 加载服务配置
    
    def load_service_config(self, config_file):
        """
        加载服务配置信息
        从文件中读取当前运行的模型服务信息
        
        参数说明：
        config_file: 配置文件路径
        
        比喻：就像读取"考生名单"，了解有哪些学生要参加考试，
        每个学生坐在哪个位置（端口号），如何联系他们
        """
        # 检查配置文件是否存在
        if not Path(config_file).exists():
            print(f"错误: 服务配置文件 {config_file} 不存在")
            print("请先使用启动器启动模型服务")
            return
        
        try:
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                service_info = json.load(f)
            
            # 为每个服务创建连接信息
            # 就像为每个考生准备不同的联系方式
            for model_name, info in service_info.items():
                port = info['port']  # 获取服务端口号
                self.services[model_name] = {
                    'name': model_name,  # 模型名称（考生姓名）
                    'base_url': f"http://localhost:{port}",  # 基础URL
                    'chat_url': f"http://localhost:{port}/v1/chat/completions",  # OpenAI Chat API格式
                    'completions_url': f"http://localhost:{port}/v1/completions",  # OpenAI Completions API格式
                    'generate_url': f"http://localhost:{port}/generate",  # vLLM原生API格式
                    'models_url': f"http://localhost:{port}/v1/models",  # 获取模型列表的API
                    'port': port,  # 端口号
                    'actual_model_id': None  # 将存储从API获取的真实模型ID
                }
            
            print(f"已加载 {len(self.services)} 个模型服务:")
            for name, info in self.services.items():
                print(f"  - {name}: {info['base_url']}")
                
        except Exception as e:
            print(f"加载服务配置失败: {e}")
    
    async def get_actual_model_id(self, session, service_info):
        """
        从API获取实际的模型ID
        
        参数说明：
        session: HTTP会话对象（用于发送网络请求）
        service_info: 模型服务信息
        
        返回值：
        实际的模型ID字符串
        
        比喻：就像确认考生的真实姓名和学号，
        有时候报名时用的是昵称，但考试时需要用真实姓名
        """
        try:
            # 向模型服务发送请求，获取模型列表
            async with session.get(service_info['models_url']) as response:
                if response.status == 200:  # 请求成功
                    result = await response.json()
                    # 从返回的数据中提取模型ID
                    if 'data' in result and len(result['data']) > 0:
                        model_id = result['data'][0]['id']  # 获取第一个模型的ID
                        print(f"✓ {service_info['name']} 的实际模型ID: {model_id}")
                        return model_id
                    else:
                        print(f"✗ {service_info['name']} 没有返回模型信息")
                        return None
                else:
                    print(f"✗ {service_info['name']} 获取模型列表失败: HTTP {response.status}")
                    return None
        except Exception as e:
            print(f"✗ {service_info['name']} 获取模型ID时出错: {e}")
            return None
    
    async def test_api_endpoints(self, session, service_info):
        """
        测试不同的API端点，找到可用的格式
        
        参数说明：
        session: HTTP会话对象
        service_info: 模型服务信息
        
        返回值：
        可用的API端点信息
        
        比喻：就像测试不同的"沟通方式"，
        有些学生喜欢面对面交流，有些喜欢书面交流，
        我们需要找到每个学生最适合的沟通方式
        """
        # 首先获取实际的模型ID
        actual_model_id = await self.get_actual_model_id(session, service_info)
        if not actual_model_id:
            print(f"✗ {service_info['name']} 无法获取模型ID，跳过测试")
            return None
        
        # 更新服务信息中的实际模型ID
        service_info['actual_model_id'] = actual_model_id
        
        # 定义要测试的不同API端点
        # 就像准备不同格式的"试卷"，看看学生更适合哪种格式
        endpoints_to_test = [
            {
                'name': 'chat_completions',  # 对话格式API
                'url': service_info['chat_url'],
                'test_data': {
                    "model": actual_model_id,  # 使用实际的模型ID
                    "messages": [{"role": "user", "content": "Hello"}],  # 对话格式的测试消息
                    "max_tokens": 10  # 限制回答长度（这只是测试）
                }
            },
            {
                'name': 'completions',  # 补全格式API
                'url': service_info['completions_url'],
                'test_data': {
                    "model": actual_model_id,  # 使用实际的模型ID
                    "prompt": "Hello",  # 简单的提示词
                    "max_tokens": 10  # 限制回答长度
                }
            },
            {
                'name': 'generate',  # vLLM原生格式API
                'url': service_info['generate_url'],
                'test_data': {
                    "prompt": "Hello",  # 提示词
                    "max_tokens": 10  # 限制回答长度
                }
            }
        ]
        
        # 逐个测试每种API格式
        for endpoint in endpoints_to_test:
            try:
                # 发送测试请求
                async with session.post(endpoint['url'], json=endpoint['test_data']) as response:
                    if response.status == 200:  # 请求成功
                        print(f"✓ {service_info['name']} 支持 {endpoint['name']} API")
                        return endpoint  # 找到可用的API格式，直接返回
                    else:
                        # 请求失败，记录错误信息
                        error_text = await response.text()
                        print(f"✗ {service_info['name']} {endpoint['name']} API 返回 {response.status}: {error_text[:100]}")
            except Exception as e:
                print(f"✗ {service_info['name']} {endpoint['name']} API 测试失败: {e}")
        
        return None  # 没有找到可用的API格式
    
    def format_prompt(self, instruction, input_text=""):
        """
        格式化提示词
        将问题转换为模型能理解的格式
        
        参数说明：
        instruction: 指令或问题（比如"请翻译以下文本"）
        input_text: 额外的输入文本（比如要翻译的具体内容）
        
        返回值：
        格式化后的提示词字符串
        
        比喻：就像把考试题目整理成标准格式，
        确保所有学生都能清楚地理解题目要求
        """
        if input_text and input_text.strip():
            # 如果有额外输入，将指令和输入组合
            # 格式：<|im_start|>user\n指令\n\n输入内容<|im_end|>\n<|im_start|>assistant\n
            prompt = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # 如果只有指令，使用简化格式
            # 格式：<|im_start|>user\n指令<|im_end|>\n<|im_start|>assistant\n
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        return prompt
    
    async def ask_model_with_endpoint(self, session, service_info, endpoint_info, prompt):
        """
        使用指定的API端点向模型发送问题
        
        参数说明：
        session: HTTP会话对象
        service_info: 模型服务信息
        endpoint_info: API端点信息（使用哪种格式与模型通信）
        prompt: 格式化后的提示词
        
        返回值：
        包含回答结果的字典
        
        比喻：就像用特定的方式向学生提问，
        有些学生适合口头提问，有些适合书面提问
        """
        try:
            # 根据不同的API格式构建请求数据
            # 就像根据不同学生的特点，用不同的方式提问
            if endpoint_info['name'] == 'chat_completions':
                # 对话格式：适合聊天式的交互
                request_data = {
                    "model": service_info['actual_model_id'],  # 使用实际的模型ID
                    "messages": [{"role": "user", "content": prompt}],  # 用户消息
                    "max_tokens": 2048,  # 最大回答长度
                    "temperature": 0.7,  # 创造性参数（0-1，越高越有创意）
                    "top_p": 0.9,  # 多样性参数
                    "stop": ["<|im_end|>", "</s>"]  # 停止标记
                }
            elif endpoint_info['name'] == 'completions':
                # 补全格式：适合文本续写
                request_data = {
                    "model": service_info['actual_model_id'],  # 使用实际的模型ID
                    "prompt": prompt,  # 提示词
                    "max_tokens": 2048,  # 最大回答长度
                    "temperature": 0.7,  # 创造性参数
                    "top_p": 0.9,  # 多样性参数
                    "stop": ["<|im_end|>", "</s>"]  # 停止标记
                }
            else:  # generate格式
                # vLLM原生格式
                request_data = {
                    "prompt": prompt,  # 提示词
                    "max_tokens": 2048,  # 最大回答长度
                    "temperature": 0.7,  # 创造性参数
                    "top_p": 0.9,  # 多样性参数
                    "stop": ["<|im_end|>", "</s>"]  # 停止标记
                }
            
            # 发送请求给模型
            async with session.post(endpoint_info['url'], json=request_data) as response:
                if response.status == 200:  # 请求成功
                    result = await response.json()
                    
                    # 根据不同的API格式提取生成的文本
                    # 就像从不同格式的答卷中提取学生的答案
                    if endpoint_info['name'] == 'chat_completions':
                        generated_text = result["choices"][0]["message"]["content"].strip()
                    elif endpoint_info['name'] == 'completions':
                        generated_text = result["choices"][0]["text"].strip()
                    else:  # generate
                        generated_text = result["text"][0].strip()
                    
                    # 返回成功结果
                    return {
                        "success": True,
                        "model_name": service_info['name'],
                        "response": generated_text,
                        "api_type": endpoint_info['name'],
                        "error": None
                    }
                else:
                    # 请求失败
                    error_text = await response.text()
                    return {
                        "success": False,
                        "model_name": service_info['name'],
                        "response": "",
                        "api_type": endpoint_info['name'],
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            # 发生异常
            return {
                "success": False,
                "model_name": service_info['name'],
                "response": "",
                "api_type": endpoint_info['name'] if endpoint_info else "unknown",
                "error": str(e)
            }
    
    async def ask_model(self, session, service_info, prompt, endpoint_info=None):
        """
        向单个模型发送问题并获取回答
        这是一个异步函数，可以同时向多个模型发送请求
        
        参数说明：
        session: HTTP会话对象
        service_info: 模型服务信息
        prompt: 格式化后的提示词
        endpoint_info: 可选的API端点信息
        
        返回值：
        包含回答结果的字典
        
        比喻：就像向一个学生提问，
        如果不知道这个学生喜欢什么样的提问方式，
        就先测试一下，找到最适合的方式再正式提问
        """
        # 如果没有提供端点信息，先测试可用的端点
        if endpoint_info is None:
            endpoint_info = await self.test_api_endpoints(session, service_info)
            if endpoint_info is None:
                return {
                    "success": False,
                    "model_name": service_info['name'],
                    "response": "",
                    "api_type": "none",
                    "error": "没有找到可用的API端点"
                }
        
        # 使用找到的端点发送问题
        return await self.ask_model_with_endpoint(session, service_info, endpoint_info, prompt)
    
    def detect_file_format(self, file_path):
        """
        自动检测文件格式
        
        参数说明：
        file_path: 文件路径
        
        返回值：
        'json' 或 'jsonl'
        
        比喻：就像自动识别试卷是什么格式，
        是标准的JSON格式，还是每行一个题目的JSONL格式
        """
        file_path = str(file_path)
        # 根据文件扩展名判断
        if file_path.endswith('.jsonl'):
            return 'jsonl'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            # 如果扩展名不明确，尝试读取第一行来判断
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('{') and first_line.endswith('}'):
                        return 'jsonl'  # 每行一个JSON对象
                    elif first_line.startswith('['):
                        return 'json'   # 标准JSON数组
            except:
                pass
            return 'json'  # 默认假设是json格式
    
    def load_test_data(self, file_path):
        """
        加载测试数据
        从文件中读取要测试的问题
        
        参数说明：
        file_path: 测试数据文件路径
        
        返回值：
        测试数据列表
        
        比喻：就像从题库中加载考试题目，
        每道题目包含问题、输入数据和标准答案
        """
        # 去除路径中的多余空格
        file_path = str(file_path).strip()
        
        # 检查文件是否存在
        if not Path(file_path).exists():
            print(f"错误: 测试数据文件 {file_path} 不存在")
            print(f"当前工作目录: {Path.cwd()}")
            
            # 尝试查找类似的文件，帮助用户发现可能的文件
            parent_dir = Path(file_path).parent
            if parent_dir.exists():
                print(f"在目录 {parent_dir} 中找到的文件:")
                for f in parent_dir.glob("*.json*"):
                    print(f"  - {f}")
            
            return []
        
        try:
            # 自动检测文件格式
            file_format = self.detect_file_format(file_path)
            print(f"检测到文件格式: {file_format}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_format == 'json':
                    # JSON格式文件：整个文件是一个JSON数组
                    data = json.load(f)
                else:
                    # JSONL格式文件：每行一个JSON对象
                    data = []
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"警告: 第{line_num}行JSON格式错误: {e}")
                                continue
            
            print(f"成功加载 {len(data)} 条测试数据")
            
            # 显示第一条数据的结构，帮助用户了解数据格式
            if data:
                first_item = data[0]
                print(f"数据结构预览:")
                for key in first_item.keys():
                    # 如果值太长，只显示前100个字符
                    value = str(first_item[key])[:100] + "..." if len(str(first_item[key])) > 100 else str(first_item[key])
                    print(f"  {key}: {value}")
            
            return data
            
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            return []
    
    def select_random_samples(self, data, sample_size):
        """
        随机选择测试样本
        
        参数说明：
        data: 完整的测试数据列表
        sample_size: 要选择的样本数量
        
        返回值：
        随机选择的样本列表
        
        比喻：就像从题库中随机抽取一部分题目进行考试，
        这样可以节省时间，同时保证测试的代表性
        """
        if sample_size <= 0 or sample_size >= len(data):
            print(f"使用全部 {len(data)} 条数据进行测试")
            return data
        
        # 设置随机种子，确保结果可重现
        # 这样每次运行程序，抽取的题目都是一样的，便于对比
        random.seed(42)
        selected = random.sample(data, sample_size)
        print(f"随机选择了 {len(selected)} 条数据进行测试")
        return selected
    
    async def run_comparison_test(self, test_data, sample_size=10):
        """
        运行对比测试
        这是主要的测试函数，会向所有模型发送问题并收集回答
        
        参数说明：
        test_data: 测试数据列表
        sample_size: 要测试的样本数量
        
        返回值：
        测试结果列表
        
        比喻：就像组织一场正式考试，
        给所有学生发放相同的试卷，收集他们的答案，
        然后对比分析每个学生的表现
        """
        # 检查是否有可用的模型服务
        if len(self.services) < 1:
            print("错误: 没有可用的模型服务")
            return []
        
        if len(self.services) < 2:
            print("警告: 只有1个模型服务，无法进行对比")
        
        # 选择测试样本
        samples = self.select_random_samples(test_data, sample_size)
        if not samples:
            return []
        
        print(f"\n开始对比测试")
        print(f"参与测试的模型: {list(self.services.keys())}")
        print(f"测试样本数量: {len(samples)}")
        print("=" * 80)
        
        # 首先测试所有服务的API端点
        # 就像考试前先确认每个学生的答题方式
        print("正在测试API端点...")
        service_endpoints = {}  # 存储每个服务可用的API端点
        async with aiohttp.ClientSession() as session:
            for service_name, service_info in self.services.items():
                endpoint_info = await self.test_api_endpoints(session, service_info)
                if endpoint_info:
                    service_endpoints[service_name] = endpoint_info
                    print(f"✓ {service_name} 将使用 {endpoint_info['name']} API")
                else:
                    print(f"✗ {service_name} 没有可用的API端点")
        
        if not service_endpoints:
            print("错误: 没有找到任何可用的API端点")
            return []
        
        results = []  # 存储所有测试结果
        
        # 创建HTTP会话，用于发送请求
        async with aiohttp.ClientSession() as session:
            # 逐个处理测试样本
            for i, sample in enumerate(samples, 1):
                print(f"\n[{i}/{len(samples)}] 测试样本 {i}")
                print("-" * 60)
                
                # 提取问题信息
                # 每个样本通常包含：指令、输入、期望输出
                instruction = sample.get("instruction", "")  # 指令（比如"请翻译以下文本"）
                input_text = sample.get("input", "")         # 输入（比如要翻译的文本）
                reference_answer = sample.get("output", "") # 参考答案
                
                # 显示问题信息
                print(f"问题: {instruction}")
                if input_text:
                    # 如果输入文本太长，只显示前200个字符
                    display_input = input_text[:200] + "..." if len(input_text) > 200 else input_text
                    print(f"输入: {display_input}")
                
                # 格式化提示词
                prompt = self.format_prompt(instruction, input_text)
                
                # 创建任务列表，同时向所有可用的模型发送请求
                # 这是异步编程的核心：同时进行多个操作，提高效率
                tasks = []
                for service_name, service_info in self.services.items():
                    if service_name in service_endpoints:
                        endpoint_info = service_endpoints[service_name]
                        # 创建异步任务
                        task = self.ask_model_with_endpoint(session, service_info, endpoint_info, prompt)
                        tasks.append(task)
                
                # 等待所有模型回答完成
                # 就像等待所有学生完成答题
                print("正在获取模型回答...")
                model_responses = await asyncio.gather(*tasks)
                
                # 显示各个模型的回答
                response_dict = {}
                for response in model_responses:
                    model_name = response['model_name']
                    response_dict[model_name] = response
                    
                    print(f"\n{model_name} 的回答 ({response.get('api_type', 'unknown')} API):")
                    print("-" * 40)
                    if response['success']:
                        # 如果回答太长，只显示前500个字符
                        display_response = response['response'][:500] + "..." if len(response['response']) > 500 else response['response']
                        print(display_response)
                    else:
                        print(f"❌ 生成失败: {response['error']}")
                
                # 显示参考答案
                print(f"\n📚 参考答案:")
                print("-" * 40)
                display_reference = reference_answer[:500] + "..." if len(reference_answer) > 500 else reference_answer
                print(display_reference)
                
                # 保存这个样本的测试结果
                result = {
                    "sample_id": i,                    # 样本编号
                    "instruction": instruction,        # 指令
                    "input": input_text,              # 输入
                    "reference_answer": reference_answer,  # 参考答案
                    "model_responses": response_dict,  # 所有模型的回答
                    "timestamp": time.time()          # 时间戳
                }
                results.append(result)
                
                # 在处理下一个样本前稍作等待，避免请求过于频繁
                # 就像考试中给学生一点休息时间
                if i < len(samples):
                    await asyncio.sleep(1)
        
        return results
    
    def save_results(self, results, output_file):
        """
        保存测试结果到文件
        
        参数说明：
        results: 测试结果列表
        output_file: 输出文件路径
        
        比喻：就像把考试成绩单保存到文件中，
        方便后续查看和分析
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 使用缩进格式保存，便于阅读
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 测试结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def print_summary(self, results):
        """
        打印测试结果摘要
        
        参数说明：
        results: 测试结果列表
        
        比喻：就像打印考试成绩汇总表，
        显示每个学生的总体表现
        """
        if not results:
            return
        
        print(f"\n" + "=" * 80)
        print(f"测试结果摘要")
        print(f"=" * 80)
        print(f"总测试样本数: {len(results)}")
        
        # 统计每个模型的成功率
        model_names = list(self.services.keys())
        for model_name in model_names:
            successful_count = 0  # 成功回答的题目数
            total_count = len(results)  # 总题目数
            
            # 遍历所有测试结果，统计成功次数
            for result in results:
                model_response = result['model_responses'].get(model_name, {})
                if model_response.get('success', False):
                    successful_count += 1
            
            # 计算成功率
            success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0
            print(f"{model_name}: {successful_count}/{total_count} 成功 ({success_rate:.1f}%)")

async def main():
    """
    主函数 - 程序的入口点
    
    比喻：就像考试系统的总控制台，
    负责接收用户的指令，协调整个考试流程
    """
    # 创建命令行参数解析器
    # 让用户可以通过命令行自定义测试参数
    parser = argparse.ArgumentParser(description='模型对比测试工具')
    parser.add_argument('--data_file', 
                       type=str, 
                       default='/root/autodl-tmp/data/sft/deepspeek_sft_dataset_1_1k.jsonl',
                       help='测试数据文件路径（题库文件）')
    parser.add_argument('--sample_size', 
                       type=int, 
                       default=10,
                       help='随机选择的测试样本数量（抽取多少道题）')
    parser.add_argument('--output_file', 
                       type=str, 
                       default='comparison_results.json',
                       help='结果输出文件路径（成绩单保存位置）')
    parser.add_argument('--service_config', 
                       type=str,
                       default='running_services.json',
                       help='服务配置文件路径（考生名单文件）')
    
    args = parser.parse_args()
    
    print("🚀 启动模型对比测试")
    print("=" * 60)
    
    # 创建测试器实例
    tester = ModelComparisonTester(args.service_config)
    
    # 检查是否有可用的模型服务
    if not tester.services:
        print("❌ 没有找到可用的模型服务")
        print("请先使用启动器启动模型服务")
        return
    
    # 加载测试数据
    print(f"📁 加载测试数据: {args.data_file}")
    test_data = tester.load_test_data(args.data_file)
    if not test_data:
        return
    
    # 运行对比测试
    results = await tester.run_comparison_test(test_data, args.sample_size)
    
    if results:
        # 保存结果
        tester.save_results(results, args.output_file)
        
        # 显示摘要
        tester.print_summary(results)
        
        print(f"\n🎉 测试完成！")
        print(f"详细结果请查看: {args.output_file}")
    else:
        print("❌ 测试失败")

# 当直接运行这个脚本时，执行main函数
# 这是Python的标准写法，确保只有直接运行脚本时才执行main函数
if __name__ == "__main__":
    asyncio.run(main())

# 执行
# python model_comparison_test.py --sample_size 5