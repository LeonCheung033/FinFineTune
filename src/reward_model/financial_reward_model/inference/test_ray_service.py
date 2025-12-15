#!/usr/bin/env python3

import os
import sys
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import statistics
import random

class ModelValidator:
    """
    奖励模型验证器类
    
    这个类是整个验证系统的核心，负责：
    1. 连接到Ray推理服务
    2. 加载测试数据
    3. 并发执行推理请求
    4. 统计和分析结果
    
    在整个系统中的作用：
    - 作为客户端，向Ray推理服务发送验证请求
    - 评估奖励模型的准确率和性能
    - 生成详细的验证报告
    """
    
    def __init__(self, api_url, max_workers=8):
        """
        初始化验证器
        
        参数:
            api_url: Ray推理服务的API地址
            max_workers: 最大并发线程数，用于并行发送请求
        
        在整个系统中的作用：
        - 设置验证器的基本配置
        - 初始化HTTP会话和统计数据结构
        """
        self.api_url = api_url
        self.max_workers = max_workers
        # 创建HTTP会话，复用连接以提高性能
        self.session = requests.Session()
        
        # 初始化统计数据结构
        # 这些统计数据用于跟踪验证过程中的各种指标
        self.stats = {
            'total': 0,                    # 总样本数
            'success': 0,                  # 成功请求数
            'failed': 0,                   # 失败请求数
            'correct': 0,                  # 正确预测数
            'times': [],                   # 响应时间列表
            'preference_strengths': []     # 偏好强度列表
        }
        # 线程锁，用于保护统计数据的并发访问
        self.lock = threading.Lock()
    
    def test_connection(self):
        """
        测试与API服务的连接
        
        这个方法通过发送健康检查请求来验证服务是否可用
        
        返回:
            True: 连接正常
            False: 连接失败
        
        在整个系统中的作用：
        - 在开始验证之前确保服务可用
        - 获取服务的基本信息（模型路径、设备等）
        """
        try:
            # 发送健康检查请求
            # /health端点返回服务的状态信息
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                print("API连接正常")
                print(f"模型路径: {health.get('model_path', 'Unknown')}")
                print(f"设备: {health.get('device', 'Unknown')}")
                print(f"节点ID: {health.get('node_id', 'Unknown')}")
                return True
            else:
                print(f"API响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def load_test_data(self, max_samples=None):
        """
        加载测试数据
        
        这个方法从固定路径加载preference数据集
        
        参数:
            max_samples: 最大样本数量，如果指定则随机采样
        
        返回:
            测试数据列表，每个元素包含question、chosen、rejected字段
        
        在整个系统中的作用：
        - 提供验证所需的标准测试数据
        - 支持采样功能，可以控制验证规模
        """
        # 固定的测试数据路径
        # 这个路径指向预处理好的preference数据集
        test_file = "/shared/reward_model_data/reward_data/test/preference_dataset.jsonl"
        
        # 检查测试文件是否存在
        if not os.path.exists(test_file):
            print(f"测试数据文件不存在: {test_file}")
            return []
        
        print(f"加载测试数据: {test_file}")
        
        data = []
        # 逐行读取JSONL文件
        # 每行是一个JSON对象，包含一个测试样本
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 验证数据格式，确保包含必需的字段
                if all(k in item for k in ['question', 'chosen', 'rejected']):
                    data.append(item)
        
        # 如果指定了最大样本数，进行随机采样
        if max_samples and len(data) > max_samples:
            data = random.sample(data, max_samples)
            print(f"随机采样 {max_samples} 条数据")
        
        print(f"测试数据加载完成: {len(data)} 条")
        return data
    
    def single_test(self, item):
        """
        执行单个测试样本的推理
        
        这个方法向API发送单个推理请求并处理响应
        
        参数:
            item: 包含question、chosen、rejected的测试样本
        
        返回:
            包含推理结果和统计信息的字典
        
        在整个系统中的作用：
        - 这是验证的核心方法，每个测试样本都会调用这个方法
        - 负责发送HTTP请求、解析响应、计算准确性
        """
        start_time = time.time()
        
        try:
            # 发送POST请求到推理服务
            # 请求体包含question、chosen、rejected三个字段
            response = self.session.post(
                f"{self.api_url}/",
                json={
                    "question": item["question"],
                    "chosen": item["chosen"], 
                    "rejected": item["rejected"]
                },
                timeout=30  # 30秒超时
            )
            
            # 计算响应时间
            elapsed = time.time() - start_time
            
            # 检查HTTP状态码
            if response.status_code == 200:
                result = response.json()
                
                # 提取推理结果
                chosen_score = result.get("chosen_score", 0)
                rejected_score = result.get("rejected_score", 0)
                # 判断预测是否正确
                # 正确的预测应该是chosen_score > rejected_score
                is_correct = chosen_score > rejected_score
                # 计算偏好强度（分数差的绝对值）
                preference_strength = abs(chosen_score - rejected_score)
                
                # 线程安全地更新统计数据
                with self.lock:
                    self.stats['success'] += 1
                    self.stats['times'].append(elapsed)
                    if is_correct:
                        self.stats['correct'] += 1
                    self.stats['preference_strengths'].append(preference_strength)
                
                return {
                    'success': True,
                    'correct': is_correct,
                    'time': elapsed,
                    'chosen_score': chosen_score,
                    'rejected_score': rejected_score,
                    'preference_strength': preference_strength
                }
            else:
                # HTTP错误情况
                with self.lock:
                    self.stats['failed'] += 1
                return {
                    'success': False, 
                    'error': f"HTTP {response.status_code}",
                    'time': elapsed
                }
                
        except Exception as e:
            # 异常情况（网络错误、超时等）
            elapsed = time.time() - start_time
            with self.lock:
                self.stats['failed'] += 1
            return {
                'success': False,
                'error': str(e)[:100],  # 限制错误信息长度
                'time': elapsed
            }
    
    def run_validation(self, test_data):
        """
        运行完整的验证流程
        
        这个方法是验证的主要入口，负责并发执行所有测试样本
        
        参数:
            test_data: 测试数据列表
        
        返回:
            包含完整验证结果的字典
        
        在整个系统中的作用：
        - 协调整个验证过程
        - 管理并发执行
        - 生成最终的验证报告
        """
        print(f"开始验证，共 {len(test_data)} 条数据")
        print(f"并发数: {self.max_workers}")
        
        self.stats['total'] = len(test_data)
        start_time = time.time()
        
        results = []
        
        # 使用线程池并发执行推理请求
        # 这可以显著提高验证速度
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务到线程池
            future_to_item = {
                executor.submit(self.single_test, item): item 
                for item in test_data
            }
            
            # 使用进度条显示验证进度
            with tqdm(total=len(test_data), desc="验证进度") as pbar:
                # 等待任务完成并收集结果
                for future in as_completed(future_to_item):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # 实时更新进度条显示的统计信息
                    if self.stats['success'] > 0:
                        accuracy = self.stats['correct'] / self.stats['success'] * 100
                        pbar.set_postfix({
                            'accuracy': f"{accuracy:.1f}%",
                            'success': self.stats['success'],
                            'failed': self.stats['failed']
                        })
        
        total_time = time.time() - start_time
        # 打印详细的验证报告
        self.print_report(results, total_time)
        
        # 返回汇总结果
        return {
            'total_samples': len(test_data),
            'successful_requests': self.stats['success'],
            'failed_requests': self.stats['failed'],
            'accuracy': self.stats['correct'] / max(self.stats['success'], 1) * 100,
            'avg_response_time': statistics.mean(self.stats['times']) if self.stats['times'] else 0,
            'avg_preference_strength': statistics.mean(self.stats['preference_strengths']) if self.stats['preference_strengths'] else 0,
            'total_time': total_time
        }
    
    def print_report(self, results, total_time):
        """
        打印详细的验证报告
        
        这个方法生成并显示完整的验证统计报告
        
        参数:
            results: 所有测试结果的列表
            total_time: 总验证时间
        
        在整个系统中的作用：
        - 提供人类可读的验证结果
        - 显示关键的性能和准确性指标
        """
        print("\n验证报告")
        print("="*60)
        
        # 基本统计信息
        print(f"总样本数: {self.stats['total']}")
        print(f"成功请求: {self.stats['success']}")
        print(f"失败请求: {self.stats['failed']}")
        print(f"成功率: {self.stats['success']/self.stats['total']*100:.1f}%")
        
        # 如果有成功的请求，显示详细统计
        if self.stats['success'] > 0:
            # 计算模型准确率
            # 这是最重要的指标，表示模型正确判断chosen > rejected的比例
            accuracy = self.stats['correct'] / self.stats['success'] * 100
            print(f"模型准确率: {accuracy:.2f}%")
            
            # 性能统计
            times = self.stats['times']
            print(f"\n性能统计:")
            print(f"  平均响应时间: {statistics.mean(times):.3f}s")
            print(f"  最快响应时间: {min(times):.3f}s")
            print(f"  最慢响应时间: {max(times):.3f}s")
            print(f"  总耗时: {total_time:.2f}s")
            print(f"  吞吐量: {self.stats['success']/total_time:.1f} 请求/秒")
            
            # 偏好强度统计
            # 偏好强度反映模型对判断的置信度
            if self.stats['preference_strengths']:
                strengths = self.stats['preference_strengths']
                print(f"\n偏好强度统计:")
                print(f"  平均偏好强度: {statistics.mean(strengths):.4f}")
                print(f"  最大偏好强度: {max(strengths):.4f}")
                print(f"  最小偏好强度: {min(strengths):.4f}")
        
        print("="*60)

def main():
    """
    主函数，程序的入口点
    
    这个函数负责：
    1. 解析命令行参数
    2. 创建验证器实例
    3. 执行验证流程
    4. 保存结果
    
    在整个系统中的作用：
    - 这是验证程序的启动入口
    - 协调整个验证流程的执行
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="奖励模型验证")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000", 
                       help="API服务地址")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="最大测试样本数")
    parser.add_argument("--max_workers", type=int, default=8, 
                       help="并发线程数")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建验证器实例
    validator = ModelValidator(args.api_url, args.max_workers)
    
    # 测试API连接
    if not validator.test_connection():
        print("无法连接到API服务")
        print("请确保Ray服务已启动")
        return
    
    # 加载测试数据
    test_data = validator.load_test_data(args.max_samples)
    if not test_data:
        print("无法加载测试数据")
        return
    
    # 执行验证
    results = validator.run_validation(test_data)
    
    # 保存验证结果到文件
    result_file = f"validation_results_{int(time.time())}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"验证结果已保存到: {result_file}")

# 如果这个文件被直接运行（而不是被导入），则执行主函数
if __name__ == "__main__":
    main()