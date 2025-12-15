#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融领域奖励模型数据生成 - 答案生成器（优化版）
支持可控的并发处理，提高生成效率
"""

import json
import random
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple
import openai
from tqdm import tqdm

class AnswerGenerator:
    """答案生成器类 - 支持可控并发"""
    
    def __init__(self, config, output_dir: str = None, concurrency_level: int = None):
        """
        初始化答案生成器
        
        Args:
            config: 配置对象，包含API设置和质量等级定义
            output_dir: 输出目录路径，如果不指定则使用配置中的默认值
            concurrency_level: 总并发级别，控制同时进行的API调用数量
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path(config.OUTPUT_DIR)
        
        # 创建answers子目录用于存储批次文件
        self.answers_dir = self.output_dir / "answers"
        self.answers_dir.mkdir(parents=True, exist_ok=True)
        
        # 并发控制参数计算
        # 总并发数：同时进行的API调用数量
        total_concurrency = concurrency_level or getattr(config, 'CONCURRENCY_NUM', 15)
        
        # 答案级别的并发数：每个问题同时生成多少个答案（固定为5）
        self.answer_concurrency = 5
        
        # 问题级别的并发数：同时处理多少个问题
        # 计算公式：总并发数 / 每个问题的答案数 = 问题级别并发数
        self.question_concurrency = max(1, total_concurrency // self.answer_concurrency)
        
        # 线程安全的锁
        self.print_lock = threading.Lock()  # 用于安全打印
        self.file_lock = threading.Lock()   # 用于安全文件操作
        self.stats_lock = threading.Lock()  # 用于安全统计更新
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # 统计信息
        self.stats = {
            "total_questions": 0,
            "completed_questions": 0,
            "total_api_calls": 0,
            "successful_api_calls": 0,
            "failed_api_calls": 0,
            "start_time": None,
            "end_time": None
        }
        
        # 打印并发控制设置
        self.safe_print(f"🔧 并发控制设置:")
        self.safe_print(f"   - 总并发数: {total_concurrency}")
        self.safe_print(f"   - 问题级别并发数: {self.question_concurrency}")
        self.safe_print(f"   - 答案级别并发数: {self.answer_concurrency}")
        self.safe_print(f"   - 实际同时API调用数: {self.question_concurrency * self.answer_concurrency}")
        
    def safe_print(self, message: str):
        """线程安全的打印函数"""
        with self.print_lock:
            print(message)
            
    def update_stats(self, **kwargs):
        """线程安全的统计更新"""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    if isinstance(self.stats[key], int):
                        self.stats[key] += value
                    else:
                        self.stats[key] = value

    def generate_single_answer(self, question: str, quality_level: int, question_id: int) -> Dict[str, Any]:
        """
        为单个问题生成指定质量等级的答案
        
        Args:
            question: 问题内容
            quality_level: 质量等级 (1-5)，数字越大质量越高
            question_id: 问题ID，用于日志记录
            
        Returns:
            包含答案内容和元数据的字典
        """
        quality_info = self.config.QUALITY_LEVELS[quality_level]
            
        # 构建完整的提示词
        full_prompt = f"""你是一个金融领域的专业分析师。

{quality_info['instruction']}

问题：{question}

请按照以下格式回答：
<think>
[在这里写出你的思考过程，包括：
1. 问题分析和理解
2. 相关金融概念和理论
3. 分析步骤和逻辑推理
4. 结论推导过程]
</think>

[最终答案内容]

注意：
1. 必须使用<think>标签包裹思考过程
2. 思考过程要体现金融专业知识和分析能力
3. 最终答案要简洁明了且符合质量要求
4. 确保答案质量符合{quality_info['name']}的标准"""

        try:
            # 记录API调用
            self.update_stats(total_api_calls=1)
            
            # 调用API生成答案
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的金融分析师，擅长提供不同质量等级的专业分析。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            answer_content = response.choices[0].message.content.strip()
                
            # 验证答案格式
            if "<think>" not in answer_content or "</think>" not in answer_content:
                self.safe_print(f"⚠️  问题{question_id} 质量等级{quality_level}的答案缺少<think>标签")  
            
            # 记录成功的API调用
            self.update_stats(successful_api_calls=1)
            
            return {
                "quality_level": quality_level,
                "quality_name": quality_info["name"],
                "quality_description": quality_info["description"],
                "content": answer_content,
                "length": len(answer_content),
                "has_think_tags": "<think>" in answer_content and "</think>" in answer_content
            }
                
        except Exception as e:
            # 记录失败的API调用
            self.update_stats(failed_api_calls=1)
            self.safe_print(f"❌ 问题{question_id} 质量等级{quality_level}答案生成失败: {e}")
            
            return {
                "quality_level": quality_level,
                "quality_name": quality_info["name"],
                "quality_description": quality_info["description"],
                "content": f"生成失败: {str(e)}",
                "error": True,
                "error_message": str(e)
            }

    def generate_answers_for_question(self, question_data: Dict[str, Any], question_index: int) -> Dict[str, Any]:
        """
        为单个问题并行生成5个不同质量等级的答案
        
        Args:
            question_data: 包含问题内容的字典
            question_index: 问题在列表中的索引（用于显示进度）
            
        Returns:
            包含问题和所有答案的完整数据
        """
        question = question_data.get("question", "")
        question_id = question_data.get("id", question_index)
        
        self.safe_print(f"🔄 处理问题 {question_index + 1}/{self.stats['total_questions']} (ID: {question_id})")
        
        # 使用线程池并行生成5个不同质量的答案
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.answer_concurrency) as executor:
            # 提交所有答案生成任务
            future_to_level = {
                executor.submit(self.generate_single_answer, question, level, question_id): level 
                for level in range(1, 6)
            }
            
            # 收集结果
            answers = []
            for future in concurrent.futures.as_completed(future_to_level):
                level = future_to_level[future]
                try:
                    answer = future.result()
                    answers.append(answer)
                except Exception as e:
                    self.safe_print(f"❌ 问题{question_id} 质量等级{level}处理异常: {e}")
                    answers.append({
                        "quality_level": level,
                        "quality_name": self.config.QUALITY_LEVELS[level]["name"],
                        "quality_description": self.config.QUALITY_LEVELS[level]["description"],
                        "content": f"处理异常: {str(e)}",
                        "error": True,
                        "error_message": str(e)
                    })
            
            # 按质量等级排序
            answers.sort(key=lambda x: x["quality_level"])
        
        # 更新完成统计
        self.update_stats(completed_questions=1)
        
        result = {
            "question_id": question_id,
            "question": question,
            "answers": answers,
            "metadata": {
                "total_answers": len(answers),
                "successful_answers": len([a for a in answers if not a.get("error", False)]),
                "failed_answers": len([a for a in answers if a.get("error", False)])
            }
        }
        
        return result

    def save_intermediate_results(self, results: List[Dict[str, Any]], batch_num: int):
        """
        保存中间结果到answers子目录，防止程序崩溃时丢失数据
        
        Args:
            results: 要保存的结果列表
            batch_num: 批次编号
        """
        if not results:
            return
            
        # 保存到answers子目录
        answers_file = self.answers_dir / f"answers_batch_{batch_num}.jsonl"
        
        with self.file_lock:
            # 保存答案数据
            with open(answers_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        self.safe_print(f"💾 已保存批次 {batch_num} 的中间结果")
        self.safe_print(f"   - 答案文件: {answers_file}")

    def merge_all_batch_files(self) -> str:
        """
        合并所有批次文件为一个完整的问题-答案文件，保存到reward_data根目录
        
        Returns:
            合并后的文件路径
        """
        self.safe_print("🔄 合并所有批次文件...")
        
        # 查找answers目录下的所有批次文件
        batch_files = list(self.answers_dir.glob("answers_batch_*.jsonl"))
        batch_files.sort()  # 按文件名排序
        
        if not batch_files:
            raise FileNotFoundError(f"在 {self.answers_dir} 中未找到任何answers_batch文件")
        
        # 合并文件，保存到reward_data根目录
        merged_file = self.output_dir / "complete_qa_dataset.jsonl"
        all_data = []
        
        for batch_file in batch_files:
            self.safe_print(f"   - 读取: {batch_file.name}")
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            self.safe_print(f"⚠️  跳过无效行: {e}")
        
        # 保存合并后的文件到根目录
        with open(merged_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.safe_print(f"✅ 合并完成: {merged_file}")
        self.safe_print(f"   - 总问题数: {len(all_data)}")
        self.safe_print(f"   - 批次文件位置: {self.answers_dir}")
        self.safe_print(f"   - 合并文件位置: {merged_file}")
        
        return str(merged_file)

    def generate_preference_dataset(self, questions: List[Dict[str, Any]], max_questions: int = None) -> Dict[str, Any]:
        """
        为问题列表生成偏好数据集（主要入口函数）
        
        Args:
            questions: 问题列表
            max_questions: 最大处理问题数，用于测试或限制处理量
            
        Returns:
            包含所有结果和统计信息的字典
        """
        # 限制处理的问题数量
        if max_questions:
            questions = questions[:max_questions]
            self.safe_print(f"🔢 限制处理数量为: {max_questions}")
        
        # 初始化统计信息
        self.stats["total_questions"] = len(questions)
        self.stats["start_time"] = time.time()
        
        self.safe_print(f"🚀 开始为 {len(questions)} 个问题生成答案...")
        self.safe_print(f"📊 并发设置: 问题级别={self.question_concurrency}, 答案级别={self.answer_concurrency}")
        self.safe_print(f"📁 批次文件保存位置: {self.answers_dir}")
        
        all_results = []
        batch_size = getattr(self.config, 'BATCH_SIZE', 10)
        
        # 使用线程池并行处理多个问题
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.question_concurrency) as executor:
            # 提交所有问题处理任务
            future_to_index = {
                executor.submit(self.generate_answers_for_question, question, i): i
                for i, question in enumerate(questions)
            }
            
            # 收集结果并定期保存
            batch_results = []
            batch_num = 1
            
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    batch_results.append(result)
                    
                    # 定期保存中间结果
                    if len(batch_results) >= batch_size:
                        self.save_intermediate_results(batch_results, batch_num)
                        batch_results = []
                        batch_num += 1
                
                except Exception as e:
                    self.safe_print(f"❌ 问题 {index + 1} 处理失败: {e}")
            
            # 保存最后一批结果
            if batch_results:
                self.save_intermediate_results(batch_results, batch_num)
        
        # 记录结束时间
        self.stats["end_time"] = time.time()
        
        # 合并所有批次文件
        merged_file_path = self.merge_all_batch_files()
        
        # 生成最终统计报告
        self.generate_final_report(all_results)
        
        return {
            "results": all_results,
            "statistics": self.stats,
            "total_questions": len(questions),
            "merged_file_path": merged_file_path
        }

    def generate_final_report(self, results: List[Dict[str, Any]]):
        """生成最终的统计报告"""
        total_time = self.stats["end_time"] - self.stats["start_time"]
        
        # 计算详细统计
        successful_questions = len([r for r in results if r["metadata"]["failed_answers"] == 0])
        
        report = {
            "generation_summary": {
                "total_questions": self.stats["total_questions"],
                "completed_questions": self.stats["completed_questions"],
                "successful_questions": successful_questions,
                "total_time_seconds": total_time,
                "questions_per_minute": (self.stats["completed_questions"] / total_time) * 60 if total_time > 0 else 0
            },
            "api_statistics": {
                "total_api_calls": self.stats["total_api_calls"],
                "successful_api_calls": self.stats["successful_api_calls"],
                "failed_api_calls": self.stats["failed_api_calls"],
                "success_rate": (self.stats["successful_api_calls"] / self.stats["total_api_calls"]) * 100 if self.stats["total_api_calls"] > 0 else 0
            },
            "concurrency_settings": {
                "question_concurrency": self.question_concurrency,
                "answer_concurrency": self.answer_concurrency
            },
            "file_locations": {
                "batch_files_directory": str(self.answers_dir),
                "merged_file": str(self.output_dir / "complete_qa_dataset.jsonl")
            }
        }
        
        # 保存报告
        report_file = self.output_dir / "generation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, indent=2, ensure_ascii=False, fp=f)
        
        # 打印总结
        self.safe_print("\n" + "="*60)
        self.safe_print("🎉 答案生成完成！")
        self.safe_print("="*60)
        self.safe_print(f"📊 处理问题: {self.stats['completed_questions']}/{self.stats['total_questions']}")
        self.safe_print(f"✅ 成功问题: {successful_questions}")
        self.safe_print(f"⏱️  总耗时: {total_time:.1f}秒")
        self.safe_print(f"🚀 处理速度: {(self.stats['completed_questions'] / total_time) * 60:.1f} 问题/分钟")
        self.safe_print(f"📡 API成功率: {(self.stats['successful_api_calls'] / self.stats['total_api_calls']) * 100:.1f}%")
        self.safe_print(f"📁 批次文件目录: {self.answers_dir}")
        self.safe_print(f"📄 合并文件: {self.output_dir / 'complete_qa_dataset.jsonl'}")
        self.safe_print(f"📄 报告文件: {report_file}")
        self.safe_print("="*60)
