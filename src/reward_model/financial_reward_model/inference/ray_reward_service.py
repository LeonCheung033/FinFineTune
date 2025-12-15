#!/usr/bin/env python3

import os
import sys
import time
import signal
import argparse
import logging

import ray
from ray import serve
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer

# 配置日志系统，设置日志级别和格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 使用Ray Serve装饰器定义服务部署
# 这个装饰器告诉Ray如何部署这个服务类
# num_gpus=1: 每个服务实例使用1个GPU
# num_cpus=2: 每个服务实例使用2个CPU核心
@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 2
    }
)
class RewardService:
    """
    奖励模型推理服务类
    
    这个类是整个推理服务的核心，负责：
    1. 加载奖励模型和tokenizer
    2. 处理推理请求
    3. 返回chosen和rejected回答的分数
    
    在整个系统中的作用：
    - 接收HTTP请求，包含question、chosen、rejected三个字段
    - 使用训练好的奖励模型对chosen和rejected回答进行评分
    - 返回分数结果，用于判断哪个回答更好
    """
    
    def __init__(self, model_path: str):
        """
        初始化奖励模型服务
        
        这个方法在服务启动时被调用，负责加载模型和tokenizer
        
        参数:
            model_path: 训练好的奖励模型的路径
        
        在整个系统中的作用：
        - 这是服务的初始化入口，只在服务启动时执行一次
        - 加载的模型和tokenizer会被后续的推理请求重复使用
        """
        self.model_path = model_path
        # 检测是否有可用的GPU，优先使用GPU进行推理
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 第一步：加载tokenizer（文本分词器）
        # tokenizer的作用是将文本转换为模型可以理解的数字序列
        try:
            logger.info("开始加载tokenizer...")
            logger.info(f"模型路径: {model_path}")
            
            # 检查模型路径是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            # 检查tokenizer相关文件是否存在
            # 这些文件是tokenizer正常工作所必需的
            tokenizer_files = [
                "tokenizer.json",          # 主要的tokenizer配置
                "tokenizer_config.json",   # tokenizer配置文件
                "vocab.json",              # 词汇表文件
                "merges.txt",              # BPE合并规则（如果使用BPE）
                "special_tokens_map.json"  # 特殊token映射
            ]
            
            existing_files = []
            for file in tokenizer_files:
                file_path = os.path.join(model_path, file)
                if os.path.exists(file_path):
                    existing_files.append(file)
            
            logger.info(f"找到的tokenizer文件: {existing_files}")
            
            # 尝试多种方式加载tokenizer
            # 这是为了提高兼容性，如果一种方法失败，会尝试其他方法
            tokenizer = None
            
            # 方法1: 使用AutoTokenizer自动识别tokenizer类型
            try:
                logger.info("尝试使用AutoTokenizer加载...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    use_fast=False,        # 不使用快速tokenizer，提高兼容性
                    local_files_only=True, # 只使用本地文件，不从网络下载
                    trust_remote_code=False # 不信任远程代码，提高安全性
                )
                logger.info(f"AutoTokenizer加载成功，类型: {type(tokenizer)}")
                
                # 验证tokenizer不是布尔值
                # 这是为了防止某些异常情况下tokenizer被错误地设置为布尔值
                if isinstance(tokenizer, bool):
                    logger.error(f"AutoTokenizer返回了布尔值: {tokenizer}")
                    tokenizer = None
                    
            except Exception as e:
                logger.warning(f"AutoTokenizer加载失败: {e}")
                tokenizer = None
            
            # 方法2: 如果AutoTokenizer失败，尝试直接使用LlamaTokenizer
            if tokenizer is None:
                try:
                    logger.info("尝试使用LlamaTokenizer加载...")
                    tokenizer = LlamaTokenizer.from_pretrained(
                        model_path,
                        use_fast=False,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                    logger.info(f"LlamaTokenizer加载成功，类型: {type(tokenizer)}")
                    
                    if isinstance(tokenizer, bool):
                        logger.error(f"LlamaTokenizer返回了布尔值: {tokenizer}")
                        tokenizer = None
                        
                except Exception as e:
                    logger.warning(f"LlamaTokenizer加载失败: {e}")
                    tokenizer = None
            
            # 方法3: 尝试从基础模型路径加载tokenizer
            # 这是最后的备选方案，使用原始的基础模型的tokenizer
            if tokenizer is None:
                try:
                    logger.info("尝试从基础模型路径加载tokenizer...")
                    base_model_path = "/shared/Skywork-Reward-Llama-3.1-8B"
                    if os.path.exists(base_model_path):
                        tokenizer = AutoTokenizer.from_pretrained(
                            base_model_path,
                            use_fast=False,
                            local_files_only=True,
                            trust_remote_code=False
                        )
                        logger.info(f"从基础模型加载tokenizer成功，类型: {type(tokenizer)}")
                        
                        if isinstance(tokenizer, bool):
                            logger.error(f"从基础模型加载的tokenizer是布尔值: {tokenizer}")
                            tokenizer = None
                    else:
                        logger.warning(f"基础模型路径不存在: {base_model_path}")
                        
                except Exception as e:
                    logger.warning(f"从基础模型加载tokenizer失败: {e}")
                    tokenizer = None
            
            # 如果所有方法都失败了，抛出错误
            if tokenizer is None:
                raise RuntimeError("所有tokenizer加载方法都失败了")
            
            # 验证tokenizer对象是否正常
            logger.info(f"Tokenizer最终类型: {type(tokenizer)}")
            logger.info(f"Tokenizer是否可调用: {callable(tokenizer)}")
            
            # 检查tokenizer是否有pad_token属性
            # pad_token用于将不同长度的文本序列填充到相同长度
            if not hasattr(tokenizer, 'pad_token'):
                raise RuntimeError(f"Tokenizer对象没有pad_token属性: {type(tokenizer)}")
            
            # 设置pad_token
            # pad_token是用于填充序列的特殊token，确保批次中所有序列长度相同
            logger.info(f"当前pad_token: {getattr(tokenizer, 'pad_token', 'None')}")
            
            if tokenizer.pad_token is None:
                # 如果没有pad_token，尝试使用eos_token作为pad_token
                if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info(f"设置pad_token为eos_token: {tokenizer.pad_token}")
                else:
                    # 如果没有eos_token，手动添加一个pad_token
                    tokenizer.add_special_tokens({'pad_token': '<pad>'})
                    logger.info(f"添加新的pad_token: {tokenizer.pad_token}")
            
            logger.info(f"最终pad_token: {tokenizer.pad_token}")
            
            # 只有在一切正常的情况下才将tokenizer赋值给实例变量
            self.tokenizer = tokenizer
            logger.info("Tokenizer初始化完成")
            
        except Exception as e:
            logger.error(f"tokenizer加载失败: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"Tokenizer加载失败: {e}")
        
        # 第二步：加载奖励模型
        # 奖励模型是一个分类模型，用于对文本回答进行评分
        try:
            logger.info("开始加载模型...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,  # 使用bfloat16精度，节省显存
                device_map="auto",           # 自动分配设备
                trust_remote_code=False,     # 不信任远程代码
                local_files_only=True        # 只使用本地文件
            )
            # 设置模型为评估模式，关闭dropout等训练时的随机性
            self.model.eval()
            logger.info(f"模型加载完成，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"模型加载失败: {e}")
    
    def format_conversation(self, question: str, answer: str) -> str:
        """
        格式化对话文本
        
        这个方法将用户问题和助手回答格式化为LLaMA-3的对话格式
        
        参数:
            question: 用户提出的问题
            answer: 助手的回答
        
        返回:
            格式化后的对话文本
        
        在整个系统中的作用：
        - 将输入的问题和回答转换为模型训练时使用的格式
        - 确保推理时的格式与训练时一致，这对模型性能很重要
        """
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
    
    async def __call__(self, request):
        """
        处理推理请求的主要方法
        
        这个方法是整个服务的核心，处理所有的HTTP请求
        
        参数:
            request: HTTP请求对象，包含question、chosen、rejected字段
        
        返回:
            包含分数结果的字典
        
        在整个系统中的作用：
        - 这是服务的主要入口点，所有的推理请求都会调用这个方法
        - 接收用户的问题和两个候选回答，返回哪个回答更好的评分
        """
        try:
            # 处理健康检查请求
            # 健康检查用于监控服务是否正常运行
            if hasattr(request, 'url') and request.url and "health" in str(request.url):
                return {
                    "status": "healthy",
                    "model_path": self.model_path,
                    "device": str(self.device),
                    "tokenizer_type": str(type(self.tokenizer)),
                    "node_id": ray.get_runtime_context().get_node_id()
                }
            
            # 解析请求数据
            # 支持两种请求格式：HTTP请求和直接的字典数据
            if hasattr(request, 'json'):
                data = await request.json()
            else:
                data = request
            
            # 提取请求中的三个关键字段
            question = data["question"]    # 用户问题
            chosen = data["chosen"]        # 人类偏好的回答（好回答）
            rejected = data["rejected"]    # 人类不偏好的回答（差回答）
            
            # 将问题和回答格式化为模型输入格式
            chosen_text = self.format_conversation(question, chosen)
            rejected_text = self.format_conversation(question, rejected)
            
            logger.info(f"开始推理 - tokenizer类型: {type(self.tokenizer)}")
            
            # 验证tokenizer状态
            # 这些检查确保tokenizer已经正确初始化
            if self.tokenizer is None:
                error_msg = "tokenizer未正确初始化"
                logger.error(error_msg)
                return {"error": error_msg}
            
            if not callable(self.tokenizer):
                error_msg = f"tokenizer不可调用，类型: {type(self.tokenizer)}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            start_time = time.time()
            
            # 执行模型推理
            # torch.no_grad()用于关闭梯度计算，节省内存和计算资源
            with torch.no_grad():
                # 对chosen回答进行tokenization（文本转数字）
                chosen_inputs = self.tokenizer(
                    chosen_text, 
                    return_tensors="pt",    # 返回PyTorch张量
                    max_length=2048,        # 最大序列长度
                    truncation=True,        # 如果超长则截断
                    padding=True            # 填充到指定长度
                ).to(self.device)           # 移动到GPU设备
                
                # 对rejected回答进行tokenization
                rejected_inputs = self.tokenizer(
                    rejected_text, 
                    return_tensors="pt", 
                    max_length=2048, 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                # 使用模型进行推理，获取chosen回答的分数
                chosen_outputs = self.model(**chosen_inputs)
                # 使用模型进行推理，获取rejected回答的分数
                rejected_outputs = self.model(**rejected_inputs)
                
                # 记录模型输出的形状和内容，用于调试
                logger.info(f"chosen_outputs.logits shape: {chosen_outputs.logits.shape}")
                logger.info(f"chosen_outputs.logits: {chosen_outputs.logits}")
                
                # 提取分数
                # [0, 0]表示取第一个样本的第一个输出值
                chosen_score = float(chosen_outputs.logits[0, 0].cpu().item())
                rejected_score = float(rejected_outputs.logits[0, 0].cpu().item())
            
            # 计算处理时间
            processing_time = time.time() - start_time
            # 计算偏好强度（chosen分数与rejected分数的差值）
            preference_strength = chosen_score - rejected_score
            # 根据分数高低判断预测结果
            prediction = "chosen" if chosen_score > rejected_score else "rejected"
            
            logger.info(f"推理完成 - chosen: {chosen_score:.4f}, rejected: {rejected_score:.4f}")
            
            # 返回推理结果
            return {
                "chosen_score": chosen_score,              # chosen回答的分数
                "rejected_score": rejected_score,          # rejected回答的分数
                "preference_strength": preference_strength, # 偏好强度
                "prediction": prediction,                  # 预测结果
                "processing_time": processing_time,        # 处理时间
                "node_id": ray.get_runtime_context().get_node_id()  # 处理节点ID
            }
            
        except Exception as e:
            logger.error(f"推理错误: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return {"error": str(e)}

def deploy_service(args):
    """
    部署Ray Serve服务
    
    这个函数负责启动和配置Ray Serve服务
    
    参数:
        args: 命令行参数对象，包含模型路径、副本数量、端口等配置
    
    返回:
        True: 部署成功
        False: 部署失败
    
    在整个系统中的作用：
    - 这是服务部署的核心函数，负责启动Ray集群和部署推理服务
    - 配置服务的副本数量、资源分配等参数
    """
    # 检查Ray是否已经初始化，如果没有则连接到现有集群
    if not ray.is_initialized():
        try:
            # 连接到已经启动的Ray集群
            ray.init(address="auto", ignore_reinit_error=True)
            logger.info("连接到Ray集群成功")
        except Exception as e:
            logger.error(f"连接Ray集群失败: {e}")
            return False
    
    # 获取集群资源信息
    cluster_resources = ray.cluster_resources()
    total_gpus = int(cluster_resources.get("GPU", 0))
    # 确保副本数量不超过可用GPU数量
    num_replicas = min(args.num_replicas, total_gpus)
    
    logger.info(f"部署配置: {num_replicas} 个副本")
    
    try:
        # 启动Ray Serve服务
        serve.start(
            detached=True,  # 后台运行
            http_options={
                "host": "0.0.0.0",  # 监听所有网络接口
                "port": args.port   # 指定端口
            }
        )
        
        # 配置服务部署选项
        app = RewardService.options(
            num_replicas=num_replicas,  # 副本数量
            ray_actor_options={
                "num_gpus": 1,  # 每个副本使用1个GPU
                "num_cpus": 2   # 每个副本使用2个CPU核心
            }
        ).bind(args.model_path)  # 绑定模型路径参数
        
        # 运行服务，设置路由前缀
        serve.run(app, route_prefix="/")
        
        logger.info(f"服务部署成功: http://0.0.0.0:{args.port}")
        return True
    except Exception as e:
        logger.error(f"服务部署失败: {e}")
        return False

def main():
    """
    主函数，程序的入口点
    
    这个函数负责：
    1. 解析命令行参数
    2. 启动服务
    3. 处理信号和保持服务运行
    
    在整个系统中的作用：
    - 这是整个服务的启动入口
    - 处理命令行参数，启动服务，并保持服务持续运行
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="奖励模型推理服务")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--num_replicas", type=int, default=2, help="副本数量")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    logger.info("启动奖励模型服务")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"副本数量: {args.num_replicas}")
    logger.info(f"服务端口: {args.port}")
    
    # 尝试部署服务
    if deploy_service(args):
        logger.info("服务启动成功，保持运行状态")
        
        # 定义信号处理函数，用于优雅地关闭服务
        def signal_handler(sig, frame):
            logger.info("收到停止信号，正在关闭服务")
            serve.shutdown()
            sys.exit(0)
        
        # 注册信号处理器，处理Ctrl+C和终止信号
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 保持服务运行，每60秒检查一次
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("服务被用户中断")
            serve.shutdown()
    else:
        logger.error("服务启动失败")
        sys.exit(1)

# 如果这个文件被直接运行（而不是被导入），则执行主函数
if __name__ == "__main__":
    main()