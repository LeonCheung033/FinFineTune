"""
日志工具类
作用：为GRPO训练过程提供统一的日志记录功能
功能：支持控制台输出、文件记录、训练过程跟踪
"""
import logging                    # Python标准日志库
import sys                       # 系统相关功能
from datetime import datetime    # 日期时间处理
from pathlib import Path        # 路径操作工具


class Logger:
    """
    训练日志管理器
    
    作用：统一管理GRPO训练过程中的所有日志输出
    功能：
    1. 支持多种日志级别（INFO、WARNING、ERROR、DEBUG）
    2. 同时输出到控制台和文件
    3. 提供训练专用的日志记录方法
    4. 自动格式化日志消息
    """
    
    def __init__(self, name: str = "GRPO", log_level: str = "INFO", 
                 log_file: str = None, console_output: bool = True):
        """
        初始化日志器
        
        参数说明：
            name: 日志器名称，用于区分不同模块的日志
            log_level: 日志级别（DEBUG < INFO < WARNING < ERROR）
            log_file: 日志文件路径，如果为None则不保存到文件
            console_output: 是否同时输出到控制台
        """
        # 创建Python标准日志器实例
        self.logger = logging.getLogger(name)
        # 设置日志级别 - 只有达到此级别的消息才会被记录
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除已有的处理器，避免重复输出
        self.logger.handlers.clear()
        
        # 设置日志消息格式
        # 格式：时间 - 日志器名称 - 级别 - 消息内容
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
        )
        
        # 控制台输出处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)  # 输出到标准输出
            console_handler.setFormatter(formatter)              # 应用格式
            self.logger.addHandler(console_handler)              # 添加到日志器
        
        # 文件输出处理器
        if log_file:
            # 确保日志目录存在
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')  # 创建文件处理器
            file_handler.setFormatter(formatter)                           # 应用格式
            self.logger.addHandler(file_handler)                           # 添加到日志器
    
    def info(self, message: str):
        """
        输出信息日志
        用途：记录一般性信息，如训练进度、状态更新等
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """
        输出警告日志
        用途：记录可能的问题，但不影响程序继续运行
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """
        输出错误日志
        用途：记录严重错误，可能导致程序异常
        """
        self.logger.error(message)
    
    def debug(self, message: str):
        """
        输出调试日志
        用途：记录详细的调试信息，通常只在开发阶段使用
        """
        self.logger.debug(message)
    
    def log_training_start(self, config):
        """
        记录训练开始信息
        
        参数：
            config: 训练配置对象，包含所有训练参数
        
        作用：在训练开始时记录完整的配置信息，便于后续追踪
        """
        self.info("=" * 60)
        self.info("GRPO金融模型训练开始")
        self.info("=" * 60)
        self.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 记录关键配置信息
        # 使用嵌套路径访问配置项（如 'model.model_name'）
        self.info(f"基座模型: {config.get('model.model_name')}")
        self.info(f"奖励模型: {config.get('model.reward_model_path')}")
        
        # 检查是否使用LoRA微调
        lora_path = config.get('model.lora_model_path')
        if lora_path:
            self.info(f"LoRA模型: {lora_path}")
        
        # 记录数据和输出配置
        self.info(f"训练数据: {config.get('data.train_data_path')}")
        self.info(f"输出目录: {config.get('output.output_dir')}")
        
        # 记录训练超参数
        self.info(f"学习率: {config.get('training.learning_rate')}")
        self.info(f"训练轮数: {config.get('training.num_train_epochs')}")
        self.info(f"批次大小: {config.get('training.per_device_batch_size')}")
        self.info(f"梯度累积: {config.get('training.gradient_accumulation_steps')}")
        self.info("=" * 60)
    
    def log_training_end(self, success: bool = True):
        """
        记录训练结束信息
        
        参数：
            success: 训练是否成功完成
        
        作用：记录训练结束状态和时间
        """
        self.info("=" * 60)
        if success:
            self.info("GRPO训练成功完成！")
        else:
            self.error("GRPO训练失败！")
        self.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 60)
    
    def log_step(self, step: int, loss: float, learning_rate: float, 
                 reward_mean: float = None):
        """
        记录训练步骤信息
        
        参数：
            step: 当前训练步数
            loss: 当前损失值
            learning_rate: 当前学习率
            reward_mean: 平均奖励值（GRPO特有）
        
        作用：记录每个训练步骤的关键指标
        """
        msg = f"Step {step} - Loss: {loss:.4f} - LR: {learning_rate:.2e}"
        if reward_mean is not None:
            msg += f" - Reward: {reward_mean:.4f}"
        self.info(msg)


# 创建全局日志实例的工厂函数
def create_logger(output_dir: str = "./output", log_level: str = "INFO") -> Logger:
    """
    创建日志实例
    
    参数：
        output_dir: 输出目录，日志文件将保存在此目录下
        log_level: 日志级别
    
    返回：
        Logger实例
    
    作用：为整个项目提供统一的日志创建入口
    """
    # 创建日志文件路径
    log_file = Path(output_dir) / "training.log"
    
    return Logger(
        name="GRPO_Training",      # 日志器名称
        log_level=log_level,       # 日志级别
        log_file=str(log_file),    # 日志文件路径
        console_output=True        # 启用控制台输出
    )