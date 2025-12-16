"""
配置文件加载工具类
作用：为GRPO训练提供统一的配置管理功能
功能：支持JSON配置文件加载、嵌套配置访问、DeepSpeed配置管理
"""
import json                              # JSON数据处理
from pathlib import Path                 # 路径操作
from typing import Dict, Any, Optional   # 类型提示


class ConfigLoader:
    """
    配置文件加载器
    
    作用：负责加载和缓存JSON配置文件
    功能：
    1. 加载JSON配置文件
    2. 缓存配置以提高性能
    3. 保存配置到文件
    4. 处理文件读写错误
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        初始化配置加载器
        
        参数：
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)  # 配置文件目录
        self._config_cache = {}             # 配置缓存字典
    
    def load_json(self, file_name: str) -> Dict[str, Any]:
        """
        加载JSON配置文件
        
        参数：
            file_name: 配置文件名
            
        返回：
            Dict: 配置字典
        
        作用：
        1. 检查缓存中是否已有配置
        2. 读取并解析JSON文件
        3. 缓存配置以提高后续访问速度
        4. 处理文件不存在和JSON格式错误
        """
        # 检查缓存
        if file_name in self._config_cache:
            return self._config_cache[file_name]
        
        # 构建文件路径
        file_path = self.config_dir / file_name
        
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        try:
            # 读取并解析JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 缓存配置
            self._config_cache[file_name] = config
            return config
            
        except json.JSONDecodeError as e:
            # JSON格式错误
            raise ValueError(f"配置文件JSON格式错误 {file_path}: {e}")
    
    def save_json(self, file_name: str, config: Dict[str, Any]):
        """
        保存配置到JSON文件
        
        参数：
            file_name: 配置文件名
            config: 配置字典
        
        作用：
        1. 确保目录存在
        2. 保存配置到JSON文件
        3. 更新缓存
        """
        file_path = self.config_dir / file_name
        
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 更新缓存
        self._config_cache[file_name] = config


class ProjectConfig:
    """
    项目配置管理器
    
    作用：管理GRPO训练项目的所有配置
    功能：
    1. 支持嵌套配置访问（如 "model.model_name"）
    2. 提供配置的获取、设置、批量更新
    3. 配置文件保存和加载
    4. 配置信息打印
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        初始化项目配置
        
        参数：
            config_file: 配置文件名
        """
        self.config_loader = ConfigLoader()                    # 配置加载器
        self.config_file = config_file                         # 配置文件名
        self._config = self.config_loader.load_json(config_file)  # 加载配置
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套键路径）
        
        参数：
            key_path: 键路径，如 "model.model_name"
            default: 默认值，当键不存在时返回
            
        返回：
            配置值
        
        作用：
        1. 支持点号分隔的嵌套路径访问
        2. 安全地处理不存在的键
        3. 提供默认值机制
        
        示例：
            config.get("model.model_name") -> "DeepSeek-R1-Distill-Qwen-7B"
            config.get("training.learning_rate") -> 5e-6
        """
        keys = key_path.split('.')  # 按点号分割路径
        value = self._config
        
        try:
            # 逐级访问嵌套字典
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # 键不存在或类型错误时返回默认值
            return default
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值
        
        参数：
            key_path: 键路径
            value: 要设置的值
        
        作用：
        1. 支持嵌套路径设置
        2. 自动创建中间层级的字典
        3. 更新内存中的配置
        """
        keys = key_path.split('.')
        config = self._config
        
        # 导航到父级，创建中间层级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置最终值
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置
        
        参数：
            updates: 更新字典，键为路径格式
        
        作用：一次性更新多个配置项
        
        示例：
            config.update({
                "training.learning_rate": 1e-5,
                "training.batch_size": 4
            })
        """
        for key_path, value in updates.items():
            self.set(key_path, value)
    
    def save(self):
        """
        保存配置到文件
        
        作用：将内存中的配置写入到原始配置文件
        """
        self.config_loader.save_json(self.config_file, self._config)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        返回完整配置字典
        
        返回：
            Dict: 配置字典的副本
        
        作用：获取完整配置的副本，避免外部修改
        """
        return self._config.copy()
    
    def print_config(self):
        """
        打印当前配置
        
        作用：
        1. 格式化显示关键配置信息
        2. 便于调试和确认配置
        3. 在训练开始前进行配置检查
        """
        print("=" * 50)
        print("GRPO训练配置")
        print("=" * 50)
        print(f"基座模型: {self.get('model.model_name')}")
        print(f"奖励模型: {self.get('model.reward_model_path')}")
        
        # 检查是否使用LoRA
        lora_path = self.get('model.lora_model_path')
        if lora_path:
            print(f"LoRA模型: {lora_path}")
        
        # 数据和输出配置
        print(f"训练数据: {self.get('data.train_data_path')}")
        print(f"输出目录: {self.get('output.output_dir')}")
        
        # 训练超参数
        print(f"学习率: {self.get('training.learning_rate')}")
        print(f"训练轮数: {self.get('training.num_train_epochs')}")
        print(f"每GPU批次: {self.get('training.per_device_batch_size')}")
        print(f"梯度累积: {self.get('training.gradient_accumulation_steps')}")
        
        # GRPO特有参数
        print(f"生成样本数: {self.get('grpo.num_generations')}")
        print(f"使用Liger优化: {self.get('grpo.use_liger_loss')}")
        print("=" * 50)


class DeepSpeedConfig:
    """
    DeepSpeed配置管理器
    
    作用：管理DeepSpeed分布式训练的配置
    功能：
    1. 加载DeepSpeed配置文件
    2. 分离Accelerate和DeepSpeed配置
    3. 动态更新GPU数量配置
    4. 支持配置导出为YAML格式
    
    DeepSpeed是什么：
    - 微软开发的分布式训练框架
    - 支持ZeRO优化器状态分片
    - 可以训练超大规模模型
    - 与PyTorch和Transformers集成
    """
    
    def __init__(self, config_file: str = "deepspeed_zero3.json"):
        """
        初始化DeepSpeed配置
        
        参数：
            config_file: DeepSpeed配置文件名
        """
        self.config_loader = ConfigLoader()
        self.config_file = config_file
        self._config = self.config_loader.load_json(config_file)
    
    def get_accelerate_config(self) -> Dict[str, Any]:
        """
        获取Accelerate使用的配置部分
        
        返回：
            Accelerate配置字典
        
        作用：
        1. 过滤掉DeepSpeed特有的配置
        2. 返回Accelerate框架需要的配置
        3. 支持Accelerate + DeepSpeed集成
        """
        accelerate_config = {}
        exclude_keys = ['deepspeed_config']  # 排除的键
        
        for key, value in self._config.items():
            if key not in exclude_keys:
                accelerate_config[key] = value
        
        return accelerate_config
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """
        获取DeepSpeed引擎配置
        
        返回：
            DeepSpeed配置字典
        
        作用：提取DeepSpeed引擎需要的配置部分
        """
        return self._config.get('deepspeed_config', {})
    
    def update_gpu_count(self, num_gpus: int):
        """
        更新GPU数量配置
        
        参数：
            num_gpus: GPU数量
        
        作用：
        1. 更新进程数量配置
        2. 重新计算全局batch size
        3. 确保配置与硬件匹配
        
        批次大小计算：
        全局batch size = GPU数量 × 每GPU batch size × 梯度累积步数
        """
        # 更新进程数量
        self._config['num_processes'] = num_gpus
        
        # 获取DeepSpeed配置
        deepspeed_config = self._config.get('deepspeed_config', {})
        micro_batch_size = deepspeed_config.get('train_micro_batch_size_per_gpu', 1)
        gradient_accumulation = deepspeed_config.get('gradient_accumulation_steps', 4)
        
        # 计算全局batch size
        global_batch_size = num_gpus * micro_batch_size * gradient_accumulation
        deepspeed_config['train_batch_size'] = global_batch_size
    
    def save_to_yaml(self, output_path: str):
        """
        保存为YAML格式（兼容accelerate launch）
        
        参数：
            output_path: 输出文件路径
        
        作用：
        1. 转换为YAML格式
        2. 兼容accelerate launch命令
        3. 便于人类阅读和编辑
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装PyYAML: pip install PyYAML")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        返回完整配置字典
        
        返回：
            Dict: 配置字典的副本
        """
        return self._config.copy()


# 全局配置实例创建函数
def load_configs():
    """
    加载所有配置
    
    返回：
        Tuple: (项目配置, DeepSpeed配置)
    
    作用：一次性加载项目需要的所有配置
    """
    project_config = ProjectConfig()
    deepspeed_config = DeepSpeedConfig()
    return project_config, deepspeed_config


# 便捷函数
def get_project_config() -> ProjectConfig:
    """
    获取项目配置实例
    
    返回：
        ProjectConfig: 项目配置实例
    
    作用：提供全局访问项目配置的便捷方法
    """
    return ProjectConfig()


def get_deepspeed_config() -> DeepSpeedConfig:
    """
    获取DeepSpeed配置实例
    
    返回：
        DeepSpeedConfig: DeepSpeed配置实例
    
    作用：提供全局访问DeepSpeed配置的便捷方法
    """
    return DeepSpeedConfig()
