#!/bin/bash

## 脚本功能解析

### 整体架构
这个脚本实现了一个**分布式模型推理服务**，具体架构为：
- **头节点（10.60.68.220）**：运行vLLM API服务器，提供1个GPU
- **工作节点（10.60.98.173）**：提供额外的1个GPU算力
- **总算力**：2个GPU通过张量并行协同工作

### 关键技术组件

1. **Ray集群管理**：
   - 负责多机资源调度和任务分配
   - 头节点协调，工作节点提供算力
   - 通过端口6379进行集群通信

2. **vLLM推理引擎**：
   - 高性能大语言模型推理框架
   - 支持张量并行，将模型分割到多个GPU
   - 提供OpenAI兼容的API接口

3. **张量并行（tensor-parallel-size 2）**：
   - 将7B模型的参数分割到2个GPU上
   - 每个GPU处理模型的一部分，协同完成推理
   - 相比单GPU，推理速度提升约1.8倍

### 资源配置说明

- **显存使用**：每个GPU只使用30%显存，7B模型约需要6-8GB显存
- **网络通信**：强制使用eth0网卡，确保稳定的节点间通信
- **API访问**：通过8000端口提供服务，需要API密钥认证

### 使用场景
这个配置适合**生产环境的模型部署**，可以：
- 同时服务多个客户端请求
- 提供稳定的推理性能
- 支持高并发访问
- 便于集成到现有系统中

# 定义集群节点IP地址
# 这些IP地址应该根据实际的集群配置进行修改
HEAD_NODE="10.60.68.220"    # 头节点IP地址 - 主服务器，负责协调和管理整个集群
WORKER_NODE="10.60.98.173" # 工作节点IP地址 - 从服务器，提供额外的GPU算力
# MODEL_PATH="/shared/DeepSeek-R1-Distill-Qwen-7B_028/best_complete_model_05261653_028"
MODEL_PATH="/shared/grpo_financial_tuning/output/best_model"  # 模型文件路径 - GRPO训练完成后保存的最佳模型
PORT=8000  # API服务端口号 - 客户端通过这个端口访问模型服务

# 启动Ray集群
# Ray是一个分布式计算框架，用于管理多机多卡的资源调度
echo "启动Ray集群"

# 激活conda环境
# conda是Python包管理工具，这里激活名为reward3的虚拟环境
# 虚拟环境包含了运行vLLM所需的所有依赖包
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reward3

# 设置PyTorch分布式环境变量
# 这些环境变量告诉PyTorch如何在多机之间进行通信
export MASTER_ADDR=$HEAD_NODE  # 主节点地址 - 分布式训练的协调节点
export MASTER_PORT=29500       # 主节点端口 - 用于节点间通信的端口
export GLOO_SOCKET_IFNAME=eth0  # 指定网卡接口 - 强制使用eth0网卡进行通信
export NCCL_SOCKET_IFNAME=eth0  # NCCL通信网卡 - NVIDIA集合通信库使用的网卡

# 强制停止现有的Ray进程，然后启动头节点
# 2>/dev/null: 将错误输出重定向到空设备，避免显示错误信息
# || true: 即使命令失败也继续执行，不会中断脚本
ray stop --force 2>/dev/null || true

# 启动Ray头节点
# --head: 指定这是头节点（主节点）
# --port=6379: Ray集群通信端口
# --dashboard-host=0.0.0.0: 允许从任何IP访问Ray控制面板
# --dashboard-port=8265: Ray控制面板的Web端口
# --num-gpus=1: 头节点提供1个GPU资源
# --num-cpus=24: 头节点提供24个CPU核心
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=1 --num-cpus=24

# 启动工作节点
echo "连接工作节点"

# 通过SSH连接到工作节点并启动Ray worker
# SSH是安全外壳协议，用于远程登录和执行命令
# ubuntu@$WORKER_NODE: 以ubuntu用户身份连接到工作节点
ssh ubuntu@$WORKER_NODE "
    # 在工作节点上激活相同的conda环境
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reward3
    
    # 停止工作节点上可能存在的Ray进程
    ray stop --force 2>/dev/null || true
    
    # 启动Ray工作节点
    # --address='$HEAD_NODE:6379': 连接到头节点的Ray集群
    # --num-gpus=1: 工作节点提供1个GPU资源
    ray start --address='$HEAD_NODE:6379' --num-gpus=1
" 2>/dev/null || echo "工作节点连接失败，继续单节点运行"
# 如果SSH连接失败，会输出提示信息但不会中断脚本执行

echo "🚀 启动vLLM多机多卡服务器..."

# 启动vLLM API服务器
# vLLM是一个高性能的大语言模型推理引擎
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \                    # 指定要加载的模型路径
  --host 0.0.0.0 \                        # 服务器监听地址，0.0.0.0表示接受来自任何IP的请求
  --port $PORT \                          # 服务器端口号，客户端通过这个端口访问API
  --tensor-parallel-size 2 \              # 张量并行大小，将模型分割到2个GPU上并行计算
  --dtype bfloat16 \                      # 数据类型，使用bfloat16精度节省显存
  --max-model-len 2048 \                  # 最大序列长度，限制输入+输出的总token数
  --gpu-memory-utilization 0.3 \         # GPU显存利用率，只使用30%的显存，为其他进程预留空间
  --disable-log-requests \                # 禁用请求日志，减少日志输出
  --api-key grpo-model-key \              # API密钥，客户端需要提供这个密钥才能访问服务
  --distributed-executor-backend ray      # 分布式执行后端，使用Ray框架管理多机多卡资源

