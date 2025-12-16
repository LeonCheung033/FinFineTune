#!/bin/bash

# ==================== GRPO训练脚本 ====================
# 作用：支持nohup后台运行和多机多卡训练的启动脚本
# 功能：环境配置、进程管理、分布式训练、错误处理

# 设置严格模式：任何命令失败都会导致脚本退出
set -e

# ==================== 多机多卡配置 ====================
# 定义集群节点配置：IP地址到GPU数量的映射
declare -A NODES=(
    ["10.60.68.220"]=1    # 主节点：拥有2张GPU卡
    ["10.60.98.173"]=1    # 从节点：拥有1张GPU卡
)
MASTER_ADDR="10.60.68.220"  # 主节点IP地址

# ==================== 脚本参数解析 ====================
# 初始化命令行参数变量
USE_NOHUP=false        # 是否使用nohup后台运行
STOP_TRAINING=false    # 是否停止训练
CHECKPOINT_PATH=""     # checkpoint恢复路径
USE_MULTINODE=false    # 是否使用多机训练

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nohup) USE_NOHUP=true; shift ;;           # 启用后台运行
        --stop) STOP_TRAINING=true; shift ;;        # 停止训练
        --resume) CHECKPOINT_PATH="$2"; shift 2 ;;  # 从checkpoint恢复
        --multinode) USE_MULTINODE=true; shift ;;   # 启用多机训练
        *) echo "用法: $0 [--nohup] [--stop] [--resume <checkpoint_path>] [--multinode]"; exit 1 ;;
    esac
done

# ==================== 工具函数定义 ====================

create_hostfile() {
    """
    创建DeepSpeed hostfile
    
    作用：
    1. 生成DeepSpeed分布式训练需要的主机文件
    2. 指定每个节点的GPU数量
    3. 支持多机多卡训练配置
    
    hostfile格式：
    IP地址 slots=GPU数量
    """
    echo "创建DeepSpeed hostfile..."
    > ./hostfile  # 清空hostfile
    
    # 遍历所有节点配置
    for node_ip in "${!NODES[@]}"; do
        gpu_count=${NODES[$node_ip]}
        echo "$node_ip slots=$gpu_count" >> ./hostfile
    done
    
    echo "Hostfile创建完成:"
    cat ./hostfile  # 显示hostfile内容
}

stop_all_training() {
    """
    停止所有节点的训练进程
    
    作用：
    1. 在所有节点上查找并终止训练进程
    2. 支持单机和多机环境
    3. 确保进程完全清理
    
    进程识别：
    - 通过进程命令行参数识别训练进程
    - 使用pkill命令批量终止进程
    """
    echo "停止所有节点的训练进程..."
    
    for node_ip in "${!NODES[@]}"; do
        if [[ "$node_ip" == "$MASTER_ADDR" ]]; then
            # 在主节点上直接执行
            pkill -f "deepspeed.*train.py" 2>/dev/null || true
        else
            # 通过SSH在从节点上执行
            ssh ubuntu@$node_ip "pkill -f 'deepspeed.*train.py' 2>/dev/null || true" &
        fi
    done
    
    wait    # 等待所有后台任务完成
    sleep 3 # 等待进程完全退出
    echo "所有节点训练进程清理完成"
}

# ==================== 停止训练功能 ====================
if [ "$STOP_TRAINING" = true ]; then
    if [ "$USE_MULTINODE" = true ]; then
        # 多机环境：停止所有节点
        stop_all_training
    else
        # 单机环境：只停止本地进程
    echo "正在停止GRPO训练进程..."
    TRAIN_PIDS=$(ps aux | grep "deepspeed.*train.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$TRAIN_PIDS" ]; then
        echo "未找到运行中的训练进程"
    else
        echo "停止进程: $TRAIN_PIDS"
        for pid in $TRAIN_PIDS; do
                kill -TERM $pid  # 发送终止信号
        done
        echo "训练进程已停止"
        fi
    fi
    exit 0
fi

# ==================== 环境准备 ====================

# 切换到脚本所在目录
cd "$(dirname "$0")"
    
# 创建日志目录（带时间戳）
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ==================== 环境变量设置 ====================
# GPU和CUDA配置
export CUDA_VISIBLE_DEVICES=0,1        # 指定可见的GPU设备
export NCCL_DEBUG=WARN                 # NCCL调试级别
export NCCL_IB_DISABLE=1               # 禁用InfiniBand
export NCCL_IGNORE_DISABLED_P2P=1      # 忽略P2P通信问题

# Python环境配置
export PYTHONPATH=$PYTHONPATH:$(pwd)   # 添加当前目录到Python路径
export PDSH_RCMD_TYPE=ssh              # 设置远程命令执行方式

# ==================== NCCL优化配置 ====================
# NCCL（NVIDIA Collective Communication Library）是GPU间通信库
# 这些配置用于解决多机多卡训练中的通信问题
export NCCL_BUFFSIZE=524288             # 强制使用524288字节缓冲区
export NCCL_MAX_NCHANNELS=1             # 限制最大通道数量
export NCCL_MIN_NCHANNELS=1             # 限制最小通道数量
export NCCL_PROTO=Simple                # 使用简单通信协议
export NCCL_ALGO=Ring                   # 使用Ring通信算法
export NCCL_TREE_THRESHOLD=0            # 禁用树形通信
export NCCL_NET_GDR_LEVEL=0             # 禁用GPU Direct RDMA
export NCCL_SOCKET_IFNAME=eth0          # 强制使用eth0网卡

# 创建输出目录
mkdir -p "./output/"

# ==================== 训练命令构建 ====================

build_train_command() {
    """
    构建训练命令
    
    参数：无（使用全局变量）
    
    返回：完整的训练命令字符串
    
    作用：
    1. 根据运行模式构建不同的命令
    2. 支持单机和多机训练
    3. 处理checkpoint恢复
    4. 动态配置参数
    """
    local cmd
    local current_dir=$(pwd)
    
    if [ "$USE_MULTINODE" = true ]; then
        # 多机训练命令
        cmd="cd $current_dir && deepspeed --hostfile=hostfile $current_dir/train.py --deepspeed $current_dir/configs/deepspeed_zero2.json"
    else
        # 单机训练命令
        cmd="deepspeed --num_gpus=2 train.py --deepspeed configs/deepspeed_zero2.json"
    fi
    
    # 添加checkpoint恢复参数
    if [ -n "$CHECKPOINT_PATH" ]; then
        cmd="$cmd --resume_from_checkpoint $CHECKPOINT_PATH"
        echo "从checkpoint恢复训练: $CHECKPOINT_PATH" >&2
    fi
    
    echo "$cmd"
}

# ==================== 训练执行函数 ====================

run_training() {
    """
    执行训练
    
    作用：
    1. 激活Python环境
    2. 准备多机环境（如需要）
    3. 执行训练命令
    4. 处理训练结果
    
    返回：0表示成功，1表示失败
    """
    echo "开始GRPO训练 - $(date)"
    
    # 激活conda环境
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reward3 
    
    # 多机模式需要先准备环境
    if [ "$USE_MULTINODE" = true ]; then
        stop_all_training  # 清理旧进程
        create_hostfile    # 创建hostfile
    fi
    
    # 构建并执行训练命令
    local train_cmd=$(build_train_command)
    if eval "$train_cmd"; then
        echo "训练成功完成 - $(date)"
        return 0
    else
        echo "训练失败 - $(date)"
        return 1
    fi
}

# ==================== 主执行逻辑 ====================

if [ "$USE_NOHUP" = true ]; then
    # ==================== 后台运行模式 ====================
    echo "后台运行训练，日志: $LOG_DIR/training.log"
    echo "查看进度: tail -f $LOG_DIR/training.log"
    echo "停止训练: pkill -f 'deepspeed.*train.py'"
    
    # 多机模式需要先准备环境
    if [ "$USE_MULTINODE" = true ]; then
        stop_all_training
        create_hostfile
    fi
    
    # 构建完整的后台命令
    train_cmd=$(build_train_command)
    nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate reward3 && $train_cmd" > "$LOG_DIR/training.log" 2>&1 &
    
    # 记录进程ID
    TRAIN_PID=$!
    echo "训练进程已启动 (PID: $TRAIN_PID)"
    echo $TRAIN_PID > "$LOG_DIR/train.pid"
else
    # ==================== 前台运行模式 ====================
    # 重定向输出到日志文件同时显示在控制台
    exec > >(tee -a "$LOG_DIR/training.log")
    exec 2>&1
    run_training
fi