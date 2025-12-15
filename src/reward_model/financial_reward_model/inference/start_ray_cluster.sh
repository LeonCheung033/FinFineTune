#!/bin/bash

# 设置脚本在遇到错误时立即退出
set -e

# 定义集群节点IP地址
# 这些IP地址应该根据实际的集群配置进行修改
HEAD_NODE="10.60.11.131"    # 头节点IP地址
WORKER_NODE="10.60.240.249" # 工作节点IP地址

# 检查命令行参数数量
# 这个脚本至少需要一个参数（模型路径）
if [ $# -lt 1 ]; then
    echo "用法: $0 <模型路径> [副本数] [端口]"
    echo "示例: $0 output/checkpoint-700 2 8000"
    exit 1
fi

# 解析命令行参数
MODEL_PATH=$1              # 第一个参数：模型路径
NUM_REPLICAS=${2:-2}       # 第二个参数：副本数量，默认为2
PORT=${3:-8000}            # 第三个参数：服务端口，默认为8000

# 将相对路径转换为绝对路径
# 这确保了无论在哪个目录执行脚本，路径都是正确的
MODEL_PATH=$(realpath "$MODEL_PATH")

echo "启动验证服务"
echo "模型路径: $MODEL_PATH"
echo "副本数: $NUM_REPLICAS"
echo "端口: $PORT"

# 检查模型路径和必要文件是否存在
# 这些检查确保模型文件完整，避免启动后才发现文件缺失
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "错误: 模型配置文件不存在: $MODEL_PATH/config.json"
    exit 1
fi

if [ ! -f "$MODEL_PATH/tokenizer_config.json" ]; then
    echo "错误: tokenizer配置文件不存在: $MODEL_PATH/tokenizer_config.json"
    exit 1
fi

# 停止现有的服务进程
# 这确保了不会有端口冲突或资源冲突
echo "停止现有服务"
# 在本地停止ray_reward_service进程
pkill -f "ray_reward_service" 2>/dev/null || true
# 在工作节点停止ray进程
ssh ubuntu@$WORKER_NODE "pkill -f ray 2>/dev/null || true" 2>/dev/null || true

# 启动Ray集群
echo "启动Ray集群"
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# 强制停止现有的Ray进程，然后启动头节点
ray stop --force 2>/dev/null || true
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=4 --num-cpus=24

# 启动工作节点
echo "连接工作节点"
# 通过SSH连接到工作节点并启动Ray worker
ssh ubuntu@$WORKER_NODE "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate vllm
    ray stop --force 2>/dev/null || true
    ray start --address='$HEAD_NODE:6379' --num-gpus=2
" 2>/dev/null || echo "工作节点连接失败，继续单节点运行"

# 等待集群初始化完成
sleep 5

# 检查集群状态
# 这个命令会显示集群中的节点数量、资源等信息
echo "检查集群状态"
ray status

# 启动推理服务
echo "启动推理服务"
# 在后台启动Python推理服务，并将输出重定向到日志文件
python inference/ray_reward_service.py \
    --model_path "$MODEL_PATH" \
    --num_replicas "$NUM_REPLICAS" \
    --port "$PORT" > validation_service.log 2>&1 &

# 获取服务进程ID并保存到文件
SERVICE_PID=$!
echo $SERVICE_PID > validation_service.pid

echo "服务启动完成"
echo "PID: $SERVICE_PID"
echo "API地址: http://localhost:$PORT"

# 等待服务启动并检查服务状态
echo "等待服务启动"
for i in {1..20}; do
    echo "检查服务状态 ($i/20)"
    
    # 检查服务进程是否还在运行
    # kill -0 用于检查进程是否存在，不会实际终止进程
    if ! kill -0 $SERVICE_PID 2>/dev/null; then
        echo "错误: 服务进程已退出"
        echo "查看日志:"
        cat validation_service.log
        exit 1
    fi
    
    # 等待一段时间再检查健康状态
    # 给服务一些时间来完成初始化
    sleep 10
    
    # 检查服务的健康状态
    # 通过发送HTTP请求到健康检查端点来验证服务是否正常
    if curl -s --connect-timeout 5 --max-time 10 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "服务连接正常"
        # 获取并格式化显示健康检查结果
        curl -s "http://localhost:$PORT/health" | python -m json.tool 2>/dev/null || echo "健康检查通过"
        break
    else
        echo "服务尚未就绪，继续等待..."
        # 如果达到最大重试次数，报告超时错误
        if [ $i -eq 20 ]; then
            echo "错误: 服务启动超时"
            echo "查看日志:"
            tail -50 validation_service.log
            kill $SERVICE_PID 2>/dev/null || true
            exit 1
        fi
    fi
done

echo "验证服务已就绪"
echo "日志文件: validation_service.log"
echo "停止命令: kill $SERVICE_PID"