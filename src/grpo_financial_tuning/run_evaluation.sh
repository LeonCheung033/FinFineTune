#!/bin/bash

# 多机多卡评测脚本

# 配置参数
# MODEL_PATH="./output/best_model"
MODEL_PATH="/shared/DeepSeek-R1-Distill-Qwen-7B_028/best_complete_model_05261653_028"
REWARD_MODEL_PATH="/shared/Skywork-Reward_checkpoint-1000/checkpoint-1000"
PROMPTS_PATH="/shared/grpo_financial_tuning/data/test_prompts_dataset.jsonl"
SAMPLE_SIZE=50

# 多机多卡配置
declare -A NODES=(
    ["10.60.68.220"]=1    # 主节点：2张GPU
    ["10.60.98.173"]=1    # 从节点：1张GPU
)
MASTER_ADDR="10.60.68.220"

# 创建hostfile
create_hostfile() {
    echo "创建DeepSpeed hostfile..."
    > ./hostfile
    for node_ip in "${!NODES[@]}"; do
        gpu_count=${NODES[$node_ip]}
        echo "$node_ip slots=$gpu_count" >> ./hostfile
    done
    cat ./hostfile
}

# 单机多卡模式
single_node() {
    echo "启动单机多卡评测..."
    deepspeed --num_gpus=2 model_eval/multigpu_evaluator.py \
        --model $MODEL_PATH \
        --reward_model $REWARD_MODEL_PATH \
        --prompts $PROMPTS_PATH \
        --sample_size $SAMPLE_SIZE
}

# 多机多卡模式
multi_node() {
    echo "启动多机多卡评测..."
    create_hostfile
    
    deepspeed --hostfile=hostfile model_eval/multigpu_evaluator.py \
        --model $MODEL_PATH \
        --reward_model $REWARD_MODEL_PATH \
        --prompts $PROMPTS_PATH \
        --sample_size $SAMPLE_SIZE
}

# 解析参数
case $1 in
    --single)
        single_node
        ;;
    --multi)
        multi_node
        ;;
    *)
        echo "用法: $0 [--single|--multi]"
        echo "  --single: 单机多卡模式"
        echo "  --multi:  多机多卡模式"
        exit 1
        ;;
esac