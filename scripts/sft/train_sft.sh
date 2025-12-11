# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 添加PyTorch内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动分布式训练
# 使用deepspeed的launcher
deepspeed 3_finetuning.py \
    --deepspeed deepspeed_zero2.json \
    --num_gpus=4

# nohup bash train_sft.sh > 2025_12_11_14_41_sft.log 2>&1 &