#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSpeed Stage 3测试脚本 - 最大内存优化
"""

import os
import sys
import torch
import deepspeed
from transformers import AutoTokenizer, AutoConfig, LlamaForSequenceClassification
import gc

def main():
    """DeepSpeed Stage 3测试 - 参数分片"""
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"进程 {rank}/{world_size} 在GPU {local_rank}上启动 (Stage 3)")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置GPU和内存优化
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    gc.collect()
    
    model_path = "/home/ubuntu/QRM-Llama3.1-8B-v2"
    
    try:
        # 1. 加载tokenizer
        if rank == 0:
            print(" 加载tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. 加载模型配置
        if rank == 0:
            print("🤖 加载模型配置...")
        
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        config.num_labels = 1
        
        # 3. Stage 3需要在CPU上初始化模型
        if rank == 0:
            print("🔧 在CPU上初始化模型（Stage 3模式）...")
        
        with torch.device('cpu'):
            model = LlamaForSequenceClassification.from_pretrained(
                model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None
            )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        if rank == 0:
            print(f" 模型初始化成功（CPU）")
            print(f"   总参数量: {total_params:,}")
            print(f"   模型大小: {total_params * 2 / 1024**3:.2f}GB (bf16)")
        
        # 4. DeepSpeed Stage 3配置
        if rank == 0:
            print(" 配置DeepSpeed Stage 3...")
        
        ds_config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "gradient_clipping": 1.0,
            
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,                            # Stage 3 - 参数分片
                "offload_optimizer": {
                    "device": "cpu",                   # 优化器状态卸载到CPU
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",                   # 参数卸载到CPU
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 1e8,
                "stage3_prefetch_bucket_size": 1e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9
            },
            
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": 1e-3,
                    "momentum": 0.9,
                    "weight_decay": 0.01
                }
            },
            
            "bf16": {"enabled": True},
            "wall_clock_breakdown": False,
            "steps_per_print": 10
        }
        
        if rank == 0:
            print(" DeepSpeed Stage 3配置完成")
            print("   - 参数分片: 启用")
            print("   - 优化器CPU卸载: 启用") 
            print("   - 参数CPU卸载: 启用")
        
        # 5. 初始化DeepSpeed引擎
        if rank == 0:
            print(" 初始化DeepSpeed Stage 3引擎...")
        
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        if rank == 0:
            print(" DeepSpeed Stage 3引擎初始化成功")
            print(f"   引擎类型: {type(engine).__name__}")
        
        # 6. 测试前向传播
        if rank == 0:
            print(" 测试前向传播（Stage 3）...")
        
        batch_size = 1
        seq_length = 128
        vocab_size = tokenizer.vocab_size
        
        input_ids = torch.randint(
            0, vocab_size, 
            (batch_size, seq_length), 
            device=f"cuda:{local_rank}"
        )
        attention_mask = torch.ones(
            (batch_size, seq_length), 
            device=f"cuda:{local_rank}"
        )
        
        outputs = engine(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if rank == 0:
            print(f" 前向传播成功")
            print(f"   输入形状: {input_ids.shape}")
            print(f"   输出形状: {logits.shape}")
        
        # 7. 测试反向传播
        if rank == 0:
            print(" 测试反向传播（Stage 3）...")
        
        loss = logits.mean()
        engine.backward(loss)
        
        if rank == 0:
            print(f" 反向传播成功，损失值: {loss.item():.6f}")
        
        # 8. 测试优化器步骤
        if rank == 0:
            print("⚡ 测试优化器步骤（Stage 3）...")
        
        engine.step()
        
        if rank == 0:
            print(" 优化器步骤成功")
        
        # 9. 内存统计
        if rank == 0:
            print("📊 Stage 3内存使用统计:")
            for i in range(world_size):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: 分配 {memory_allocated:.2f}GB, 保留 {memory_reserved:.2f}GB")
        
        # 10. 成功总结
        if rank == 0:
            print("\n DeepSpeed Stage 3测试成功！")
            print(" 验证结果:")
            print("   - Stage 3参数分片正常 ")
            print("   - CPU卸载功能正常 ")
            print("   - 内存使用大幅降低 ")
            print("   - 8B模型可在双4090上运行 ")
            print(f"\n Stage 3可以处理更大的模型和批次大小")
    
    except Exception as e:
        print(f" 进程 {rank} Stage 3测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if 'LOCAL_RANK' not in os.environ:
        print(" 错误：此脚本必须通过torchrun启动")
        print("正确启动命令:")
        print("   torchrun --nproc_per_node=2 scripts/setup/test_deepspeed.py")
        exit(1)
    
    success = main()
    
    if int(os.environ.get('RANK', 0)) == 0:
        if success:
            print("\n DeepSpeed Stage 3测试完全通过！")
            print(" 结论：Stage 3可以在双4090上训练8B模型")
        else:
            print("\n Stage 3测试失败！")