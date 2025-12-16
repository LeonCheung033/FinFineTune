#!/bin/bash

# vLLM量化模型部署脚本
#
# 这个脚本的主要作用：
# 1. 启动Ray分布式计算框架（用于多GPU并行推理）
# 2. 部署量化模型为API服务
# 3. 提供OpenAI兼容的API接口
# 4. 支持高并发推理请求
#
# 使用场景：
# - 生产环境部署量化模型
# - 提供HTTP API接口供其他系统调用
# - 支持多用户并发访问
# - 实现模型推理服务化

# 基本配置参数
HEAD_NODE="10.60.68.220" 
MODEL_PATH="/shared/gptq_model/output/best_model_gptq_int4"
PORT=8000

echo "启动Ray集群"

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate reward3

# 设置分布式训练环境变量
export MASTER_ADDR=$HEAD_NODE
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

# 停止现有的Ray进程
# 确保没有残留的Ray进程影响新的启动
ray stop --force 2>/dev/null || true  # --force强制停止，2>/dev/null忽略错误输出

# 启动Ray头节点
# Ray是分布式计算框架，用于管理多GPU资源
ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=1 \
  --num-cpus=24

echo "启动vLLM单机服务器..."

# 启动vLLM API服务器
# vLLM是高性能推理引擎，支持量化模型推理
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  --disable-log-requests \
  --api-key gptq-model-key \
  --distributed-executor-backend ray

# 测试API服务是否正常启动
# 使用Python发送HTTP请求测试API接口
python -c "
import requests
resp = requests.post('http://localhost:8000/v1/chat/completions', 
    headers={'Authorization': 'Bearer gptq-model-key'},
    json={'model': '/shared/gptq_model/output/best_model_gptq_int4', 'messages': [{'role': 'user', 'content': '你是一个专业的金融领域分析师请对以下问题进行详细解答：周三盘前交易时段，金融板块呈现温和上涨态势：金融精选行业SPDR基金(XLF)上涨0.4%，Direxion每日三倍做多金融股ETF(FAS)上涨1.2%，而其反向产品Direxion每日三倍做空金融股ETF(FAZ)下跌1.1%。万事达卡(MA)股价上涨0.7%，因其董事会批准了最高110亿美元的普通股回购计划，并将季度股息提高16%。Cboe全球市场(CBOE)股价上涨0.5%，该公司披露其指数期权合约的11月日均交易量达到近400万份，较去年同期的330万份增长22%。基于这些市场动态，请分析以下问题：在当前市场环境下，影响金融板块ETF(XLF、FAS、FAZ)价格波动的关键驱动因素有哪些？如何评估万事达卡大规模股票回购和股息增长对其长期估值的影响？Cboe交易量的大幅增长可能预示着哪些市场结构变化？请综合考虑宏观经济环境、行业竞争格局和公司特定事件等多重因素展开分析。'}], 'max_tokens': 1024})
print('Status Code:', resp.status_code)
print('Response:', resp.json())
"