#!/usr/bin/env python3
"""
量化模型简单测试脚本

这个脚本的主要作用：
1. 快速测试量化模型是否正常工作
2. 验证模型加载和推理功能
3. 提供基本的模型调用示例
4. 用于开发和调试阶段的功能验证

使用场景：
- 量化完成后的功能验证
- 调试模型加载问题
- 测试不同的推理参数
- 验证模型输出质量
"""

import torch
from gptqmodel import GPTQModel        # 量化模型加载库
from transformers import AutoTokenizer  # 文本分词库

# 模型路径配置
model_path = "./output/best_model_gptq_int4"  # 量化模型的保存路径

# 加载量化模型
# GPTQModel.from_quantized用于加载已经量化的模型
print("正在加载量化模型...")
model = GPTQModel.from_quantized(
    model_path,              # 模型路径：量化模型的文件夹路径
    device_map="auto"        # 设备映射：自动分配CPU/GPU资源
)

# 加载对应的分词器
# 分词器用于将文本转换为模型可理解的token序列
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 测试推理
prompt = "你是一个专业的金融领域分析师请对以下问题进行详细解答：周三盘前交易时段，金融板块呈现温和上涨态势：金融精选行业SPDR基金(XLF)上涨0.4%，Direxion每日三倍做多金融股ETF(FAS)上涨1.2%，而其反向产品Direxion每日三倍做空金融股ETF(FAZ)下跌1.1%。万事达卡(MA)股价上涨0.7%，因其董事会批准了最高110亿美元的普通股回购计划，并将季度股息提高16%。Cboe全球市场(CBOE)股价上涨0.5%，该公司披露其指数期权合约的11月日均交易量达到近400万份，较去年同期的330万份增长22%。基于这些市场动态，请分析以下问题：在当前市场环境下，影响金融板块ETF(XLF、FAS、FAZ)价格波动的关键驱动因素有哪些？如何评估万事达卡大规模股票回购和股息增长对其长期估值的影响？Cboe交易量的大幅增长可能预示着哪些市场结构变化？请综合考虑宏观经济环境、行业竞争格局和公司特定事件等多重因素展开分析。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"输入: {prompt}")
print(f"输出: {response}")


