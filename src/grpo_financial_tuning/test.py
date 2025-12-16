# ==================== 模型测试脚本 ====================
# 作用：加载训练好的GRPO模型并进行推理测试
# 用途：验证模型是否正常工作，测试模型的推理能力

import torch                                    # PyTorch深度学习框架
from transformers import AutoTokenizer, AutoModelForCausalLM  # Hugging Face模型和分词器

def test_model():
    """
    测试已训练的GRPO模型
    
    功能：
    1. 加载保存的模型和分词器
    2. 使用金融领域的复杂问题进行推理测试
    3. 验证模型的文本生成能力
    """
    # 设置模型路径 - 可以是本地路径或远程模型路径
    model_path = "./output/best_model"  # 默认使用本地最佳模型
    # model_path = "/shared/DeepSeek-R1-Distill-Qwen-7B_028/best_complete_model_05261653_028"  # 备用路径
    
    print("加载分词器...")
    # 加载分词器 - 负责将文本转换为模型可理解的数字序列
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,                    # 模型路径
        trust_remote_code=True         # 允许执行远程代码（某些模型需要）
    )
    print("分词器加载成功")
    
    print("加载模型...")
    # 加载语言模型 - 用于文本生成
    model = AutoModelForCausalLM.from_pretrained(
        model_path,                    # 模型路径
        torch_dtype=torch.bfloat16,    # 使用bfloat16精度节省显存
        device_map="auto",             # 自动分配设备（GPU/CPU）
        trust_remote_code=True         # 允许执行远程代码
    )
    print("模型加载成功")
    
    print("测试推理...")
    # 构造测试用的金融问题 - 这是一个复杂的金融分析问题
    prompt = "你是一个专业的金融领域分析师请对以下问题进行详细解答：在周二的交易中，First Trust纳斯达克网络安全ETF(CIBR)表现优于其他ETF，当日上涨约1.3%。该ETF中表现尤为强劲的成分股包括Sentinelone(上涨约5.3%)和Okta(上涨约3.8%)。与此同时，Invesco太阳能ETF(TAN)表现逊于其他ETF，周二下午交易时段下跌约3.7%。该ETF中表现最弱的成分股包括Maxeon Solar Technologies(下跌约7.8%)和Solaredge Technologies(下跌约6.7%)。基于这些市场表现数据，请分析：1) 网络安全和太阳能行业ETF表现差异可能反映出的宏观经济或行业特定因素；2) 成分股价格变动与ETF整体表现之间的传导机制；3) 这种行业间表现差异可能对投资者资产配置策略产生的影响；4) 如何利用ETF成分股的价格离散度来构建潜在的投资组合策略。请详细说明你的分析框架和逻辑推理过程。"
    
    # 将文本转换为模型输入格式
    inputs = tokenizer.encode(
        prompt,                        # 输入文本
        return_tensors="pt"           # 返回PyTorch张量格式
    ).to(model.device)                # 移动到模型所在设备
    
    # 使用模型生成回复
    with torch.no_grad():             # 禁用梯度计算，节省内存
        outputs = model.generate(
            inputs,                    # 输入序列
            max_new_tokens=2048,       # 最大生成长度
            temperature=0.7,           # 控制生成的随机性（0-1，越高越随机）
            do_sample=True,            # 启用采样生成
            pad_token_id=tokenizer.eos_token_id  # 设置填充token
        )
    
    # 将生成的数字序列转换回文本
    result = tokenizer.decode(
        outputs[0],                   # 取第一个生成结果
        skip_special_tokens=True      # 跳过特殊标记
    )
    
    print("推理成功")
    print("结果:", result)
    print("测试完成，模型正常工作")

# 程序入口点
if __name__ == "__main__":
    test_model()  # 执行模型测试