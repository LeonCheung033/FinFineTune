#!/usr/bin/env python3

import requests
import json
import sys

# 设置API服务的地址
# 这个地址应该与启动的Ray Serve服务地址一致
api_url = "http://localhost:8000"

# 定义一个简单的测试用例
# 这个测试用例包含了奖励模型推理所需的三个基本字段
simple_test = {
    "question": "Hello",      # 用户问题
    "chosen": "Hi there!",    # 人类偏好的回答（应该得到更高分数）
    "rejected": "No."         # 人类不偏好的回答（应该得到更低分数）
}

print("发送简单测试请求...")
try:
    # 发送POST请求到推理服务
    # timeout=30表示30秒超时
    response = requests.post(f"{api_url}/", json=simple_test, timeout=30)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    # 检查HTTP状态码是否为200（成功）
    if response.status_code == 200:
        try:
            # 尝试解析JSON响应
            result = response.json()
            print("JSON解析成功:")
            # 格式化输出JSON结果，便于阅读
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
    else:
        print(f"HTTP错误: {response.status_code}")
        
except Exception as e:
    # 捕获所有异常，包括网络连接错误、超时等
    print(f"请求失败: {e}")
    sys.exit(1)