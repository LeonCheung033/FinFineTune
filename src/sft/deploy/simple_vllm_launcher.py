#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版vLLM模型启动器
这个脚本的作用是：启动两个大语言模型服务，让它们可以通过网络接口提供文本生成服务

什么是vLLM？
- vLLM是一个高性能的大语言模型推理引擎
- 它可以把训练好的大模型（如Qwen、ChatGLM等）变成可以通过网络访问的API服务
- 就像把一个聪明的AI助手部署到服务器上，让其他程序可以调用它

这个脚本的作用：
1. 读取配置文件，了解要启动哪些模型
2. 使用vLLM把模型启动成网络服务
3. 管理这些服务（启动、停止、查看状态）
4. 让其他应用（如Dify）可以通过HTTP接口使用这些模型
"""

import os           # 用于操作系统相关功能，比如设置环境变量
import sys          # 用于获取Python解释器路径
import time         # 用于时间相关操作，比如等待
import json         # 用于处理JSON格式的配置文件
import signal       # 用于处理进程信号，比如停止进程
import subprocess   # 用于启动和管理子进程
import psutil       # 用于检查进程状态
from pathlib import Path  # 用于处理文件路径

class SimpleVLLMLauncher:
    """
    简化版vLLM启动器类
    
    这个类就像一个"模型服务管理员"，它的主要职责是：
    1. 读取配置文件，了解要启动哪些模型
    2. 启动模型服务进程（把模型变成可以网络访问的服务）
    3. 管理这些进程（启动、停止、查看状态）
    4. 持久化保存进程信息，重启后也能管理之前的进程
    
    比喻：就像一个餐厅经理，负责开启多个厨房（模型服务），
    每个厨房都有自己的菜谱（模型）和服务窗口（端口），
    顾客（其他应用）可以通过窗口点菜（发送请求）
    """
    
    def __init__(self):
        """
        初始化启动器
        创建一个空的字典来存储正在运行的进程信息
        
        就像准备一个记录本，用来记录所有正在运行的模型服务
        """
        self.processes = {}  # 存储所有启动的模型进程信息，格式：{模型名: 进程信息}
        self.service_info_file = "running_services.json"  # 保存运行中服务信息的文件
        # 启动时尝试加载之前保存的服务信息
        self.load_existing_services()
    
    def load_existing_services(self):
        """
        加载之前保存的服务信息
        这样即使重启启动器，也能管理之前启动的服务
        
        比喻：就像餐厅经理上班时，先查看昨天的记录，
        看看哪些厨房还在营业，哪些已经关门了
        """
        if Path(self.service_info_file).exists():
            try:
                # 读取之前保存的服务信息文件
                with open(self.service_info_file, 'r', encoding='utf-8') as f:
                    service_info = json.load(f)
                
                print("检查之前启动的服务...")
                for model_name, info in service_info.items():
                    pid = info.get('pid')  # 获取进程ID
                    if pid and self.is_process_running(pid):
                        # 如果进程还在运行，重新创建进程对象
                        try:
                            process = psutil.Process(pid)
                            self.processes[model_name] = {
                                'process': process,
                                'config': {
                                    'name': model_name,
                                    'port': info['port'],
                                    'gpu_devices': info.get('gpu_devices', [])
                                },
                                'log_file': info.get('log_file', f"{model_name}_service.log"),
                                'pid': pid
                            }
                            print(f"✓ 发现运行中的服务: {model_name} (PID: {pid})")
                        except:
                            pass
                    else:
                        print(f"✗ 服务 {model_name} 已停止 (PID: {pid})")
                
            except Exception as e:
                print(f"加载服务信息时出错: {e}")
    
    def is_process_running(self, pid):
        """
        检查指定PID的进程是否还在运行
        
        参数说明：
        pid: 进程ID（每个运行的程序都有一个唯一的ID号）
        
        返回值：
        True: 进程还在运行
        False: 进程已停止
        
        比喻：就像查看某个员工是否还在工作岗位上
        """
        try:
            return psutil.pid_exists(pid)
        except:
            return False
    
    def check_gpu_status(self):
        """
        检查GPU状态
        这个方法会调用nvidia-smi命令来查看有哪些GPU可用
        
        GPU（显卡）是运行大模型的重要硬件，就像厨房的炉灶，
        需要确认有多少个炉灶可用，每个炉灶的规格如何
        """
        try:
            print("正在检查GPU状态...")
            # 运行nvidia-smi命令获取GPU信息
            # nvidia-smi是NVIDIA显卡的管理工具，可以查看显卡状态
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("可用的GPU:")
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        gpu_id = parts[0]      # GPU编号
                        gpu_name = parts[1]    # GPU名称
                        gpu_memory = parts[2]  # GPU内存大小
                        print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory}MB)")
            else:
                print("无法获取GPU信息，请确保安装了NVIDIA驱动")
        except Exception as e:
            print(f"检查GPU状态时出错: {e}")
    
    def check_port_in_use(self, port):
        """
        检查端口是否被占用
        
        参数说明：
        port: 要检查的端口号（端口就像门牌号，每个网络服务都需要一个唯一的端口）
        
        返回值：
        True: 端口被占用（已经有其他服务在使用）
        False: 端口可用（可以使用这个端口）
        
        比喻：就像检查某个房间号是否已经有人入住
        """
        try:
            import socket
            # 尝试连接到指定端口，如果能连接成功说明端口被占用
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                return result == 0  # 0表示连接成功，即端口被占用
        except:
            return False
    
    def start_single_model(self, model_config):
        """
        启动单个模型服务
        
        参数说明：
        model_config: 字典，包含模型的配置信息
        - name: 模型名称（比如"original"或"finetuned"）
        - path: 模型文件路径（模型文件存放的位置）
        - port: 服务端口号（这个服务将在哪个端口提供服务）
        - gpu_devices: 使用的GPU设备列表（使用哪些显卡）
        - 其他配置参数
        
        比喻：就像开启一个新的厨房，需要指定：
        - 厨房名称
        - 菜谱位置（模型路径）
        - 服务窗口号（端口）
        - 使用哪些炉灶（GPU）
        """
        # 从配置中提取信息
        model_name = model_config['name']           # 模型名称
        model_path = model_config['path']           # 模型文件路径
        port = model_config['port']                 # 服务端口
        gpu_devices = model_config.get('gpu_devices', [0])  # 使用的GPU，默认使用GPU 0
        
        print(f"\n开始启动模型: {model_name}")
        print(f"模型路径: {model_path}")
        print(f"服务端口: {port}")
        print(f"使用GPU: {gpu_devices}")
        
        # 检查端口是否已被占用
        if self.check_port_in_use(port):
            print(f"警告: 端口 {port} 已被占用，可能该模型已在运行")
            return None
        
        # 检查模型路径是否存在
        if not Path(model_path).exists():
            print(f"错误: 模型路径不存在 - {model_path}")
            return None
        
        # 设置要使用的GPU
        gpu_str = ','.join(map(str, gpu_devices))  # 将GPU列表转换为字符串，比如[0,1]变成"0,1"
        tensor_parallel_size = len(gpu_devices)    # 并行度等于GPU数量
        
        # 构建启动命令
        # 这个命令会启动vLLM的OpenAI兼容API服务器
        # 就像告诉厨师："用这个菜谱，在这个窗口，用这些炉灶开始营业"
        cmd = [
            sys.executable,                          # Python解释器路径
            '-m', 'vllm.entrypoints.openai.api_server',  # vLLM的API服务器模块
            '--model', model_path,                   # 指定模型路径
            '--port', str(port),                     # 指定服务端口
            '--tensor-parallel-size', str(tensor_parallel_size),  # 设置并行度（使用多少个GPU）
            '--gpu-memory-utilization', str(model_config.get('gpu_memory_utilization', 0.85)),  # GPU内存使用率
            '--max-model-len', str(model_config.get('max_model_len', 4096)),  # 最大序列长度
            '--trust-remote-code',                   # 信任模型代码
            '--disable-log-requests',                # 禁用请求日志（减少输出）
        ]
        
        # 如果配置中指定了数据类型，添加到命令中
        if model_config.get('dtype'):
            cmd.extend(['--dtype', model_config['dtype']])
        
        # 设置环境变量
        env = os.environ.copy()  # 复制当前环境变量
        env['CUDA_VISIBLE_DEVICES'] = gpu_str  # 设置可见的GPU设备
        
        # 创建日志文件
        log_file = f"{model_name}_service.log"
        print(f"日志文件: {log_file}")
        
        try:
            # 启动进程
            # 就像正式开启厨房营业，所有的操作记录都会写入日志文件
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,                    # 要执行的命令
                    env=env,               # 环境变量
                    stdout=f,              # 标准输出重定向到日志文件
                    stderr=subprocess.STDOUT,  # 错误输出也重定向到日志文件
                    preexec_fn=os.setsid   # 创建新的进程组（方便后续管理）
                )
            
            # 保存进程信息
            self.processes[model_name] = {
                'process': process,      # 进程对象
                'config': model_config,  # 配置信息
                'log_file': log_file,    # 日志文件路径
                'pid': process.pid       # 进程ID
            }
            
            print(f"✓ 模型 {model_name} 启动成功，进程ID: {process.pid}")
            return process
            
        except Exception as e:
            print(f"✗ 启动模型 {model_name} 失败: {e}")
            return None
    
    def wait_for_service_ready(self, port, timeout=300):
        """
        等待服务启动完成
        
        参数说明：
        port: 服务端口号
        timeout: 超时时间（秒），默认5分钟
        
        返回值：
        True: 服务启动成功
        False: 服务启动失败或超时
        
        比喻：就像等待厨房准备完毕，可以开始接受订单
        大模型启动需要时间加载到内存中，这个过程可能需要几分钟
        """
        print(f"等待端口 {port} 上的服务启动...")
        
        # 尝试导入requests库，用于发送HTTP请求
        try:
            import requests
        except ImportError:
            print("需要安装requests库: pip install requests")
            return False
        
        start_time = time.time()  # 记录开始时间
        
        # 在超时时间内不断尝试连接服务
        while time.time() - start_time < timeout:
            try:
                # 尝试访问服务的模型列表接口
                # 这是vLLM提供的标准接口，如果能正常访问说明服务已就绪
                response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                if response.status_code == 200:
                    print(f"✓ 端口 {port} 上的服务已就绪")
                    return True
            except:
                # 如果连接失败，继续等待
                pass
            
            # 每5秒检查一次
            time.sleep(5)
            print(".", end="", flush=True)  # 显示等待进度
        
        print(f"\n✗ 端口 {port} 上的服务启动超时")
        return False
    
    def start_all_models(self, config_file="vllm_config.json"):
        """
        启动所有配置的模型
        
        参数说明：
        config_file: 配置文件路径
        
        比喻：就像同时开启餐厅的所有厨房，每个厨房都有不同的菜谱和服务窗口
        """
        print("=" * 60)
        print("开始启动vLLM模型服务")
        print("=" * 60)
        
        # 检查配置文件是否存在
        if not Path(config_file).exists():
            print(f"错误: 配置文件 {config_file} 不存在")
            print("请先创建配置文件")
            return False
        
        # 读取配置文件
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"错误: 无法读取配置文件 - {e}")
            return False
        
        # 检查GPU状态
        self.check_gpu_status()
        
        # 获取要启动的模型列表
        models = config.get('models', [])
        if not models:
            print("错误: 配置文件中没有找到模型配置")
            return False
        
        print(f"\n准备启动 {len(models)} 个模型...")
        
        # 逐个启动模型
        for i, model_config in enumerate(models, 1):
            model_name = model_config.get('name', f'model_{i}')
            
            # 检查模型是否已经在运行
            if model_name in self.processes:
                print(f"\n[{i}/{len(models)}] 模型 {model_name} 已在运行，跳过启动")
                continue
            
            print(f"\n[{i}/{len(models)}] 启动模型...")
            
            # 启动单个模型
            process = self.start_single_model(model_config)
            
            if process:
                # 如果不是最后一个模型，等待一段时间再启动下一个
                # 这样可以避免GPU资源冲突
                if i < len(models):
                    print("等待10秒后启动下一个模型...")
                    time.sleep(10)
            else:
                print(f"模型启动失败，跳过")
        
        # 等待所有服务启动完成
        print(f"\n等待所有服务启动完成...")
        all_ready = True
        
        for model_name, info in self.processes.items():
            port = info['config']['port']
            if self.wait_for_service_ready(port):
                print(f"✓ {model_name} 服务就绪")
            else:
                print(f"✗ {model_name} 服务启动失败")
                all_ready = False
        
        # 保存服务信息到文件
        self.save_service_info()
        
        if all_ready:
            print(f"\n🎉 所有模型服务启动成功！")
            self.show_running_services()
            return True
        else:
            print(f"\n⚠️  部分模型服务启动失败，请检查日志")
            return False
    
    def stop_all_models(self):
        """
        停止所有运行中的模型服务
        
        比喻：就像关闭餐厅的所有厨房，让所有厨师下班
        """
        print("正在停止所有模型服务...")
        
        if not self.processes:
            print("没有运行中的模型服务")
            return
        
        # 逐个停止进程
        for model_name, info in list(self.processes.items()):
            print(f"停止 {model_name}...")
            
            try:
                pid = info['pid']
                
                # 使用psutil来停止进程
                if self.is_process_running(pid):
                    process = psutil.Process(pid)
                    # 停止进程及其子进程
                    children = process.children(recursive=True)
                    for child in children:
                        try:
                            child.terminate()
                        except:
                            pass
                    
                    process.terminate()
                    
                    # 等待进程结束
                    try:
                        process.wait(timeout=30)
                        print(f"✓ {model_name} 已停止")
                    except psutil.TimeoutExpired:
                        # 如果30秒内没有结束，强制杀死进程
                        process.kill()
                        print(f"✓ {model_name} 已强制停止")
                else:
                    print(f"✓ {model_name} 进程已不存在")
                    
            except Exception as e:
                print(f"✗ 停止 {model_name} 时出错: {e}")
            
            # 从进程列表中移除
            del self.processes[model_name]
        
        # 清空服务信息文件
        self.save_service_info()
        print("所有模型服务已停止")
    
    def show_running_services(self):
        """
        显示当前运行中的服务状态
        
        比喻：就像查看餐厅状态报告，看看哪些厨房在营业，
        每个厨房的窗口号是多少，使用了哪些设备
        """
        if not self.processes:
            print("没有运行中的模型服务")
            return
        
        print("\n" + "=" * 80)
        print("运行中的模型服务:")
        print("=" * 80)
        
        for model_name, info in self.processes.items():
            config = info['config']
            pid = info['pid']
            
            # 检查进程是否还在运行
            if self.is_process_running(pid):
                status = "🟢 运行中"
            else:
                status = "🔴 已停止"
            
            print(f"\n模型名称: {model_name}")
            print(f"  状态: {status}")
            print(f"  进程ID: {pid}")
            print(f"  服务端口: {config['port']}")
            print(f"  使用GPU: {config.get('gpu_devices', [])}")
            print(f"  API地址: http://localhost:{config['port']}")
            print(f"  日志文件: {info['log_file']}")
    
    def save_service_info(self):
        """
        保存服务信息到文件
        这个文件会被对比测试脚本读取，用于知道哪些服务在运行
        
        比喻：就像把餐厅的营业状态记录在册，
        这样其他人（比如外卖平台）就知道哪些厨房在营业，可以接单
        """
        service_info = {}
        
        for model_name, info in self.processes.items():
            service_info[model_name] = {
                'port': info['config']['port'],
                'pid': info['pid'],
                'gpu_devices': info['config'].get('gpu_devices', []),
                'log_file': info['log_file']
            }
        
        # 写入文件
        with open(self.service_info_file, 'w', encoding='utf-8') as f:
            json.dump(service_info, f, indent=2, ensure_ascii=False)

def main():
    """
    主函数 - 程序的入口点
    
    这是程序开始执行的地方，就像餐厅的总经理，
    根据用户的指令决定要执行什么操作
    """
    import argparse
    
    # 创建命令行参数解析器
    # 这让用户可以通过命令行告诉程序要做什么
    parser = argparse.ArgumentParser(description='简化版vLLM模型启动器')
    parser.add_argument('action', 
                       choices=['start', 'stop', 'status'], 
                       help='要执行的操作: start(启动), stop(停止), status(查看状态)')
    parser.add_argument('--config', 
                       type=str, 
                       default='vllm_config.json',
                       help='配置文件路径 (默认: vllm_config.json)')
    
    args = parser.parse_args()
    
    # 创建启动器实例
    launcher = SimpleVLLMLauncher()
    
    # 根据用户指定的操作执行相应功能
    if args.action == 'start':
        # 启动所有模型
        launcher.start_all_models(args.config)
        
    elif args.action == 'stop':
        # 停止所有模型
        launcher.stop_all_models()
        
    elif args.action == 'status':
        # 显示服务状态
        launcher.show_running_services()

# 当直接运行这个脚本时，执行main函数
# 这是Python的标准写法，确保只有直接运行脚本时才执行main函数
if __name__ == "__main__":
    main()
