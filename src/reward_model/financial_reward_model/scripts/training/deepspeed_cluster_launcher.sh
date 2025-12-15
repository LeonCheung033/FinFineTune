#!/bin/bash

# ================================================================================
#  文件作用说明：
# 这是DeepSpeed多机多卡分布式训练的主启动脚本
# 
#  项目中的整体作用：
# 1. 作为分布式训练的入口点，负责启动多台服务器上的训练进程
# 2. 自动管理集群节点配置，创建DeepSpeed需要的hostfile文件
# 3. 提供训练进程的启动、停止、日志查看等管理功能
# 4. 这是你在多机训练时实际使用的主要脚本
#
#  使用方式：
# bash deepspeed_cluster_launcher.sh start  # 启动训练
# bash deepspeed_cluster_launcher.sh stop   # 停止训练  
# bash deepspeed_cluster_launcher.sh logs   # 查看日志
# ================================================================================

# 修正版DeepSpeed多机多卡启动脚本
# 解决GPU设备编号问题

# ============================================================================
#  颜色定义：用于在终端中显示不同颜色的日志信息，提高可读性
# ============================================================================
RED='\033[0;31m'      # 红色：用于错误信息
GREEN='\033[0;32m'    # 绿色：用于成功信息  
BLUE='\033[0;34m'     # 蓝色：用于一般信息
YELLOW='\033[1;33m'   # 黄色：用于调试信息
NC='\033[0m'          # 无颜色：重置颜色

# ============================================================================
#  日志输出函数：统一管理日志格式，方便调试和监控
# 这些函数在整个项目中用于：输出格式化的日志信息，便于运维人员查看
# ============================================================================
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }      # 输出蓝色的信息日志
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; } # 输出绿色的成功日志
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }     # 输出红色的错误日志  
log_debug() { echo -e "${YELLOW}[DEBUG]${NC} $1"; }  # 输出黄色的调试日志

# ============================================================================
#  集群节点配置：定义所有参与训练的服务器节点和GPU数量
# 
#  项目中的整体作用：
# 1. 这是分布式训练的核心配置，告诉DeepSpeed有哪些服务器可以用
# 2. 每个IP对应一台服务器，数字表示该服务器有多少张GPU卡
# 3. DeepSpeed会根据这个配置自动分配训练任务到各个GPU上
# 4. 如果要增加/减少训练节点，只需要修改这个数组即可
# ============================================================================
declare -A NODES=(
    ["10.60.11.131"]=4    # 主节点（服务器1）：拥有4张GPU卡
    ["10.60.240.249"]=2   # 工作节点2：拥有2张GPU卡
    # ["10.60.79.49"]=2    # 工作节点3：拥有2张GPU卡（已注释，表示暂时不使用）
    # ["10.60.11.133"]=2    # 工作节点4：拥有2张GPU卡（已注释，表示暂时不使用）
)

# 主节点（第一个节点）：负责协调整个训练过程的服务器
MASTER_ADDR="10.60.11.131"

# ============================================================================
#  创建hostfile函数：生成DeepSpeed分布式训练需要的节点配置文件
# 
#  项目中的整体作用：
# 1. DeepSpeed需要一个hostfile文件来知道有哪些服务器参与训练
# 2. 这个文件告诉DeepSpeed每台服务器的IP地址和GPU数量
# 3. 训练开始前必须先创建这个文件，否则DeepSpeed无法启动
# 4. 每次启动训练都会重新生成，确保配置是最新的
# ============================================================================
create_hostfile() {
    log_info "创建DeepSpeed hostfile..."
    
    # 清空现有hostfile文件（如果存在的话）
    > /shared/financial_reward_model/hostfile
    
    # 遍历所有配置的节点，将每个节点信息写入hostfile
    for node_ip in "${!NODES[@]}"; do
        gpu_count=${NODES[$node_ip]}  # 获取该节点的GPU数量
        # 按DeepSpeed要求的格式写入：IP地址 slots=GPU数量
        echo "$node_ip slots=$gpu_count" >> /shared/financial_reward_model/hostfile
    done
    
    log_success "Hostfile创建完成"
    # 显示创建的hostfile内容，便于确认配置正确
    cat /shared/financial_reward_model/hostfile
}

# ============================================================================
#  停止训练函数：安全地停止所有服务器上正在运行的训练进程
# 
#  项目中的整体作用：
# 1. 在启动新训练前，先停止可能存在的旧训练进程，避免冲突
# 2. 确保所有服务器上的训练进程都被正确清理
# 3. 防止出现多个训练进程同时运行的问题
# 4. 为新的训练启动做准备
# ============================================================================
stop_all_training() {
    log_info " 停止所有节点的训练进程..."
    
    # 遍历所有配置的节点
    for node_ip in "${!NODES[@]}"; do
        if [[ "$node_ip" == "$MASTER_ADDR" ]]; then
            # 如果是主节点，直接在本地执行停止命令
            pkill -f "deepspeed.*train_reward_model" 2>/dev/null || true
        else
            # 如果是其他节点，通过SSH远程执行停止命令
            # 使用&符号让SSH命令在后台并行执行，提高效率
            ssh ubuntu@$node_ip "pkill -f 'deepspeed.*train_reward_model' 2>/dev/null || true" &
        fi
    done
    
    wait  # 等待所有SSH命令完成
    sleep 3  # 稍等片刻，确保进程完全停止
    log_success "所有节点训练进程清理完成"
}

# ============================================================================
#  启动训练函数：这是整个脚本的核心功能，负责启动分布式训练
# 
#  项目中的整体作用：
# 1. 这是分布式训练的主入口，协调整个训练流程的启动
# 2. 按顺序执行：停止旧进程 → 创建配置文件 → 激活环境 → 启动训练
# 3. 负责管理训练日志的保存和进程监控
# 4. 确保训练能够正确启动并提供监控信息
# ============================================================================
start_training() {
    log_info " 启动DeepSpeed分布式训练..."
    
    # 第1步：清理环境，停止可能存在的旧训练进程
    stop_all_training
    
    # 第2步：创建DeepSpeed需要的hostfile配置文件
    create_hostfile
    
    # 第3步：切换到项目根目录
    cd /shared/financial_reward_model
    
    # 第4步：创建日志存储目录
    mkdir -p logs/distributed
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # 生成时间戳，用于日志文件命名
    
    # 第5步：激活conda虚拟环境（包含所需的Python包）
    log_info "激活conda环境..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reward2  # 激活名为"reward"的conda环境
    
    log_info "使用修正的配置启动多机训练..."
    
    # 🔑 支持额外参数的启动命令
    EXTRA_ARGS="$2"  # 接收额外参数
    if [[ -n "$EXTRA_ARGS" ]]; then
        LAUNCH_CMD="deepspeed --hostfile=hostfile src/train_reward_model.py $EXTRA_ARGS"
    else
        LAUNCH_CMD="deepspeed --hostfile=hostfile src/train_reward_model.py"
    fi
    LOG_FILE="logs/distributed/deepspeed_final_${TIMESTAMP}.log"  # 日志文件路径
    
    log_debug "启动命令: $LAUNCH_CMD"
    log_debug "日志文件: $LOG_FILE"
    
    # 第7步：启动训练进程
    # > $LOG_FILE 2>&1 将标准输出和错误输出都重定向到日志文件
    # & 让训练在后台运行，不阻塞当前终端
    $LAUNCH_CMD > $LOG_FILE 2>&1 &
    TRAIN_PID=$!  # 获取训练进程的PID
    
    log_success "训练启动完成，PID: $TRAIN_PID"
    
    # 第8步：等待并检查训练是否成功启动
    sleep 15  # 等待15秒，让训练进程有时间初始化
    
    # 检查训练进程是否还在运行
    if ps -p $TRAIN_PID > /dev/null; then
        log_success "训练进程运行中"
        log_info "日志文件: $LOG_FILE"
        log_info "监控命令: tail -f -n 300 $LOG_FILE"
        
        # 显示训练日志的前30行，便于快速确认启动状态
        sleep 5
        echo "=== 修正版训练日志前30行 ==="
        head -30 $LOG_FILE
    else
        # 如果进程已经退出，说明启动失败
        log_error "训练进程已退出"
        echo "=== 错误日志 ==="
        cat $LOG_FILE  # 显示完整的错误日志
        return 1
    fi
    
    return 0
}

# ============================================================================
#  主函数：根据用户输入的参数执行相应的操作
# 
# 项目中的整体作用：
# 1. 这是脚本的入口点，解析用户的命令行参数
# 2. 提供统一的界面来管理分布式训练的各种操作
# 3. 支持start（启动）、stop（停止）、logs（查看日志）三种操作
# 4. 让运维人员可以方便地管理整个训练流程
# ============================================================================
case "$1" in
    "start")
        start_training "$@"  # 传递所有参数
        ;;
    "resume")
        #  新增resume命令
        if [[ -z "$2" ]]; then
            log_error "请指定checkpoint路径"
            echo "用法: $0 resume <checkpoint_path>"
            exit 1
        fi
        start_training "start" "--resume_from_checkpoint $2"
        ;;
    "stop")
        # 停止训练：调用stop_all_training函数
        stop_all_training
        ;;
    "logs")
        # 查看日志：找到最新的日志文件并实时显示
        LATEST_LOG=$(ls -t /shared/financial_reward_model/logs/distributed/deepspeed_final_*.log 2>/dev/null | head -1)
        if [[ -n "$LATEST_LOG" ]]; then
            tail -f -n 300 "$LATEST_LOG"  # 实时显示日志内容
        else
            # 如果没找到deepspeed_final开头的日志，尝试查找deepspeed开头的
            LATEST_LOG=$(ls -t /shared/financial_reward_model/logs/distributed/deepspeed_*.log 2>/dev/null | head -1)
            if [[ -n "$LATEST_LOG" ]]; then
                tail -f "$LATEST_LOG"
            else
                log_error "未找到日志文件"
            fi
        fi
        ;;
    *)
        echo "修正版DeepSpeed多机多卡训练"
        echo "用法: $0 {start|stop|logs|resume}"
        echo ""
        echo "  start  - 启动新训练"
        echo "  stop   - 停止训练"
        echo "  logs   - 监控日志"
        echo "  resume <checkpoint> - 从checkpoint恢复训练"
        ;;
esac
