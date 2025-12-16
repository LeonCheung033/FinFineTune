#!/bin/bash

# ==================== 项目部署脚本 ====================
# 作用：将本地项目文件同步到远程服务器，用于A800 GPU服务器部署
# 使用方法：./deploy_A800.sh

echo "🚀 开始同步项目到服务器..."

# 使用rsync命令进行文件同步
# -a：归档模式，保持文件属性
# -v：详细输出，显示同步过程
# -z：压缩传输，节省带宽
# --exclude：排除不需要同步的文件和目录
rsync -avz --progress \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".DS_Store" \
  ./ \
  ubuntu@117.50.194.2:/shared/gptq_model/

echo "✅ 文件同步完成，执行远程操作..."

# 通过SSH连接到远程服务器并执行命令
# << 'EOF' 表示开始多行命令输入，直到遇到EOF结束
ssh ubuntu@117.50.194.2 << 'EOF'
cd /shared/gptq_model    # 切换到项目目录
chmod +x vllm_run.sh           # 给训练脚本添加执行权限
# 如果有需要重启的服务或运行命令，请在下面加
# 示例: pkill -f app.py && nohup python3 app.py > out.log 2>&1 &
echo "🚀 已进入远程项目目录，可根据需要添加执行命令"
EOF

echo "🎉 部署完毕！"