#!/bin/bash

echo "🚀 开始同步项目到服务器..."

rsync -avz \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  ./ \
  ubuntu@117.50.34.28:/shared/financial_reward_model/

echo "✅ 文件同步完成，执行远程操作..."

ssh ubuntu@117.50.34.28 << 'EOF'
cd /shared/financial_reward_model
# 如果有需要重启的服务或运行命令，请在下面加
# 示例: pkill -f app.py && nohup python3 app.py > out.log 2>&1 &
echo "🚀 已进入远程项目目录，可根据需要添加执行命令"
EOF

echo "🎉 部署完毕！"