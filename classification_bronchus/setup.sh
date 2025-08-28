#!/bin/bash

# 支气管分类项目环境配置脚本

echo "🚀 支气管分类项目环境配置"
echo "=========================="

# 检查Python版本
echo "📋 检查Python版本..."
python --version

# 检查pip版本
echo "📋 检查pip版本..."
pip --version

# 安装依赖
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 创建必要目录
echo "📁 创建项目目录..."
mkdir -p ../data/data_split/{train,val,test}
mkdir -p ../weight
mkdir -p ../logs

# 运行环境测试
echo "🔍 运行环境测试..."
python test_environment.py

echo ""
echo "✅ 环境配置完成！"
echo ""
echo "📝 下一步操作："
echo "1. 将您的数据放入 data/data_split/ 目录"
echo "2. 根据需要修改 code/config.py"
echo "3. 运行 cd code && python train.py 开始训练"
echo ""
echo "📖 更多信息请查看 README.md"