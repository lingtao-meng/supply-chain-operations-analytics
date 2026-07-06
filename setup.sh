#!/bin/bash
# 供应链运营分析平台 — 一键初始化
set -e
echo "📦 供应链运营分析平台 — 初始化"
echo "================================="

# 安装依赖
echo "[1/3] 安装Python依赖..."
pip install -r requirements.txt -q

# 数据清洗
echo "[2/3] 数据清洗 & 入库..."
python3 python/01_data_cleaning.py
python3 python/02_load_to_db.py

# 完成
echo "[3/3] 启动仪表板..."
echo ""
echo "✅ 初始化完成！"
echo "   运行: streamlit run app/app.py"
echo "   或直接: bash setup.sh && streamlit run app/app.py"
