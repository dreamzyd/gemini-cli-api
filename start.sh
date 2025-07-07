#!/bin/bash

# 确保日志目录存在
mkdir -p logs

# **激活虚拟环境并启动 FastAPI 应用**
source /app/venv/bin/activate
exec /app/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

