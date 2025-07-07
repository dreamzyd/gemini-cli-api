FROM node:18-alpine

# 安装系统依赖
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-virtualenv \
    bash \
    curl \
    git \
    gcc \
    musl-dev \
    python3-dev

# 创建工作目录
WORKDIR /app

# **全局安装 Gemini CLI**
RUN npm install -g @google/gemini-cli

# **创建 Python 虚拟环境**
RUN python3 -m venv /app/venv

# **激活虚拟环境并安装 Python 依赖**
COPY requirements.txt .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# 复制应用文件
COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

# 创建日志目录
RUN mkdir -p logs

# 暴露端口
EXPOSE 8000

# 启动命令 - 使用虚拟环境中的 Python
CMD ["./start.sh"]

