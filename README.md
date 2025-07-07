# Gemini CLI API Docker Container

一个集成了 Gemini CLI 的 Docker 容器，提供 OpenAI 兼容的 REST API，支持多 API Key 轮询和令牌认证。

## 功能特点

- **OpenAI 兼容接口** (`/v1/chat/completions`)
- **直接 Gemini 命令执行** (`/run-gemini/`)
- **API Key 自动轮询**，实现负载均衡
- **令牌认证机制**
- **IP 白名单**，增强安全性
- **详细的请求日志**
- **容器健康检查**
- **API 使用统计**

## 安装要求

- Docker
- Docker Compose
- Gemini API key(s)

## 快速开始

1. 克隆此仓库:

bash
git clone https://github.com/dreamzyd/gemini-cli-api.git
cd gemini-cli-api 


2. 复制并编辑环境配置文件

cp .env.example .env

3. 构建并启动容器:

docker compose up -d

4. 测试 API:

curl -X GET http://localhost:8080/health

## API 端点

### Chat Completions (OpenAI-兼容)

curl -X POST http://localhost:8080/v1/chat/completions

-H "Content-Type: application/json"

-H "Authorization: Bearer YOUR_TOKEN"

-d '{
"model": "gemini-2.0-flash-exp",
"messages": [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "Tell me about quantum computing."}
]
}'


### 直接 Gemini 命令

curl -X POST http://localhost:8080/run-gemini

-H "Content-Type: application/json"

-H "Authorization: Bearer YOUR_TOKEN"

-d '{
"command": "Tell me about quantum computing.",
"model": "gemini-2.0-flash-exp"
}'


### API 使用统计

curl -X GET http://localhost:8080/stats

## 配置

编辑 `.env` 文件配置:

- `API_KEYS`: Gemini API 密钥 JSON 数组
- `ALLOWED_TOKENS`: 允许的认证令牌 JSON 数组
- `ALLOWED_IPS`: 允许的 IP 地址 JSON 数组
- `LOG_LEVEL`: 日志级别 (DEBUG, INFO, WARNING, ERROR)

## 日志

日志存储在 `./logs` 目录，按天轮换。


5. 使用说明
部署后，你可以通过以下方式使用:

创建环境文件:
cp .env.example .env
编辑 .env 文件:

添加你的 Gemini API 密钥
设置允许的访问令牌
配置 IP 白名单
启动服务:

docker compose up -d
测试服务:
curl http://localhost:8080/health
查看日志:
docker compose logs -f gemini-cli-api

## 许可证

MIT
