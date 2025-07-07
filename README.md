# Gemini CLI API Docker Container

一个集成了 Gemini CLI 的 Docker 容器，提供 OpenAI 兼容的 REST API，支持多 API Key 轮询和令牌认证。

---

## ✨ 功能特点

- ✅ **OpenAI 兼容接口**（`/v1/chat/completions`）
- 🔧 **直接 Gemini 命令执行**（`/run-gemini/`）
- 🔁 **API Key 自动轮询**，实现负载均衡
- 🔐 **令牌认证机制**
- 🛡️ **IP 白名单控制**，增强安全性
- 📄 **详细的请求日志**
- ❤️ **容器健康检查**
- 📊 **API 使用统计**

---

## ⚙️ 安装要求

- Docker
- Docker Compose
- 你自己的 Gemini API Key（支持多个）

---

## 🚀 快速开始

### 1. 克隆此仓库

```bash
git clone https://github.com/dreamzyd/gemini-cli-api.git
cd gemini-cli-api
```

### 2. 复制并编辑环境配置文件

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的 API Key 与令牌。

### 3. 构建并启动容器

```bash
docker compose up -d
```

### 4. 测试服务运行状态

```bash
curl -X GET http://localhost:8080/health
```

---

## 🔌 API 端点

### ✅ Chat Completions（OpenAI 兼容）

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "gemini-2.0-flash-exp",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about quantum computing."}
    ]
  }'
```

---

### 🧠 直接执行 Gemini 命令

```bash
curl -X POST http://localhost:8080/run-gemini \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "command": "Tell me about quantum computing.",
    "model": "gemini-2.0-flash-exp"
  }'
```

---

### 📊 API 使用统计

```bash
curl -X GET http://localhost:8080/stats
```

---

## ⚙️ 配置项说明

编辑 `.env` 文件，支持以下变量：

| 变量名           | 描述                        |
|------------------|-----------------------------|
| `API_KEYS`       | Gemini API 密钥列表（JSON 数组） |
| `ALLOWED_TOKENS` | 允许访问的 Bearer Token（数组）  |
| `ALLOWED_IPS`    | 允许的 IP 地址（数组）           |
| `LOG_LEVEL`      | 日志级别：DEBUG / INFO / WARNING / ERROR |

---

## 📁 日志

日志将存储在 `./logs` 目录下，并按天轮换，保留最近 30 天的记录。

---

## 📘 使用说明摘要

1. 创建 `.env` 文件：

```bash
cp .env.example .env
```

2. 编辑 `.env`，填写：

- Gemini API 密钥（多个）
- 授权 Token
- IP 白名单（如有）

3. 启动服务：

```bash
docker compose up -d
```

4. 测试接口：

```bash
curl http://localhost:8080/health
```

5. 查看日志：

```bash
docker compose logs -f gemini-cli-api
```

---

## 📄 许可证 License

[MIT License](LICENSE)

