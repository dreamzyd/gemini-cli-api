# Gemini CLI API Docker Container

一个集成了 Gemini CLI 的 Docker 容器，提供 OpenAI 兼容的 REST API，支持多 API Key 轮询和令牌认证。

---

## ✨ 功能特点

- ✅ **OpenAI 兼容接口**（`/v1/chat/completions`）
- 🔧 **直接 Gemini 命令执行**（`/run-gemini/`）
- 🔁 **按模型分别管理的智能负载均衡**，每个模型独立配额跟踪
- 🔐 **令牌认证机制**
- 🛡️ **增强的IP访问控制**，支持CIDR网段、单IP、通配符
- 📄 **安全的详细日志记录**，API Key自动掩码保护
- ❤️ **API Key健康状态监控**，自动检测和处理故障key
- 📊 **按模型分别的详细统计和分析**
- 🚦 **配额耗尽智能检测**，避免浪费请求检查配额
- ⚡ **每日请求计数重置**，确保负载均衡

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

### 📊 API 使用统计和健康状态

```bash
curl -X GET http://localhost:8080/stats
```

### 🔧 API Key 管理

#### 重置API Key健康状态（支持按模型分别重置）

```bash
# 重置所有API Key的所有模型
curl -X POST http://localhost:8080/reset-api-key-health \
  -H "Content-Type: application/json" \
  -d '{"reset_all": true}'

# 重置特定API Key的所有模型
curl -X POST http://localhost:8080/reset-api-key-health \
  -H "Content-Type: application/json" \
  -d '{"api_key_pattern": "AIza****abcd"}'

# 重置特定API Key的特定模型
curl -X POST http://localhost:8080/reset-api-key-health \
  -H "Content-Type: application/json" \
  -d '{"api_key_pattern": "AIza****abcd", "model": "gemini-2.0-flash-exp"}'

# 重置所有API Key的特定模型
curl -X POST http://localhost:8080/reset-api-key-health \
  -H "Content-Type: application/json" \
  -d '{"model": "gemini-1.5-pro"}'
```

#### 获取配额管理建议

```bash
curl -X GET http://localhost:8080/api-key-recommendations
```

---

## ⚙️ 配置项说明

编辑 `.env` 文件，支持以下变量：

| 变量名           | 描述                        |
|------------------|-----------------------------|
| `API_KEYS`       | Gemini API 密钥列表（JSON 数组） |
| `ALLOWED_TOKENS` | 允许访问的 Bearer Token（数组）  |
| `ALLOWED_IPS`    | 允许的 IP 地址（数组，支持多种格式） |
| `LOG_LEVEL`      | 日志级别：DEBUG / INFO / WARNING / ERROR |

---

## 🛡️ IP 访问控制（新功能）

### 支持的IP配置格式

`ALLOWED_IPS` 现在支持多种灵活的配置格式：

1. **允许所有IP**：
   ```bash
   ALLOWED_IPS=["*"]
   # 或者
   ALLOWED_IPS=["all"]
   ```

2. **网段限制（CIDR格式）**：
   ```bash
   ALLOWED_IPS=["192.168.1.0/24"]        # 允许 192.168.1.x 网段
   ALLOWED_IPS=["10.0.0.0/8"]            # 允许 10.x.x.x 网段  
   ALLOWED_IPS=["172.16.0.0/12"]         # 允许 172.16.x.x - 172.31.x.x 网段
   ```

3. **单个IP地址**：
   ```bash
   ALLOWED_IPS=["127.0.0.1"]             # 只允许本地访问
   ALLOWED_IPS=["192.168.1.100"]         # 只允许特定IP
   ```

4. **混合配置**：
   ```bash
   ALLOWED_IPS=["192.168.1.0/24", "10.0.0.1", "172.16.1.100"]
   ```

### IP管理端点

#### 查看当前IP配置

```bash
curl -X GET http://localhost:8080/ip-config
```

#### 测试IP是否被允许

```bash
curl -X POST http://localhost:8080/test-ip \
  -H "Content-Type: application/json" \
  -d '{"test_ip": "192.168.1.100"}'
```

### 配置示例

```bash
# .env 文件示例

# 允许所有IP访问（开发环境）
ALLOWED_IPS=["*"]

# 只允许内网访问（生产环境）
ALLOWED_IPS=["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"]

# 只允许特定IP（高安全环境）
ALLOWED_IPS=["192.168.1.100", "192.168.1.101"]

# 混合配置（推荐）
ALLOWED_IPS=["127.0.0.1", "192.168.1.0/24", "10.0.0.50"]
```

---

## 🔧 智能API Key管理（新功能）

### 负载均衡策略（按模型分别管理）

系统现在采用**按模型分别管理**的智能负载均衡策略：

1. **模型独立配额管理**：每个API key在不同模型上有独立的配额和健康状态
2. **按模型选择最优key**：为每个模型请求选择使用次数最少的健康key
3. **独立健康状态监控**：在某个模型上连续3次错误后，仅在该模型上标记为不健康
4. **模型间互不影响**：一个模型配额耗尽不会影响其他模型的使用
5. **每日计数重置**：每个模型的使用计数独立重置
6. **智能故障切换**：当key在某个模型上不健康时，自动切换到该模型的备用key

### 🔥 关键优势
- ✅ **精确配额控制**：`gemini-2.0-flash-exp` 配额耗尽不影响 `gemini-1.5-pro` 的使用
- ✅ **独立健康监控**：每个模型单独跟踪错误和健康状态  
- ✅ **最大化可用性**：充分利用每个key在不同模型上的配额

### 📋 实际使用场景

假设你有一个API key `AIza****abcd`：

```json
{
  "AIza****abcd": {
    "models": {
      "gemini-2.0-flash-exp": {
        "daily_requests": 95,  // 接近配额上限
        "is_healthy": false,   // 因配额问题标记为不健康
        "consecutive_errors": 3
      },
      "gemini-1.5-pro": {
        "daily_requests": 23,  // 还有充足配额
        "is_healthy": true,    // 在此模型上仍然健康
        "consecutive_errors": 0
      }
    }
  }
}
```

**系统行为**：
- 对 `gemini-2.0-flash-exp` 的请求会自动切换到其他健康的key
- 对 `gemini-1.5-pro` 的请求仍可正常使用这个key
- 每个模型的配额独立管理，互不影响

### 配额管理最佳实践

针对"检查配额也消耗配额"的困惑，系统提供以下解决方案：

#### ❌ 不推荐的做法
```bash
# 不要这样做 - 频繁检查配额会浪费配额
while true; do
  check_quota_api_call  # 这本身就消耗配额！
done
```

#### ✅ 推荐的智能方案
1. **通过错误监控判断配额状态**：
   - 监控HTTP 429状态码
   - 检查错误信息中的关键字：`quota`, `limit`, `exceeded`
   - 分析连续失败模式

2. **被动配额管理**：
   - 系统自动记录每个key的使用次数
   - 当key出现配额错误时自动切换到备用key
   - 通过日志分析识别使用模式

3. **预防性策略**：
   - 准备多个API key作为备用
   - 设置每日使用量上限提醒
   - 定期检查系统健康状态

### 安全的日志记录

- ✅ **API Key安全掩码**：只显示前4位和后4位（如：`AIza****abcd`）
- ✅ **访问令牌保护**：只显示前8位加省略号
- ✅ **详细的执行统计**：包含执行时间、成功率等
- ✅ **错误分类标记**：自动识别配额、无效key等错误类型

---

## 📁 日志

日志将存储在 `./logs` 目录下，并按天轮换，保留最近 30 天的记录。所有敏感信息（API Key、访问令牌）都会被自动掩码保护。

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

