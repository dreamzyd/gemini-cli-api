# Gemini CLI API

基于Gemini CLI的Docker容器，提供OpenAI兼容的REST API接口，用于测试和开发目的。支持API Key轮询、IP控制和按模型分别管理配额。

## 功能特点

- OpenAI兼容接口 (`/v1/chat/completions`)
- 按模型分别管理的API Key负载均衡
- IP访问控制（支持CIDR网段、单IP、通配符）
- 安全的日志记录，敏感信息自动掩码
- 详细的API使用统计和健康状态监控

## 安装步骤

### 要求
- Docker
- Docker Compose
- Gemini API Key

### 快速开始

1. **克隆仓库**
```bash
git clone https://github.com/dreamzyd/gemini-cli-api.git
cd gemini-cli-api
```

2. **配置环境**
```bash
cp .env.example .env
# 编辑.env文件，添加API Key和访问控制设置
```

3. **启动服务**
```bash
docker compose up -d
```

4. **测试服务**
```bash
curl http://localhost:8080/health
```

## API端点

### OpenAI兼容接口

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "gemini-2.0-flash-exp",
    "messages": [
      {"role": "user", "content": "Tell me about quantum computing."}
    ]
  }'
```

### 直接调用Gemini

```bash
curl -X POST http://localhost:8080/run-gemini \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "command": "Tell me about quantum computing.",
    "model": "gemini-2.0-flash-exp"
  }'
```

### 查看API使用统计

```bash
curl -X GET http://localhost:8080/stats
```

### 可视化监控面板

1. **访问登录页面**
   ```
   http://localhost:8080/login
   ```

2. **输入访问令牌**
   在登录页面输入您在`.env`文件中配置的`ALLOWED_TOKENS`之一。

3. **查看仪表板**
   登录成功后，您将被重定向到仪表板页面。仪表板提供以下功能：
   - 实时API Key使用统计
   - 按模型分别的使用情况图表
   - API Key健康状态监控
   - 自动30秒刷新数据

## 配置选项

编辑`.env`文件配置以下选项：

| 配置项 | 描述 | 示例 |
|-------|------|-----|
| `API_KEYS` | Gemini API Key列表 (JSON数组) | `["key1", "key2"]` |
| `ALLOWED_TOKENS` | 允许的访问令牌 (JSON数组) | `["token1", "token2"]` |
| `ALLOWED_IPS` | 允许的IP地址 (JSON数组) | `["127.0.0.1", "192.168.1.0/24"]` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## IP访问控制

支持多种IP控制格式:

```bash
# 允许所有IP
ALLOWED_IPS=["*"]

# 允许特定网段
ALLOWED_IPS=["192.168.1.0/24"]

# 允许特定IP
ALLOWED_IPS=["127.0.0.1", "192.168.1.100"]

# 混合模式
ALLOWED_IPS=["127.0.0.1", "192.168.1.0/24", "10.0.0.50"]
```

## API Key管理

系统按模型分别管理API Key，每个模型有独立的配额和健康状态跟踪。

**重要**：配额重置时间为**北京时间每天下午3点**（GMT+8 15:00），与Gemini API官方重置时间同步。

### 健康状态管理

```bash
# 重置所有Key的所有模型
curl -X POST http://localhost:8080/reset-api-key-health \
  -H "Content-Type: application/json" \
  -d '{"reset_all": true}'

# 重置特定模型
curl -X POST http://localhost:8080/reset-api-key-health \
  -H "Content-Type: application/json" \
  -d '{"model": "gemini-1.5-pro"}'
```

### 获取配额管理建议

```bash
curl -X GET http://localhost:8080/api-key-recommendations
```

## 日志

日志存储在`./logs`目录，按天轮换，保留30天。所有API Key和令牌均自动掩码保护。

## 许可证

[MIT License](LICENSE)

