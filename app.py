import subprocess
import time
import uuid
import json
import asyncio
import logging
import tiktoken
import os
import random
import ipaddress
from datetime import datetime, timezone, timedelta
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Depends, Form, Cookie
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- Configuration ---
API_KEYS = json.loads(os.getenv("API_KEYS", '[]'))
ALLOWED_TOKENS = json.loads(os.getenv("ALLOWED_TOKENS", '[]'))
ALLOWED_IPS = json.loads(os.getenv("ALLOWED_IPS", '["127.0.0.1"]'))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 北京时间时区
BEIJING_TZ = timezone(timedelta(hours=8))

def get_beijing_time():
    """获取当前北京时间"""
    return datetime.now(BEIJING_TZ)

def get_last_reset_time():
    """获取最近一次的重置时间点（北京时间下午3点）"""
    now = get_beijing_time()
    reset_time_today = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    if now >= reset_time_today:
        # 如果当前时间已过今天的重置时间，返回今天的重置时间
        return reset_time_today
    else:
        # 如果当前时间未到今天的重置时间，返回昨天的重置时间
        return reset_time_today - timedelta(days=1)

# API Key 使用统计和健康状态（按模型分别管理）
api_key_stats = {
    key: {
        "total_requests": 0, 
        "daily_requests": 0,
        "last_used": None,
        "last_reset_time": get_last_reset_time().timestamp(),
        "models": {}  # 每个模型单独跟踪：requests, errors, health, daily_requests等
    } for key in API_KEYS
}

# --- Logging Setup ---
def setup_logging():
    try:
        encoding = tiktoken.get_encoding("gpt2")
    except Exception:
        encoding = None

    logger = logging.getLogger("api_logger")
    logger.setLevel(getattr(logging, LOG_LEVEL))
    logger.propagate = False

    handler = TimedRotatingFileHandler(
        "logs/api_log.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "message": record.getMessage(),
                "details": record.__dict__.get("details", {})
            }
            return json.dumps(log_record, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    
    if not logger.handlers:
        logger.addHandler(handler)
        
    return logger, encoding

api_logger, tokenizer = setup_logging()

# --- Security ---
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in ALLOWED_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid token")
    return credentials.credentials

def get_next_api_key(model: str = "gemini-2.0-flash-exp") -> str:
    """智能选择最优的 API Key，基于模型特定的使用量和健康状态"""
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys configured")
    
    # 检查是否需要重置每日计数器（基于北京时间下午3点）
    current_reset_time = get_last_reset_time().timestamp()
    for key in api_key_stats:
        if api_key_stats[key]["last_reset_time"] < current_reset_time:
            api_key_stats[key]["daily_requests"] = 0
            api_key_stats[key]["last_reset_time"] = current_reset_time
            # 重置所有模型的每日计数
            for model_name in api_key_stats[key]["models"]:
                api_key_stats[key]["models"][model_name]["daily_requests"] = 0
            
            api_logger.info(f"Reset daily counters for key {mask_api_key(key)} at Beijing time {get_beijing_time().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确保所有key都有这个模型的统计记录
    for key in API_KEYS:
        if model not in api_key_stats[key]["models"]:
            api_key_stats[key]["models"][model] = {
                "requests": 0,
                "daily_requests": 0,
                "errors": 0,
                "consecutive_errors": 0,
                "is_healthy": True,
                "last_error": None,
                "last_used": None
            }
    
    # 过滤出对此模型健康的API keys
    healthy_keys = [
        key for key in API_KEYS 
        if api_key_stats[key]["models"][model]["is_healthy"]
    ]
    
    if not healthy_keys:
        # 如果没有对此模型健康的key，重置所有key在此模型上的状态
        api_logger.warning(f"No healthy API keys for model {model}, resetting model-specific statuses")
        for key in API_KEYS:
            api_key_stats[key]["models"][model]["is_healthy"] = True
            api_key_stats[key]["models"][model]["consecutive_errors"] = 0
            api_key_stats[key]["models"][model]["last_error"] = None
        healthy_keys = API_KEYS
    
    # 选择在此模型上使用量最少的健康key
    selected_key = min(healthy_keys, key=lambda k: (
        api_key_stats[k]["models"][model]["daily_requests"],
        api_key_stats[k]["models"][model]["requests"],
        api_key_stats[k]["daily_requests"]  # 全局请求数作为次要排序
    ))
    
    api_logger.info(f"Selected API key for model {model}: {mask_api_key(selected_key)}", extra={
        "details": {
            "model": model,
            "model_daily_requests": api_key_stats[selected_key]["models"][model]["daily_requests"],
            "model_total_requests": api_key_stats[selected_key]["models"][model]["requests"],
            "global_daily_requests": api_key_stats[selected_key]["daily_requests"],
            "healthy_keys_for_model": len(healthy_keys)
        }
    })
    
    return selected_key

def mask_api_key(api_key: str) -> str:
    """安全地隐藏API key，只显示前4位和后4位"""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}****{api_key[-4:]}"

def update_api_key_stats(api_key: str, model: str, success: bool = True, error_msg: str = None):
    """更新 API Key 使用统计和健康状态（按模型分别管理）"""
    if api_key in api_key_stats:
        # 更新全局统计
        api_key_stats[api_key]["total_requests"] += 1
        api_key_stats[api_key]["daily_requests"] += 1
        api_key_stats[api_key]["last_used"] = time.time()
        
        # 确保模型统计记录存在
        if model not in api_key_stats[api_key]["models"]:
            api_key_stats[api_key]["models"][model] = {
                "requests": 0,
                "daily_requests": 0,
                "errors": 0,
                "consecutive_errors": 0,
                "is_healthy": True,
                "last_error": None,
                "last_used": None
            }
        
        # 更新模型特定统计
        model_stats = api_key_stats[api_key]["models"][model]
        model_stats["requests"] += 1
        model_stats["daily_requests"] += 1
        model_stats["last_used"] = time.time()
        
        # 处理成功/失败状态（按模型分别处理）
        if success:
            model_stats["consecutive_errors"] = 0
            model_stats["is_healthy"] = True
            api_logger.debug(f"Successful request for model {model} with key {mask_api_key(api_key)}")
        else:
            model_stats["consecutive_errors"] += 1
            model_stats["last_error"] = error_msg
            model_stats["errors"] += 1
            
            # 如果此模型连续错误超过3次，标记此模型为不健康
            if model_stats["consecutive_errors"] >= 3:
                model_stats["is_healthy"] = False
                api_logger.warning(
                    f"API key marked as unhealthy for model {model}: {mask_api_key(api_key)}",
                    extra={"details": {
                        "model": model,
                        "consecutive_errors": model_stats["consecutive_errors"],
                        "last_error": error_msg,
                        "api_key_masked": mask_api_key(api_key),
                        "model_total_requests": model_stats["requests"],
                        "model_daily_requests": model_stats["daily_requests"]
                    }}
                )
            else:
                api_logger.warning(
                    f"Error with model {model} (attempt {model_stats['consecutive_errors']}/3): {mask_api_key(api_key)}",
                    extra={"details": {
                        "model": model,
                        "consecutive_errors": model_stats["consecutive_errors"],
                        "error_message": error_msg,
                        "api_key_masked": mask_api_key(api_key)
                    }}
                )

def is_ip_allowed(client_ip: str, allowed_ips: List[str]) -> bool:
    """
    检查客户端IP是否被允许访问
    
    支持的格式：
    1. "*" 或 "all" - 允许所有IP
    2. "192.168.1.0/24" - CIDR网段格式
    3. "192.168.1.100" - 单个IP地址
    
    Args:
        client_ip: 客户端IP地址
        allowed_ips: 允许的IP列表
        
    Returns:
        bool: 是否允许访问
    """
    if not allowed_ips:
        return False
        
    # 转换客户端IP为IPv4或IPv6地址对象
    try:
        client_ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        # 如果IP格式无效，拒绝访问
        return False
    
    for allowed_ip in allowed_ips:
        # 检查是否允许所有IP
        if allowed_ip.lower() in ["*", "all"]:
            return True
            
        try:
            # 尝试解析为网段（CIDR格式）
            if "/" in allowed_ip:
                network = ipaddress.ip_network(allowed_ip, strict=False)
                if client_ip_obj in network:
                    return True
            else:
                # 尝试解析为单个IP地址
                allowed_ip_obj = ipaddress.ip_address(allowed_ip)
                if client_ip_obj == allowed_ip_obj:
                    return True
        except ValueError:
            # 忽略无效的IP或网段格式
            continue
    
    return False

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gemini-2.0-flash-exp"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class GeminiCommand(BaseModel):
    command: str
    model: Optional[str] = "gemini-2.0-flash-exp"

# --- Helper Functions ---
def run_gemini_subprocess(command: str, model: Optional[str] = None, api_key: str = None) -> Dict[str, Any]:
    """执行 Gemini CLI 命令并记录详细统计"""
    
    # 设置环境变量
    env = os.environ.copy()
    if api_key:
        env["GEMINI_API_KEY"] = api_key
    
    full_command = ["gemini"]
    if model:
        full_command.extend(["-m", model])
    full_command.extend(["-p", command])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            full_command, 
            capture_output=True, 
            text=True, 
            check=False, 
            timeout=900,
            env=env
        )
        
        execution_time = time.time() - start_time
        
        # 判断是否成功（退出码为0且没有严重错误）
        success = result.returncode == 0
        error_msg = None
        
        if not success:
            error_msg = f"Exit code: {result.returncode}, Stderr: {result.stderr}"
            # 检查是否是配额相关错误
            if "quota" in result.stderr.lower() or "limit" in result.stderr.lower():
                error_msg += " [QUOTA_EXCEEDED]"
            elif "invalid" in result.stderr.lower() and "key" in result.stderr.lower():
                error_msg += " [INVALID_KEY]"
        
        # 更新API key统计
        if api_key:
            update_api_key_stats(api_key, model or "unknown", success, error_msg)
        
        # 记录详细日志
        api_logger.info(
            f"Gemini CLI execution completed: {mask_api_key(api_key) if api_key else 'no-key'}",
            extra={"details": {
                "success": success,
                "execution_time": round(execution_time, 2),
                "exit_code": result.returncode,
                "model": model,
                "command_length": len(command),
                "response_length": len(result.stdout),
                "api_key_masked": mask_api_key(api_key) if api_key else None,
                "error_message": error_msg if not success else None
            }}
        )
        
        return {
            "stdout": result.stdout, 
            "stderr": result.stderr, 
            "exit_code": result.returncode,
            "execution_time": execution_time,
            "success": success
        }
        
    except FileNotFoundError:
        error_msg = "Gemini CLI command not found"
        if api_key:
            update_api_key_stats(api_key, model or "unknown", False, error_msg)
        raise HTTPException(status_code=500, detail="The 'gemini' command is not found.")
    except subprocess.TimeoutExpired:
        error_msg = "Command execution timeout"
        if api_key:
            update_api_key_stats(api_key, model or "unknown", False, error_msg)
        raise HTTPException(status_code=504, detail="Command timed out.")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if api_key:
            update_api_key_stats(api_key, model or "unknown", False, error_msg)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# --- FastAPI App ---
app = FastAPI(
    title="**Gemini CLI Remote API**",
    description="**An API to remotely execute Gemini CLI commands with OpenAI-compatible endpoints**",
    version="1.0.0"
)

# --- Middleware ---
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Skip IP check for health endpoint
    if request.url.path in ["/health", "/stats"]:
        return await call_next(request)
    
    # **增强的IP访问控制**
    client_ip = request.client.host
    if not is_ip_allowed(client_ip, ALLOWED_IPS):
        # 记录被拒绝的访问尝试
        api_logger.warning(
            f"Access denied for IP: {client_ip}", 
            extra={"details": {"client_ip": client_ip, "allowed_ips": ALLOWED_IPS}}
        )
        return JSONResponse(
            status_code=403, 
            content={
                "detail": f"Access denied: IP {client_ip} is not allowed.",
                "allowed_formats": [
                    "Use '*' or 'all' to allow all IPs",
                    "Use CIDR format like '192.168.1.0/24' for subnets", 
                    "Use specific IPs like '192.168.1.100'"
                ]
            }
        )
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- API Endpoints ---

@app.post("/v1/chat/completions", tags=["OpenAI-Compatible"])
async def chat_completions(request: Request, token: str = Depends(verify_token)):
    """
    **OpenAI-compatible chat completions endpoint** with API key rotation and comprehensive logging.
    """
    try:
        request_body = await request.json()
        chat_request = ChatCompletionRequest(**request_body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    # **构建完整提示**
    full_prompt_content = ""
    for message in chat_request.messages:
        full_prompt_content += f"{message.role}: {message.content}\n"

    command = full_prompt_content.strip()
    if not command:
        raise HTTPException(status_code=400, detail="No content found in messages.")

    # **获取针对此模型最优的 API Key 并执行命令**
    api_key = get_next_api_key(model=chat_request.model)
    cli_result = run_gemini_subprocess(command, model=chat_request.model, api_key=api_key)

    if cli_result["exit_code"] != 0:
        response_content = f"Error executing command.\nExit Code: {cli_result['exit_code']}\nStderr: {cli_result['stderr']}"
    else:
        response_content = cli_result["stdout"]

    # **Token 计算**
    prompt_tokens = len(tokenizer.encode(command)) if tokenizer else 0
    completion_tokens = len(tokenizer.encode(response_content)) if tokenizer else 0
    total_tokens = prompt_tokens + completion_tokens
    
    usage_stats = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    # **日志记录**
    log_details = {
        "client_ip": request.client.host,
        "token_used": token[:8] + "..." if len(token) > 8 else "*" * len(token),
        "api_key_used": mask_api_key(api_key),
        "request_model": chat_request.model,
        "prompt_length": len(command),
        "response_length": len(response_content),
        "usage": usage_stats,
        "is_stream": chat_request.stream,
        "exit_code": cli_result["exit_code"],
        "execution_time": cli_result.get("execution_time", 0),
        "success": cli_result.get("success", True)
    }
    api_logger.info("Chat completion request processed", extra={"details": log_details})

    # **流式响应生成器**
    async def stream_generator():
        chunk_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())
        
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': chat_request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': response_content}, 'finish_reason': None}]})}\n\n"
        await asyncio.sleep(0.01)
        
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': chat_request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage_stats})}\n\n"
        yield "data: [DONE]\n\n"

    if chat_request.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_content}, "finish_reason": "stop"}],
            "usage": usage_stats,
        }

@app.post("/run-gemini/", tags=["Direct Gemini"])
async def run_gemini_command(gemini_command: GeminiCommand, token: str = Depends(verify_token)):
    """
    **Direct Gemini CLI command execution** with model-specific API key selection and intelligent load balancing.
    """
    api_key = get_next_api_key(model=gemini_command.model)
    cli_result = run_gemini_subprocess(gemini_command.command, model=gemini_command.model, api_key=api_key)
    
    return {
        "command_executed": f"gemini -m {gemini_command.model} -p '{gemini_command.command}'",
        "stdout": cli_result["stdout"],
        "stderr": cli_result["stderr"],
        "exit_code": cli_result["exit_code"],
        "execution_time": cli_result.get("execution_time", 0),
        "success": cli_result.get("success", True),
        "api_key_used": mask_api_key(api_key)
    }

@app.get("/stats", tags=["Management"])
async def get_api_stats():
    """
    **Get comprehensive API key usage statistics and health status by model**.
    """
    # 创建安全的统计信息（隐藏真实API key）
    safe_stats = {}
    model_summary = {}
    
    for key, stats in api_key_stats.items():
        safe_key = mask_api_key(key)
        
        # 处理模型统计信息，添加安全的时间格式化
        safe_models = {}
        for model, model_stats in stats["models"].items():
            safe_models[model] = {
                "requests": model_stats["requests"],
                "daily_requests": model_stats["daily_requests"],
                "errors": model_stats["errors"],
                "consecutive_errors": model_stats["consecutive_errors"],
                "is_healthy": model_stats["is_healthy"],
                "last_used": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(model_stats["last_used"])) if model_stats["last_used"] else "Never",
                "last_error": model_stats["last_error"],
                "error_rate": round(model_stats["errors"] / max(model_stats["requests"], 1) * 100, 2)
            }
            
            # 汇总模型统计
            if model not in model_summary:
                model_summary[model] = {
                    "total_requests": 0,
                    "total_daily_requests": 0,
                    "total_errors": 0,
                    "healthy_keys": 0,
                    "total_keys": 0
                }
            
            model_summary[model]["total_requests"] += model_stats["requests"]
            model_summary[model]["total_daily_requests"] += model_stats["daily_requests"]
            model_summary[model]["total_errors"] += model_stats["errors"]
            model_summary[model]["total_keys"] += 1
            if model_stats["is_healthy"]:
                model_summary[model]["healthy_keys"] += 1
        
        safe_stats[safe_key] = {
            "total_requests": stats["total_requests"],
            "daily_requests": stats["daily_requests"],
            "last_used": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats["last_used"])) if stats["last_used"] else "Never",
            "last_reset_time": datetime.fromtimestamp(stats["last_reset_time"], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S") + " (BJT)",
            "next_reset_time": (get_last_reset_time() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S") + " (BJT)",
            "models": safe_models
        }
    
    # 计算整体统计
    total_requests = sum(stats["total_requests"] for stats in api_key_stats.values())
    total_daily_requests = sum(stats["daily_requests"] for stats in api_key_stats.values())
    
    # 计算每个模型的错误率
    for model in model_summary:
        model_summary[model]["error_rate"] = round(
            model_summary[model]["total_errors"] / max(model_summary[model]["total_requests"], 1) * 100, 2
        )
    
    return {
        "api_key_stats": safe_stats,
        "model_summary": model_summary,
        "global_summary": {
            "total_api_keys": len(API_KEYS),
            "total_requests_all_time": total_requests,
            "total_requests_today": total_daily_requests,
            "total_allowed_tokens": len(ALLOWED_TOKENS),
            "models_in_use": len(model_summary)
        },
        "load_balancing": {
            "strategy": "Model-specific least used healthy key first",
            "health_check": "3 consecutive errors per model mark key as unhealthy for that model",
            "daily_reset": "Counters reset at 15:00 Beijing Time (GMT+8)",
            "model_isolation": "Each model has independent quota and health tracking",
            "current_beijing_time": get_beijing_time().strftime("%Y-%m-%d %H:%M:%S") + " (BJT)"
        }
    }

@app.get("/health", tags=["Management"])
async def health_check():
    """
    **Health check endpoint**.
    """
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "gemini_cli_available": subprocess.run(["which", "gemini"], capture_output=True).returncode == 0
    }

@app.get("/ip-config", tags=["Management"])
async def get_ip_config():
    """
    **查看当前IP访问控制配置**.
    """
    return {
        "allowed_ips": ALLOWED_IPS,
        "ip_rules_explanation": {
            "*_or_all": "允许所有IP访问",
            "cidr_format": "如 '192.168.1.0/24' 表示允许整个网段",
            "single_ip": "如 '192.168.1.100' 表示允许单个IP",
            "examples": [
                "['*'] - 允许所有IP",
                "['192.168.1.0/24'] - 只允许192.168.1.x网段",
                "['127.0.0.1', '192.168.1.100'] - 只允许指定的IP",
                "['192.168.1.0/24', '10.0.0.1'] - 允许网段和单个IP混合"
            ]
        }
    }

@app.post("/test-ip", tags=["Management"])
async def test_ip_access(request: Request):
    """
    **测试指定IP是否被允许访问**.
    
    请求体格式: {"test_ip": "192.168.1.100"}
    """
    try:
        request_body = await request.json()
        test_ip = request_body.get("test_ip")
        
        if not test_ip:
            raise HTTPException(status_code=400, detail="请提供要测试的IP地址")
        
        is_allowed = is_ip_allowed(test_ip, ALLOWED_IPS)
        
        return {
            "test_ip": test_ip,
            "is_allowed": is_allowed,
            "current_client_ip": request.client.host,
            "allowed_ips_config": ALLOWED_IPS,
            "message": f"IP {test_ip} {'允许' if is_allowed else '拒绝'} 访问"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"测试失败: {str(e)}")

@app.post("/reset-api-key-health", tags=["Management"])
async def reset_api_key_health(request: Request):
    """
    **重置API Key健康状态（支持按模型分别重置）**.
    
    请求体格式: 
    - {"reset_all": true} - 重置所有key的所有模型
    - {"api_key_pattern": "AIza****abcd"} - 重置特定key的所有模型
    - {"api_key_pattern": "AIza****abcd", "model": "gemini-2.0-flash-exp"} - 重置特定key的特定模型
    - {"model": "gemini-1.5-pro"} - 重置所有key的特定模型
    """
    try:
        request_body = await request.json()
        api_key_pattern = request_body.get("api_key_pattern")
        model = request_body.get("model")
        reset_all = request_body.get("reset_all", False)
        
        reset_count = 0
        
        if reset_all:
            # 重置所有API key的所有模型
            for key in api_key_stats:
                for model_name in api_key_stats[key]["models"]:
                    api_key_stats[key]["models"][model_name]["is_healthy"] = True
                    api_key_stats[key]["models"][model_name]["consecutive_errors"] = 0
                    api_key_stats[key]["models"][model_name]["last_error"] = None
                    reset_count += 1
            
            api_logger.info("All API keys and models health status reset", extra={
                "details": {"client_ip": request.client.host, "action": "reset_all_keys_all_models"}
            })
            
            return {"message": "所有API Key的所有模型健康状态已重置", "reset_count": reset_count}
            
        elif api_key_pattern and model:
            # 重置特定API key的特定模型
            for key in api_key_stats:
                if mask_api_key(key) == api_key_pattern:
                    if model in api_key_stats[key]["models"]:
                        api_key_stats[key]["models"][model]["is_healthy"] = True
                        api_key_stats[key]["models"][model]["consecutive_errors"] = 0
                        api_key_stats[key]["models"][model]["last_error"] = None
                        reset_count += 1
            
            if reset_count == 0:
                raise HTTPException(status_code=404, detail="未找到匹配的API Key或模型")
            
            api_logger.info(f"API key {api_key_pattern} model {model} health reset", extra={
                "details": {"client_ip": request.client.host, "api_key_pattern": api_key_pattern, "model": model}
            })
            
            return {"message": f"API Key {api_key_pattern} 的模型 {model} 健康状态已重置", "reset_count": reset_count}
            
        elif api_key_pattern:
            # 重置特定API key的所有模型
            for key in api_key_stats:
                if mask_api_key(key) == api_key_pattern:
                    for model_name in api_key_stats[key]["models"]:
                        api_key_stats[key]["models"][model_name]["is_healthy"] = True
                        api_key_stats[key]["models"][model_name]["consecutive_errors"] = 0
                        api_key_stats[key]["models"][model_name]["last_error"] = None
                        reset_count += 1
            
            if reset_count == 0:
                raise HTTPException(status_code=404, detail="未找到匹配的API Key")
            
            api_logger.info(f"API key {api_key_pattern} all models health reset", extra={
                "details": {"client_ip": request.client.host, "api_key_pattern": api_key_pattern}
            })
            
            return {"message": f"API Key {api_key_pattern} 的所有模型健康状态已重置", "reset_count": reset_count}
            
        elif model:
            # 重置所有API key的特定模型
            for key in api_key_stats:
                if model in api_key_stats[key]["models"]:
                    api_key_stats[key]["models"][model]["is_healthy"] = True
                    api_key_stats[key]["models"][model]["consecutive_errors"] = 0
                    api_key_stats[key]["models"][model]["last_error"] = None
                    reset_count += 1
            
            if reset_count == 0:
                raise HTTPException(status_code=404, detail="未找到该模型的使用记录")
            
            api_logger.info(f"All keys for model {model} health reset", extra={
                "details": {"client_ip": request.client.host, "model": model}
            })
            
            return {"message": f"所有API Key的模型 {model} 健康状态已重置", "reset_count": reset_count}
        else:
            raise HTTPException(status_code=400, detail="请提供有效的重置参数")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"重置失败: {str(e)}")

@app.get("/api-key-recommendations", tags=["Management"])
async def get_api_key_recommendations():
    """
    **获取API Key使用建议和免费配额管理策略**.
    
    解决配额检查本身消耗配额的问题。
    """
    recommendations = {
        "quota_management_strategies": {
            "避免直接检查": {
                "description": "不要频繁调用API来检查剩余配额，因为检查本身也会消耗配额",
                "solution": "通过监控API响应错误来判断配额状态"
            },
            "错误代码监控": {
                "description": "监控特定的错误响应来判断配额耗尽",
                "quota_exceeded_indicators": [
                    "HTTP 429 状态码",
                    "错误信息包含 'quota', 'limit', 'exceeded'",
                    "连续失败的请求"
                ]
            },
            "智能负载均衡": {
                "description": "当前系统已实现的按模型分别管理功能",
                "features": [
                    "按模型优先使用请求次数最少的key",
                    "每个模型独立的健康状态和配额跟踪",
                    "自动标记在特定模型上连续失败的key为不健康",
                    "每日重置计数器避免长期偏向",
                    "模型间互不影响的配额管理"
                ]
            },
            "按模型配额管理": {
                "description": "每个API key在不同模型上有独立的配额限制",
                "benefits": [
                    "避免因一个模型配额耗尽影响其他模型",
                    "更精确的配额分配和使用统计",
                    "独立的健康状态监控和故障恢复"
                ],
                "strategies": [
                    "监控每个key在每个模型上的使用情况",
                    "当特定模型配额耗尽时，仅在该模型上标记为不健康",
                    "其他模型可继续正常使用该key"
                ]
            },
            "预防性管理": {
                "description": "通过使用模式预估配额消耗",
                "recommendations": [
                    "设置每个key的每日请求上限",
                    "在高峰期间轮换使用key",
                    "监控每个key的错误率"
                ]
            }
        },
        "current_system_status": {
            "total_keys": len(API_KEYS),
            "today_total_requests": sum(stats["daily_requests"] for stats in api_key_stats.values()),
            "models_in_use": len(set(model for stats in api_key_stats.values() for model in stats["models"].keys())),
            "model_health_status": {
                model: {
                    "healthy_keys": sum(1 for stats in api_key_stats.values() 
                                      if model in stats["models"] and stats["models"][model]["is_healthy"]),
                    "total_keys_for_model": sum(1 for stats in api_key_stats.values() if model in stats["models"]),
                    "today_requests": sum(stats["models"][model]["daily_requests"] 
                                        for stats in api_key_stats.values() if model in stats["models"]),
                    "error_rate": round(sum(stats["models"][model]["errors"] for stats in api_key_stats.values() if model in stats["models"]) / 
                                      max(sum(stats["models"][model]["requests"] for stats in api_key_stats.values() if model in stats["models"]), 1) * 100, 2)
                }
                for model in set(model for stats in api_key_stats.values() for model in stats["models"].keys())
            }
        },
        "best_practices": {
            "配额监控": [
                "通过日志分析识别配额耗尽模式",
                "设置告警当错误率超过阈值",
                "定期检查API key健康状态"
            ],
            "Key管理": [
                "准备多个备用API key",
                "不要将所有请求集中在单个key上",
                "定期轮换和更新API key"
            ],
            "错误处理": [
                "实现指数退避重试机制",
                "自动切换到备用key",
                "记录详细的错误信息用于分析"
            ]
        }
    }
    
    return recommendations

@app.get("/login", response_class=HTMLResponse, tags=["Dashboard"])
async def get_login_page(request: Request):
    """显示仪表板登录页面"""
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>登录 - Gemini API Dashboard</title>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f0f2f5; margin: 0; }
                .login-card { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); text-align: center; width: 320px; }
                h2 { margin-top: 0; color: #333; }
                p { color: #666; margin-bottom: 20px; }
                input { width: 100%; padding: 12px; margin-bottom: 20px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
                button { width: 100%; padding: 12px; border: none; border-radius: 6px; background-color: #007bff; color: white; font-size: 16px; cursor: pointer; }
                button:hover { background-color: #0056b3; }
                .error { color: #d93025; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="login-card">
                <h2>Dashboard 登录</h2>
                <p>请输入您的访问令牌</p>
                <form action="/auth" method="post">
                    <input type="password" name="token" placeholder="访问令牌" required>
                    <button type="submit">登 录</button>
                </form>
            </div>
        </body>
        </html>
    """)

@app.post("/auth", tags=["Dashboard"])
async def handle_login(token: str = Form(...)):
    """处理登录请求并设置cookie"""
    if token in ALLOWED_TOKENS:
        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(key="session_token", value=token, httponly=True, max_age=86400, samesite="strict") # 24 hours
        return response
    else:
        # 在实际应用中，您可能会重定向回登录页面并显示错误
        raise HTTPException(status_code=401, detail="无效的访问令牌")

@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard(session_token: Optional[str] = Cookie(None)):
    """API Key使用情况可视化仪表板（需要Cookie认证）"""
    if session_token not in ALLOWED_TOKENS:
        return RedirectResponse(url="/login", status_code=307)
    
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini API Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <style>
        body { font-family: 'Segoe UI',-apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif; background-color: #f8f9fa; color: #343a40; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .header h1 { font-size: 2rem; color: #212529; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .card h3 { margin-top: 0; border-bottom: 1px solid #dee2e6; padding-bottom: 10px; margin-bottom: 15px; font-size: 1.1rem; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; text-align: center; }
        .stat-item .value { font-size: 1.75rem; font-weight: 600; color: #007bff; }
        .stat-item .label { font-size: 0.9rem; color: #6c757d; }
        .key-list { max-height: 500px; overflow-y: auto; padding-right: 10px; }
        .key-item { background: #fdfdff; border: 1px solid #e9ecef; border-left: 4px solid #007bff; border-radius: 6px; padding: 15px; margin-bottom: 10px; }
        .key-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-weight: 600; }
        .health-badge { padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; color: white; }
        .health-healthy { background-color: #28a745; }
        .health-unhealthy { background-color: #dc3545; }
        .model-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; margin-top: 10px; font-size: 0.85rem; }
        .model-item { background: #f8f9fa; padding: 8px; border-radius: 4px; }
        .model-name { font-weight: 600; }
        .refresh-btn { position: fixed; bottom: 20px; right: 20px; background: #007bff; color: white; border: none; width: 50px; height: 50px; border-radius: 50%; cursor: pointer; font-size: 1.5rem; box-shadow: 0 4px 10px rgba(0,0,0,0.1); transition: transform 0.2s; display: flex; align-items: center; justify-content: center; }
        .refresh-btn:hover { background-color: #0056b3; transform: scale(1.1); }
        .chart-container { height: 300px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>Gemini API Dashboard</h1></div>
        <div class="grid">
            <div class="card">
                <h3>概览</h3>
                <div class="stat-grid" id="globalStats"></div>
            </div>
            <div class="card">
                <h3>今日模型使用分布</h3>
                <div class="chart-container"><canvas id="modelChart"></canvas></div>
            </div>
            <div class="card">
                <h3>API Key 详细信息</h3>
                <div class="key-list" id="keyList"></div>
            </div>
             <div class="card">
                <h3>重置时间信息</h3>
                 <div class="stat-grid" id="timeInfo"></div>
            </div>
        </div>
    </div>
    <button class="refresh-btn" onclick="loadData()">🔄</button>
    <script>
        let modelChart;
        async function loadData() {
            try {
                const response = await fetch('/stats');
                if (!response.ok) { window.location.href = '/login'; return; }
                const data = await response.json();
                updateGlobalStats(data);
                updateCharts(data);
                updateKeyList(data.api_key_stats);
            } catch (error) { console.error('加载数据失败:', error); }
        }
        function updateGlobalStats(data) {
            const summary = data.global_summary;
            document.getElementById('globalStats').innerHTML = `
                <div class="stat-item"><div class="value">${summary.total_api_keys}</div><div class="label">总 Keys</div></div>
                <div class="stat-item"><div class="value">${summary.models_in_use}</div><div class="label">使用中模型</div></div>
                <div class="stat-item"><div class="value">${summary.total_requests_today}</div><div class="label">今日请求</div></div>
                <div class="stat-item"><div class="value">${summary.total_requests_all_time}</div><div class="label">累计请求</div></div>`;

            const keys = Object.keys(data.api_key_stats);
            const timeInfo = keys.length > 0 ? `
                <div class="stat-item"><div class="value" style="font-size: 1rem;">${data.load_balancing.current_beijing_time}</div><div class="label">当前北京时间</div></div>
                <div class="stat-item"><div class="value" style="font-size: 1rem;">${data.api_key_stats[keys[0]].next_reset_time}</div><div class="label">下次重置时间</div></div>` : '';
            document.getElementById('timeInfo').innerHTML = timeInfo;
        }
        function updateCharts(data) {
            const modelLabels = Object.keys(data.model_summary);
            const modelData = modelLabels.map(model => data.model_summary[model].total_daily_requests);
            if (modelChart) modelChart.destroy();
            modelChart = new Chart(document.getElementById('modelChart').getContext('2d'), {
                type: 'doughnut',
                data: { labels: modelLabels, datasets: [{ data: modelData, backgroundColor: ['#007bff', '#28a745', '#ffc107', '#17a2b8', '#6f42c1', '#fd7e14'] }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right' } } }
            });
        }
        function updateKeyList(keyStats) {
            const container = document.getElementById('keyList');
            container.innerHTML = Object.entries(keyStats).map(([keyName, stats]) => {
                const overallHealthy = Object.values(stats.models).some(model => model.is_healthy);
                return `
                <div class="key-item">
                    <div class="key-header"><span>${keyName}</span><span class="health-badge ${overallHealthy ? 'health-healthy' : 'health-unhealthy'}">${overallHealthy ? '可用' : '部分不可用'}</span></div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-bottom: 10px;">今日请求: ${stats.daily_requests} | 累计: ${stats.total_requests} | 最后使用: ${stats.last_used}</div>
                    <div class="model-stats">
                        ${Object.entries(stats.models).map(([modelName, modelStats]) => `
                            <div class="model-item">
                                <div class="model-name">${modelName} ${modelStats.is_healthy ? '✅' : '❌'}</div>
                                <div>今日: ${modelStats.daily_requests} | 累计: ${modelStats.requests}</div>
                            </div>`).join('')}
                    </div>
                </div>`;
            }).join('');
        }
        loadData();
        setInterval(loadData, 30000);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "**Gemini CLI API is running**",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats",
        "dashboard": "/login"
    }

