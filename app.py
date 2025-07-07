import subprocess
import time
import uuid
import json
import asyncio
import logging
import tiktoken
import os
import random
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
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

# API Key 使用统计
api_key_stats = {key: {"total_requests": 0, "models": {}} for key in API_KEYS}

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

def get_next_api_key() -> str:
    """轮询获取下一个可用的 API Key"""
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No API keys configured")
    return random.choice(API_KEYS)

def update_api_key_stats(api_key: str, model: str):
    """更新 API Key 使用统计"""
    if api_key in api_key_stats:
        api_key_stats[api_key]["total_requests"] += 1
        if model not in api_key_stats[api_key]["models"]:
            api_key_stats[api_key]["models"][model] = 0
        api_key_stats[api_key]["models"][model] += 1

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
    """执行 Gemini CLI 命令"""
    
    # 设置环境变量
    env = os.environ.copy()
    if api_key:
        env["GEMINI_API_KEY"] = api_key
    
    full_command = ["gemini"]
    if model:
        full_command.extend(["-m", model])
    full_command.extend(["-p", command])
    
    try:
        result = subprocess.run(
            full_command, 
            capture_output=True, 
            text=True, 
            check=False, 
            timeout=900,
            env=env
        )
        return {
            "stdout": result.stdout, 
            "stderr": result.stderr, 
            "exit_code": result.returncode
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="The 'gemini' command is not found.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Command timed out.")
    except Exception as e:
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
    
    # **IP Whitelisting**
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        return JSONResponse(
            status_code=403, 
            content={"detail": f"Access denied: IP {client_ip} is not allowed."}
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

    # **获取 API Key 并执行命令**
    api_key = get_next_api_key()
    cli_result = run_gemini_subprocess(command, model=chat_request.model, api_key=api_key)

    # **更新统计信息**
    update_api_key_stats(api_key, chat_request.model)

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
        "token_used": token[:10] + "...",
        "api_key_used": api_key[:10] + "...",
        "request_model": chat_request.model,
        "prompt_length": len(command),
        "response_length": len(response_content),
        "usage": usage_stats,
        "is_stream": chat_request.stream,
        "exit_code": cli_result["exit_code"]
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
    **Direct Gemini CLI command execution** with API key rotation.
    """
    api_key = get_next_api_key()
    cli_result = run_gemini_subprocess(gemini_command.command, model=gemini_command.model, api_key=api_key)
    
    update_api_key_stats(api_key, gemini_command.model)
    
    return {
        "command_executed": f"gemini -m {gemini_command.model} -p '{gemini_command.command}'",
        "stdout": cli_result["stdout"],
        "stderr": cli_result["stderr"],
        "exit_code": cli_result["exit_code"],
        "api_key_used": api_key[:10] + "..."
    }

@app.get("/stats", tags=["Management"])
async def get_api_stats():
    """
    **Get API key usage statistics**.
    """
    return {
        "api_key_stats": api_key_stats,
        "total_api_keys": len(API_KEYS),
        "total_allowed_tokens": len(ALLOWED_TOKENS)
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

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "**Gemini CLI API is running**",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats"
    }

