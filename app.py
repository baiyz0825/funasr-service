import sys
import os
import time
import logging
from pathlib import Path

# 日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("funasr-service.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("funasr-service")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.models.model_manager import ModelManager
from src.api.openai import router as openai_router, set_model_manager as set_openai_manager
from src.api.management import router as mgmt_router, set_model_manager as set_mgmt_manager
from src.api.websocket import websocket_endpoint, set_model_manager as set_ws_manager
from src.api.log_stream import router as log_router, setup_log_streaming
from src.api.config import router as config_router, load_config, set_model_manager as set_config_manager

logger.info("正在初始化模型管理器...")
MODELS_DIR = Path(__file__).parent / "models"

# 读取配置，用于初始化 ModelManager
startup_config = load_config()
model_manager = ModelManager(
    MODELS_DIR,
    max_loaded=startup_config.get("max_loaded", 2),
    ncpu=startup_config.get("ncpu", 0),
)
logger.info(f"模型缓存目录: {MODELS_DIR}")
logger.info(f"CPU 线程数: {model_manager.ncpu}")
logger.info(f"最大加载模型数: {model_manager.max_loaded}")

set_openai_manager(model_manager)
set_mgmt_manager(model_manager)
set_ws_manager(model_manager)
set_config_manager(model_manager)

# 读取配置并自动加载模型
if startup_config.get("auto_load_models"):
    for model_id in startup_config["auto_load_models"]:
        device = startup_config.get("default_device", model_manager.current_device)
        logger.info(f"自动加载模型: {model_id} → {device}")
        result = model_manager.load_model(model_id, device=device)
        if result["status"] == "success":
            logger.info(f"自动加载成功: {model_id}")
        else:
            logger.warning(f"自动加载失败: {model_id} - {result['message']}")

# 创建 FastAPI 应用
app = FastAPI(
    title="FunASR 语音转写服务",
    description=(
        "基于 FunASR 的语音转写服务，支持实时转写、字幕生成和模型管理。\n\n"
        "**WebSocket 实时转写协议** (`/ws/stream`):\n"
        "1. 发送 JSON 配置消息: `{\"model\": \"sensevoice\", \"language\": \"zh\"}`\n"
        "2. 连续发送音频二进制帧 (支持 16kHz/48kHz PCM/WAV)\n"
        "3. 发送文本消息 `end` 结束\n\n"
        "服务返回: `{\"result\": {\"text\": \"...\", \"is_final\": bool, \"accumulated\": \"...\"}}`"
    ),
    version="1.0.0",
    tags=[
        {"name": "转录", "description": "音频转录相关接口"},
        {"name": "模型管理", "description": "模型加载/卸载/状态查询"},
        {"name": "日志", "description": "实时日志流"},
        {"name": "配置", "description": "服务配置管理"},
    ],
)

# 挂载 API 路由
app.include_router(openai_router)
app.include_router(mgmt_router)
app.include_router(log_router)
app.include_router(config_router)
app.add_api_websocket_route("/ws/stream", websocket_endpoint)

# 启动日志 SSE 推送
setup_log_streaming()

# 静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>FunASR 服务已启动，请将 index.html 放到 static/ 目录</h1>")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FunASR 语音转写服务启动中...")
    logger.info(f"模型缓存目录: {MODELS_DIR}")
    logger.info(f"当前推理设备: {model_manager.current_device}")
    logger.info(f"Web UI: http://localhost:7860")
    logger.info(f"API 文档: http://localhost:7860/docs")
    logger.info(f"WebSocket: ws://localhost:7860/ws/stream")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
