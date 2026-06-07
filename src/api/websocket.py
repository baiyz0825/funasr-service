from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import JSONResponse
import json
import numpy as np
import base64
import tempfile
import os
import logging
import time

from ..models.model_manager import ModelManager

logger = logging.getLogger("funasr-service.websocket")

model_manager: ModelManager = None

STREAMING_CHUNK_SIZE = [0, 10, 5]
SAMPLE_RATE = 16000
CHUNK_STRIDE = STREAMING_CHUNK_SIZE[1] * 960  # 9600 samples = 600ms
MAX_BUFFER_DURATION_S = 3.0  # 缓冲区上限：超过 3 秒丢弃旧音频，防止延迟堆积
OFFLINE_BUFFER_S = 1.0  # 非流式模型：累积 1 秒后触发识别（伪流式）

router = APIRouter()


def set_model_manager(manager: ModelManager):
    global model_manager
    model_manager = manager


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, data: dict, websocket: WebSocket):
        await websocket.send_json(data)


mgr = ConnectionManager()


def _decode_pcm(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int16).astype(np.float32) / 32768.0


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 语音识别

    协议: config → audio* → end
    客户端无需指定模型，使用服务端当前加载的模型。
    流式/非流式行为由已加载模型自动决定。
    """
    await mgr.connect(websocket)

    audio_buffer = np.array([], dtype=np.float32)
    is_streaming = False
    cache = {}
    accumulated_text = ""
    configured = False

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "config":
                # 检查是否有已加载的模型
                if not model_manager.active_model:
                    await mgr.send_json({"type": "error", "message": "没有已加载的模型，请先在管理页面加载模型"}, websocket)
                    continue

                active_id = model_manager.active_model
                cfg = model_manager.model_configs.get(active_id, {})
                is_streaming = cfg.get("streaming", False)
                cache = {}
                accumulated_text = ""
                configured = True

                await mgr.send_json({
                    "type": "config_ack",
                    "model": active_id,
                    "model_name": cfg.get("name", active_id),
                    "streaming": is_streaming,
                }, websocket)

            elif msg_type == "audio":
                if not configured:
                    await mgr.send_json({"type": "error", "message": "请先发送 config 消息"}, websocket)
                    continue

                audio_b64 = msg.get("data", "")
                if not audio_b64:
                    continue
                try:
                    chunk = _decode_pcm(audio_b64)
                    audio_buffer = np.concatenate([audio_buffer, chunk])
                except Exception as e:
                    logger.error(f"音频解码失败: {e}")
                    continue

                asr = model_manager.asr_model
                if not asr:
                    continue

                if is_streaming:
                    # 真流式：paraformer-streaming
                    max_samples = int(MAX_BUFFER_DURATION_S * SAMPLE_RATE)
                    if len(audio_buffer) > max_samples:
                        dropped = len(audio_buffer) - max_samples
                        audio_buffer = audio_buffer[-max_samples:]
                        logger.warning(f"缓冲区溢出，丢弃 {dropped/SAMPLE_RATE:.1f}s 旧音频")

                    while len(audio_buffer) >= CHUNK_STRIDE:
                        chunk_input = audio_buffer[:CHUNK_STRIDE]
                        audio_buffer = audio_buffer[CHUNK_STRIDE:]

                        try:
                            t0 = time.monotonic()
                            res = asr.generate(
                                input=chunk_input,
                                cache=cache,
                                is_final=False,
                                chunk_size=STREAMING_CHUNK_SIZE,
                                encoder_chunk_look_back=4,
                                decoder_chunk_look_back=1,
                            )
                            elapsed = time.monotonic() - t0
                            if elapsed > 0.5:
                                logger.warning(f"流式推理耗时 {elapsed:.2f}s (音频 0.6s)，CPU 可能跟不上实时")
                            if res and res[0].get("text"):
                                partial = res[0]["text"]
                                accumulated_text += partial
                                await mgr.send_json({
                                    "type": "result",
                                    "text": partial,
                                    "is_final": False,
                                    "accumulated": accumulated_text,
                                }, websocket)
                        except Exception:
                            logger.exception("流式识别异常")

                else:
                    # 伪流式：累积 OFFLINE_BUFFER_S 秒后触发识别
                    buf_dur = len(audio_buffer) / SAMPLE_RATE
                    if buf_dur >= OFFLINE_BUFFER_S:
                        tmp_path = None
                        try:
                            import soundfile as sf
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                tmp_path = f.name
                            sf.write(tmp_path, audio_buffer, SAMPLE_RATE)
                            t0 = time.monotonic()
                            res = asr.generate(input=tmp_path)
                            elapsed = time.monotonic() - t0
                            logger.info(f"伪流式推理耗时 {elapsed:.2f}s (音频 {buf_dur:.1f}s)")
                            text = res[0].get("text", "") if res else ""
                            if text:
                                accumulated_text += text
                                await mgr.send_json({
                                    "type": "result",
                                    "text": text,
                                    "is_final": False,
                                    "accumulated": accumulated_text,
                                }, websocket)
                        except Exception:
                            logger.exception("识别异常")
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                            audio_buffer = np.array([], dtype=np.float32)

            elif msg_type == "end":
                asr = model_manager.asr_model

                if is_streaming and asr:
                    # 发送剩余缓冲区
                    if len(audio_buffer) > 0:
                        try:
                            t0 = time.monotonic()
                            res = asr.generate(
                                input=audio_buffer, cache=cache, is_final=False,
                                chunk_size=STREAMING_CHUNK_SIZE,
                                encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                            )
                            logger.info(f"最终块推理耗时 {time.monotonic()-t0:.2f}s")
                            if res and res[0].get("text"):
                                accumulated_text += res[0]["text"]
                        except Exception:
                            logger.exception("最终块识别异常")

                    # flush 缓存
                    try:
                        t0 = time.monotonic()
                        res = asr.generate(
                            input=np.array([], dtype=np.float32),
                            cache=cache, is_final=True,
                            chunk_size=STREAMING_CHUNK_SIZE,
                            encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                        )
                        logger.info(f"flush 推理耗时 {time.monotonic()-t0:.2f}s")
                        if res and res[0].get("text"):
                            accumulated_text += res[0]["text"]
                    except Exception:
                        logger.exception("flush异常")

                    cache = {}

                elif not is_streaming and asr:
                    if len(audio_buffer) >= SAMPLE_RATE * 0.1:
                        tmp_path = None
                        try:
                            import soundfile as sf
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                tmp_path = f.name
                            sf.write(tmp_path, audio_buffer, SAMPLE_RATE)
                            res = asr.generate(input=tmp_path)
                            text = res[0].get("text", "") if res else ""
                            accumulated_text += text
                        except Exception:
                            logger.exception("最终识别异常")
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)

                await mgr.send_json({
                    "type": "result",
                    "text": "",
                    "is_final": True,
                    "accumulated": accumulated_text,
                }, websocket)

                audio_buffer = np.array([], dtype=np.float32)
                accumulated_text = ""

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开")
    except Exception:
        logger.exception("WebSocket 异常")
    finally:
        mgr.disconnect(websocket)


WS_PROTOCOL_DOC = {
    "endpoint": "/ws/stream",
    "protocol": "WebSocket",
    "description": (
        "实时语音识别 WebSocket 接口。\n\n"
        "连接地址: `ws://host:7860/ws/stream`（或 `wss://` 如已启用 HTTPS）\n\n"
        "客户端无需指定模型，使用服务端当前加载的模型。"
        "流式/非流式行为由已加载模型自动决定。\n\n"
        "## 通信流程\n\n"
        "```\n"
        "客户端 → 服务端: config (JSON)\n"
        "服务端 → 客户端: config_ack (JSON)\n"
        "客户端 → 服务端: audio (JSON, 重复发送)\n"
        "服务端 → 客户端: result (JSON, 每个音频块返回)\n"
        "客户端 → 服务端: end (JSON)\n"
        "服务端 → 客户端: result [is_final=true] (JSON)\n"
        "```\n\n"
        "## 客户端消息\n\n"
        "### 1. config（必须第一条）\n"
        "```json\n"
        '{"type": "config"}\n'
        "```\n"
        "无需指定模型，使用服务端已加载的模型。\n\n"
        "### 2. audio（音频数据，重复发送）\n"
        "```json\n"
        '{"type": "audio", "data": "<base64 编码的 PCM int16 音频>"}\n'
        "```\n"
        "- **采样率**: 必须为 **16kHz 单声道**，服务端不负责转码\n"
        "- **编码**: PCM 16-bit signed little-endian → base64\n"
        "- **重要**: `AudioContext({ sampleRate: 16000 })` 仅为 hint，多数浏览器实际使用 48kHz。"
        "客户端**必须**检测实际采样率并在发送前重采样到 16kHz，否则音频会被拉伸/压缩，导致识别完全错误\n"
        "- 流式模型: 每 600ms (9600 采样点) 处理一次\n"
        "- 非流式模型: 缓冲 1 秒后处理（伪流式）\n\n"
        "### 3. end（结束识别）\n"
        "```json\n"
        '{"type": "end"}\n'
        "```\n\n"
        "## 服务端消息\n\n"
        "### config_ack（配置确认）\n"
        "```json\n"
        '{"type": "config_ack", "model": "paraformer-streaming", "model_name": "Paraformer-zh-Streaming", "streaming": true}\n'
        "```\n\n"
        "### result（识别结果）\n"
        "```json\n"
        '{"type": "result", "text": "当前文本", "is_final": false, "accumulated": "累积的全部文本"}\n'
        "```\n"
        "- `text`: 当前块识别文本\n"
        "- `accumulated`: 从开始到当前的完整累积文本\n"
        "- `is_final`: 是否最终结果（发送 `end` 后返回）\n\n"
        "### error（错误）\n"
        "```json\n"
        '{"type": "error", "message": "错误描述"}\n'
        "```\n\n"
        "## 可用模型\n\n"
        "通过管理页面加载模型，客户端无需关心模型选择。\n\n"
        "| 模型 | 流式 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| paraformer-streaming | 是 (600ms) | 中文实时识别，低延迟 |\n"
        "| funasr-nano | 否 (伪流式 1s) | 31种语言，高精度，自带标点 |\n"
    ),
    "tags": ["WebSocket"],
    "messages": {
        "client": [
            {"type": "config", "fields": {}},
            {"type": "audio", "fields": {"data": "string (base64 编码的 16kHz PCM int16 音频)"}},
            {"type": "end", "fields": {}},
        ],
        "server": [
            {"type": "config_ack", "fields": {"model": "string", "model_name": "string", "streaming": "boolean"}},
            {"type": "result", "fields": {"text": "string", "is_final": "boolean", "accumulated": "string"}},
            {"type": "error", "fields": {"message": "string"}},
        ],
    },
    "models": [
        {"id": "paraformer-streaming", "name": "Paraformer-zh-Streaming", "streaming": True, "size": "2.0GB"},
        {"id": "funasr-nano", "name": "Fun-ASR-MLT-Nano", "streaming": False, "size": "2.0GB"},
    ],
}


@router.get(
    "/ws/info",
    summary="WebSocket 识别协议文档",
    description=(
        "返回 `/ws/stream` WebSocket 接口的完整协议文档，包括消息格式、"
        "通信流程、可用模型和代码示例。"
    ),
    tags=["WebSocket"],
)
async def ws_protocol_info():
    return JSONResponse(content=WS_PROTOCOL_DOC)
