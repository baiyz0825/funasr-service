from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from fastapi.responses import JSONResponse
import json
import numpy as np
import base64
import tempfile
import os
import logging

from ..models.model_manager import ModelManager

logger = logging.getLogger("funasr-service.websocket")

model_manager: ModelManager = None

STREAMING_CHUNK_SIZE = [0, 10, 5]
SAMPLE_RATE = 16000
CHUNK_STRIDE = STREAMING_CHUNK_SIZE[1] * 960  # 9600

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
    """WebSocket 流式识别

    协议: config → audio* → end
    返回: result {text, is_final, accumulated}
    """
    await mgr.connect(websocket)

    audio_buffer = np.array([], dtype=np.float32)
    current_model_id = None
    is_streaming = False
    cache = {}
    accumulated_text = ""  # 累积的确认文本
    last_partial = ""  # 流式模式：上一次 partial 文本（用于替换更新）

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "config":
                req_model = msg.get("model", "sensevoice")

                # 确保模型已加载
                if model_manager.active_model != req_model:
                    result = model_manager.load_model(req_model)
                    if result["status"] != "success":
                        await mgr.send_json({"type": "error", "message": result["message"]}, websocket)
                        continue

                current_model_id = req_model
                cfg = model_manager.model_configs.get(req_model, {})
                is_streaming = cfg.get("streaming", False)
                cache = {}
                accumulated_text = ""
                last_partial = ""

                await mgr.send_json({
                    "type": "config_ack",
                    "model": req_model,
                    "streaming": is_streaming,
                }, websocket)

            elif msg_type == "audio":
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
                    while len(audio_buffer) >= CHUNK_STRIDE:
                        chunk_input = audio_buffer[:CHUNK_STRIDE]
                        audio_buffer = audio_buffer[CHUNK_STRIDE:]

                        try:
                            res = asr.generate(
                                input=chunk_input,
                                cache=cache,
                                is_final=False,
                                chunk_size=STREAMING_CHUNK_SIZE,
                                encoder_chunk_look_back=4,
                                decoder_chunk_look_back=1,
                            )
                            if res and res[0].get("text"):
                                partial = res[0]["text"]
                                # 流式 partial 是累积的（每次包含之前所有文本）
                                accumulated_text = partial
                                await mgr.send_json({
                                    "type": "result",
                                    "text": partial,
                                    "is_final": False,
                                    "accumulated": accumulated_text,
                                }, websocket)
                        except Exception:
                            logger.exception("流式识别异常")

                else:
                    buf_dur = len(audio_buffer) / SAMPLE_RATE
                    if buf_dur >= 2.0:
                        tmp_path = None
                        try:
                            import soundfile as sf
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                tmp_path = f.name
                            sf.write(tmp_path, audio_buffer, SAMPLE_RATE)
                            res = asr.generate(input=tmp_path)
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
                            res = asr.generate(
                                input=audio_buffer, cache=cache, is_final=False,
                                chunk_size=STREAMING_CHUNK_SIZE,
                                encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                            )
                            if res and res[0].get("text"):
                                accumulated_text = res[0]["text"]
                        except Exception:
                            logger.exception("最终块识别异常")

                    # flush 缓存，返回完整最终文本
                    try:
                        res = asr.generate(
                            input=np.array([], dtype=np.float32),
                            cache=cache, is_final=True,
                            chunk_size=STREAMING_CHUNK_SIZE,
                            encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                        )
                        if res and res[0].get("text"):
                            accumulated_text = res[0]["text"]
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
                last_partial = ""

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
        "实时流式语音识别 WebSocket 接口。\n\n"
        "连接地址: `ws://host:7860/ws/stream`（或 `wss://` 如已启用 HTTPS）\n\n"
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
        '{"type": "config", "model": "paraformer-streaming"}\n'
        "```\n"
        "- `model`: 模型 ID，可选 `sensevoice` / `paraformer` / `paraformer-streaming` / `funasr-nano`\n\n"
        "### 2. audio（音频数据，重复发送）\n"
        "```json\n"
        '{"type": "audio", "data": "<base64 编码的 PCM int16 音频>"}\n'
        "```\n"
        "- 音频格式: 16kHz 单声道 PCM int16，base64 编码\n"
        "- 流式模型: 每 600ms (9600 采样点) 处理一次\n"
        "- 非流式模型: 缓冲 2 秒后处理\n\n"
        "### 3. end（结束识别）\n"
        "```json\n"
        '{"type": "end"}\n'
        "```\n\n"
        "## 服务端消息\n\n"
        "### config_ack（配置确认）\n"
        "```json\n"
        '{"type": "config_ack", "model": "paraformer-streaming", "streaming": true}\n'
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
        "## JavaScript 示例\n\n"
        "```javascript\n"
        "const ws = new WebSocket('wss://host:7860/ws/stream');\n"
        "ws.onopen = () => {\n"
        "    ws.send(JSON.stringify({type: 'config', model: 'paraformer-streaming'}));\n"
        "};\n"
        "ws.onmessage = (e) => {\n"
        "    const d = JSON.parse(e.data);\n"
        '    if (d.type === "result") console.log("识别:", d.accumulated);\n'
        "};\n"
        "// 发送音频 (base64 编码的 16kHz PCM int16)\n"
        "ws.send(JSON.stringify({type: 'audio', data: base64Audio}));\n"
        "// 结束\n"
        "ws.send(JSON.stringify({type: 'end'}));\n"
        "```"
    ),
    "tags": ["WebSocket"],
    "messages": {
        "client": [
            {"type": "config", "fields": {"model": "string (模型ID，必填)"}},
            {"type": "audio", "fields": {"data": "string (base64 编码的 16kHz PCM int16 音频)"}},
            {"type": "end", "fields": {}},
        ],
        "server": [
            {"type": "config_ack", "fields": {"model": "string", "streaming": "boolean"}},
            {"type": "result", "fields": {"text": "string", "is_final": "boolean", "accumulated": "string"}},
            {"type": "error", "fields": {"message": "string"}},
        ],
    },
    "models": [
        {"id": "sensevoice", "name": "SenseVoice-Small", "streaming": False, "size": "1.5GB"},
        {"id": "paraformer", "name": "Paraformer-zh", "streaming": False, "size": "2.0GB"},
        {"id": "paraformer-streaming", "name": "Paraformer-zh-Streaming", "streaming": True, "size": "2.0GB"},
        {"id": "funasr-nano", "name": "Fun-ASR-Nano", "streaming": False, "size": "3.0GB"},
    ],
}


@router.get(
    "/ws/info",
    summary="WebSocket 流式识别协议文档",
    description=(
        "返回 `/ws/stream` WebSocket 接口的完整协议文档，包括消息格式、"
        "通信流程、可用模型和代码示例。"
    ),
    tags=["WebSocket"],
)
async def ws_protocol_info():
    return JSONResponse(content=WS_PROTOCOL_DOC)
