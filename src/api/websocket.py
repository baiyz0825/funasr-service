from fastapi import WebSocket, WebSocketDisconnect
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
                                await mgr.send_json({
                                    "type": "result",
                                    "text": partial,
                                    "is_final": False,
                                    "accumulated": accumulated_text + partial,
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
                                accumulated_text += res[0]["text"]
                        except Exception:
                            logger.exception("最终块识别异常")

                    # flush 缓存
                    try:
                        res = asr.generate(
                            input=np.array([], dtype=np.float32),
                            cache=cache, is_final=True,
                            chunk_size=STREAMING_CHUNK_SIZE,
                            encoder_chunk_look_back=4, decoder_chunk_look_back=1,
                        )
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
