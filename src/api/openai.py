from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import re
import tempfile
import logging

from ..models.model_manager import ModelManager
from ..utils.audio import AudioProcessor
from ..utils.subtitle import SubtitleGenerator

logger = logging.getLogger("funasr-service.openai")

router = APIRouter(prefix="/v1")

model_manager: ModelManager = None


def set_model_manager(manager: ModelManager):
    global model_manager
    model_manager = manager


_EMOTION_TAGS = {
    "<|HAPPY|>": "开心", "<|SAD|>": "悲伤", "<|ANGRY|>": "愤怒",
    "<|NEUTRAL|>": "中性", "<|FEARFUL|>": "恐惧", "<|DISGUSTED|>": "厌恶",
    "<|SURPRISED|>": "惊喜",
}
_EMOTION_PATTERN = re.compile(r"<\|[A-Z]+\|>")


def _extract_emotions(text: str):
    found = _EMOTION_PATTERN.findall(text)
    clean_text = _EMOTION_PATTERN.sub("", text).strip()
    return clean_text, [_EMOTION_TAGS[t] for t in found if t in _EMOTION_TAGS]


def _rich_postprocess(text: str) -> str:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        return rich_transcription_postprocess(text)
    except ImportError:
        return re.sub(r"<\|[^|]+\|>", "", text).strip()


def _get_active_model():
    """获取当前活跃模型实例和 generate 参数"""
    if not model_manager or not model_manager.active_model:
        return None, {}, None
    entry = model_manager._loaded.get(model_manager.active_model)
    if not entry:
        return None, {}, None
    cfg = model_manager.model_configs.get(model_manager.active_model, {})
    return entry["instance"], entry.get("generate_params", {}), cfg


@router.post(
    "/audio/transcriptions",
    summary="音频转录",
    description="上传音频文件进行语音转录。使用服务端当前加载的模型，无需指定模型。",
    tags=["转录"],
    responses={
        200: {"description": "转录成功，返回文本及可选的分段/情感信息"},
        400: {"description": "无可用模型"},
        500: {"description": "模型管理器未初始化或转录过程出错"},
    },
)
async def transcribe(
    file: UploadFile = File(..., description="音频文件（支持 WAV、MP3、FLAC 等常见格式）"),
    language: str = Form("zh", description="语言代码，如 zh（中文）、en（英文）"),
    response_format: str = Form("json", description="输出格式：json、verbose_json、srt、vtt"),
    enable_speaker_diarization: bool = Form(False, description="是否启用说话人分离"),
    enable_emotion_detection: bool = Form(False, description="是否启用情感检测（需模型支持）"),
):
    if not model_manager:
        return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})

    asr, gen_params, cfg = _get_active_model()
    if not asr:
        return JSONResponse(status_code=400, content={"error": "没有已加载的模型，请先在管理页面加载模型"})

    model_id = model_manager.active_model
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        audio, sr = AudioProcessor.load_audio(tmp_path)
        gen_kwargs = {"input": tmp_path, **gen_params}
        if language:
            gen_kwargs["language"] = language

        if enable_speaker_diarization:
            try:
                from funasr import AutoModel
                spk_kwargs = {
                    "model": cfg.get("alias", model_id),
                    "device": model_manager.current_device,
                    "spk_model": "cam++",
                    **cfg.get("model_params", {}),
                }
                if model_manager.current_device == "cpu":
                    spk_kwargs["ncpu"] = model_manager.ncpu
                spk = AutoModel(**spk_kwargs)
                result = spk.generate(input=tmp_path, batch_size_s=300)
                del spk
            except Exception:
                logger.exception("说话人分离失败，回退到普通识别")
                result = asr.generate(**gen_kwargs)
        else:
            result = asr.generate(**gen_kwargs)

        if not result:
            return {"text": ""}

        raw_text = result[0].get("text", "")
        # funasr-nano 自带标点和 ITN，不需要后处理；其他模型需要
        text = _rich_postprocess(raw_text) if model_id != "funasr-nano" else raw_text

        emotions = []
        if enable_emotion_detection:
            _, emotions = _extract_emotions(raw_text)

        segments = []
        for i, sent in enumerate(result[0].get("sentence_info", [])):
            seg_text = sent.get("text", "")
            seg_emo = []
            if model_id != "funasr-nano":
                seg_text = _rich_postprocess(seg_text)
            if enable_emotion_detection:
                _, seg_emo = _extract_emotions(sent.get("text", ""))
            seg = {
                "id": i,
                "start": sent.get("start", 0) / 1000,
                "end": sent.get("end", 0) / 1000,
                "text": seg_text,
                "speaker": sent.get("spk", sent.get("speaker")),
            }
            if seg_emo:
                seg["emotions"] = seg_emo
            segments.append(seg)

        if response_format == "srt":
            return {"text": SubtitleGenerator.generate_srt(segments)}
        elif response_format == "vtt":
            return {"text": SubtitleGenerator.generate_vtt(segments)}
        elif response_format == "verbose_json":
            resp = {"text": text, "duration": len(audio) / sr, "language": language, "segments": segments}
            if emotions:
                resp["emotions"] = emotions
            return resp
        else:
            return {"text": text}

    except Exception as e:
        logger.exception("识别失败")
        return JSONResponse(status_code=500, content={"error": f"识别失败: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
