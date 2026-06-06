import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from ..models.model_manager import ModelManager

logger = logging.getLogger("funasr-service.config")

router = APIRouter(prefix="/v1")

model_manager: ModelManager = None

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"

VALID_DEVICES = ["cpu", "mps", "cuda"]

def _detect_default_device() -> str:
    """自动检测最佳推理设备"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_CONFIG = {
    "auto_load_models": {},
    "default_device": _detect_default_device(),
    "ncpu": 0,
    "max_loaded": 0,
}


def set_model_manager(manager: ModelManager):
    global model_manager
    model_manager = manager


class ConfigPayload(BaseModel):
    auto_load_models: Optional[List[str]] = None
    default_device: Optional[str] = None
    ncpu: Optional[int] = None
    max_loaded: Optional[int] = None

    @field_validator("default_device")
    @classmethod
    def validate_device(cls, v):
        if v is not None and v not in VALID_DEVICES:
            raise ValueError(f"default_device 必须是 {VALID_DEVICES} 之一")
        return v

    @field_validator("ncpu")
    @classmethod
    def validate_ncpu(cls, v):
        if v is not None and v < 0:
            raise ValueError("ncpu 必须 >= 0")
        return v

    @field_validator("max_loaded")
    @classmethod
    def validate_max_loaded(cls, v):
        if v is not None and (v < 1 or v > 4):
            raise ValueError("max_loaded 必须在 1-4 之间")
        return v

    @field_validator("auto_load_models")
    @classmethod
    def validate_auto_load_models(cls, v):
        # 基础校验：列表项为非空字符串
        if v is not None:
            for mid in v:
                if not mid or not isinstance(mid, str):
                    raise ValueError("模型ID不能为空")
        return v


def load_config() -> dict:
    """读取配置文件，不存在则返回默认值"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            merged = {**DEFAULT_CONFIG, **saved}
            return merged
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}，使用默认配置")
    return dict(DEFAULT_CONFIG)


def _save_config(config: dict):
    """写入配置文件"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"配置已保存: {CONFIG_PATH}")


@router.get(
    "/config",
    summary="读取配置",
    description="读取当前服务配置。配置文件不存在时返回默认值。",
    tags=["配置"],
)
async def get_config():
    config = load_config()
    return {"data": config}


@router.post(
    "/config",
    summary="更新配置",
    description="保存服务配置到 config.json。仅更新请求中包含的字段。",
    tags=["配置"],
    responses={400: {"description": "参数校验失败"}},
)
async def update_config(payload: ConfigPayload):
    # 校验模型 ID（需要 model_manager）
    if payload.auto_load_models is not None:
        if not model_manager:
            return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})
        valid_ids = set(model_manager.model_configs.keys())
        for mid in payload.auto_load_models:
            if mid not in valid_ids:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"无效的模型ID: {mid}，可用: {', '.join(sorted(valid_ids))}"},
                )

    current = load_config()
    updates = payload.model_dump(exclude_none=True)
    current.update(updates)

    try:
        _save_config(current)
    except Exception as e:
        logger.exception("保存配置文件失败")
        return JSONResponse(status_code=500, content={"error": f"保存失败: {e}"})

    return {"data": current, "message": "配置已保存"}
