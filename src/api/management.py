from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from ..models.model_manager import ModelManager

router = APIRouter(prefix="/v1")

model_manager: ModelManager = None


def set_model_manager(manager: ModelManager):
    global model_manager
    model_manager = manager


@router.get(
    "/models",
    summary="列出所有模型",
    description="返回所有已配置模型及其加载状态。",
    tags=["模型管理"],
    responses={500: {"description": "模型管理器未初始化"}},
)
async def list_models():
    """列出所有模型及其状态"""
    if not model_manager:
        return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})
    return {"data": model_manager.get_model_status()}


@router.get(
    "/models/loaded",
    summary="列出已加载模型",
    description="返回当前已加载到内存中的模型列表。",
    tags=["模型管理"],
    responses={500: {"description": "模型管理器未初始化"}},
)
async def list_loaded_models():
    """列出已加载的模型"""
    if not model_manager:
        return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})
    return {"data": model_manager.get_loaded_summary()}


class LoadRequest(BaseModel):
    device: Optional[str] = None


@router.post(
    "/models/{model_id}/load",
    summary="加载模型",
    description="加载指定模型到内存，首次使用时会自动下载。资源不足时返回错误提示。",
    tags=["模型管理"],
    responses={
        500: {"description": "模型管理器未初始化"},
    },
)
async def load_model(model_id: str, body: LoadRequest = None):
    """加载模型（首次自动下载）。资源不足时返回错误提示。"""
    if not model_manager:
        return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})
    device = body.device if body and body.device else model_manager.current_device
    return model_manager.load_model(model_id, device=device)


@router.post(
    "/models/{model_id}/unload",
    summary="卸载模型",
    description="从内存中卸载指定模型，释放资源。",
    tags=["模型管理"],
    responses={500: {"description": "模型管理器未初始化"}},
)
async def unload_model(model_id: str):
    """卸载指定模型"""
    if not model_manager:
        return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})
    return model_manager.unload_model(model_id)


@router.get(
    "/devices",
    summary="列出可用设备",
    description="返回可用的推理设备列表（CPU/GPU）及显存信息。",
    tags=["模型管理"],
    responses={500: {"description": "模型管理器未初始化"}},
)
async def list_devices():
    """列出可用设备（含显存信息）"""
    if not model_manager:
        return JSONResponse(status_code=500, content={"error": "模型管理器未初始化"})
    return {"data": model_manager.get_device_info()}


@router.get(
    "/health",
    summary="健康检查",
    description="返回服务健康状态、当前活跃模型和已加载模型数量。",
    tags=["模型管理"],
)
async def health_check():
    """健康检查"""
    summary = model_manager.get_loaded_summary() if model_manager else {}
    return {
        "status": "healthy",
        "active_model": summary.get("active"),
        "loaded_count": summary.get("count", 0),
        "max_loaded": summary.get("max", 2),
    }
