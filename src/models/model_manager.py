import torch
from pathlib import Path
from typing import Dict, Optional, List
import logging
import os
import multiprocessing

logger = logging.getLogger("funasr-service.model_manager")

# 兼容性补丁：PyTorch < 2.6 缺少 float8_e8m0fnu dtype，transformers 4.48+ 需要它
# 用 torch.uint8 作为占位符，避免升级整个 PyTorch（仅影响 FP8 量化，本服务不使用）
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8

# FunASR AutoModel 不支持 cache_dir 参数，需要通过环境变量控制下载目录
# MODELSCOPE_CACHE 控制 ModelScope 模型下载路径，HF_HOME 控制 HuggingFace 模型
_DEFAULT_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
_MODELS_DIR = _DEFAULT_MODELS_DIR
os.environ.setdefault("MODELSCOPE_CACHE", str(_MODELS_DIR))
os.environ.setdefault("HF_HOME", str(_MODELS_DIR / "hf"))
# HuggingFace 镜像加速（国内网络环境）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class ModelManager:
    """多模型管理器

    支持同时加载多个模型，自动管理 GPU/CPU 内存。
    """

    def __init__(self, models_dir: Path, max_loaded: int = 0, ncpu: int = 0):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)

        # CPU 推理线程数：0=自动检测物理核心数，上限16
        if ncpu <= 0:
            physical_cores = multiprocessing.cpu_count()
            # macOS Apple Silicon 统一内存，可用全部核心；x86 取物理核心（逻辑核心的一半）
            self.ncpu = min(physical_cores, 16)
        else:
            self.ncpu = ncpu

        # FunASR AutoModel 不支持 cache_dir，通过环境变量控制下载目录
        # 必须在 import funasr 之前设置
        os.environ["MODELSCOPE_CACHE"] = str(self.models_dir)
        os.environ["HF_HOME"] = str(self.models_dir / "hf")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # max_loaded: 0=自动计算（根据 GPU 显存或 CPU 内存），1-4=手动指定
        if max_loaded <= 0:
            self.max_loaded = self._auto_detect_max_loaded()
            logger.info(f"自动检测最大加载模型数: {self.max_loaded}")
        else:
            self.max_loaded = max_loaded

        # FunASR 官方模型配置
        self.model_configs = {
            "sensevoice": {
                "name": "SenseVoice-Small",
                "alias": "iic/SenseVoiceSmall",
                "size": "1.5GB",
                "size_bytes": int(1.5 * 1024**3),
                "description": "情感+音频事件检测，5种语言，极快，自带标点和ITN",
                "streaming": False,
                "model_params": {
                    "vad_model": "fsmn-vad",
                    "vad_kwargs": {"max_single_segment_time": 30000},
                },
                "generate_params": {
                    "use_itn": True,
                    "batch_size_s": 60,
                    "merge_vad": True,
                    "merge_length_s": 15,
                },
            },
            "funasr-nano": {
                "name": "Fun-ASR-MLT-Nano",
                "alias": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
                "size": "2.0GB",
                "size_bytes": int(2.0 * 1024**3),
                "description": "800M参数，31种语言+7种方言，自带ITN和标点，支持翻译和代码切换，中文CER 1.22%",
                "streaming": False,
                "model_params": {
                    "trust_remote_code": True,
                    "remote_code": "./model.py",
                    "hub": "hf",
                    "vad_model": "fsmn-vad",
                    "vad_kwargs": {"max_single_segment_time": 30000},
                },
                "generate_params": {"batch_size": 1},
            },
        }

        # 已加载的模型 {model_id: {"instance": AutoModel, "device": str, "generate_params": dict}}
        self._loaded: Dict[str, dict] = {}
        # 当前活跃模型（用于 API 调用）
        self.active_model: Optional[str] = None

    def _auto_detect_max_loaded(self) -> int:
        """根据可用资源自动计算最大同时加载模型数

        GPU: 按可用显存 / 平均模型大小(2GB) * 1.5倍加载系数
        CPU: 按可用内存 / 平均模型大小(2GB) * 1.5倍加载系数
        返回 1-4
        """
        import psutil

        # GPU 显存
        if torch.cuda.is_available():
            try:
                free_vram = torch.cuda.mem_get_info(0)[0]  # bytes
                free_gb = free_vram / (1024 ** 3)
                # 平均模型约 2GB，加载后实际占用约 1.5x = 3GB
                count = max(1, int(free_gb / 3.0))
                return min(count, 4)
            except Exception:
                pass

        # CPU 内存
        try:
            mem = psutil.virtual_memory()
            free_gb = mem.available / (1024 ** 3)
            # CPU 模型占用约 2-3GB，留 4GB 给系统
            usable = max(free_gb - 4.0, 2.0)
            count = max(1, int(usable / 3.0))
            return min(count, 4)
        except Exception:
            return 2  # fallback

    @property
    def asr_model(self):
        """兼容旧代码：返回当前活跃模型"""
        if self.active_model and self.active_model in self._loaded:
            return self._loaded[self.active_model]["instance"]
        return None

    @property
    def current_model(self):
        return self.active_model

    @property
    def current_device(self):
        if self.active_model and self.active_model in self._loaded:
            return self._loaded[self.active_model]["device"]
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_device_memory(self, device: str) -> Optional[int]:
        """获取设备可用显存（字节），CPU/MPS 返回 None"""
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = int(device.split(":")[1]) if ":" in device else 0
            return torch.cuda.mem_get_info(idx)[0]  # 可用显存
        return None

    def _estimate_memory_need(self, model_id: str) -> int:
        """估算模型需要的显存"""
        cfg = self.model_configs.get(model_id, {})
        base = cfg.get("size_bytes", int(2 * 1024**3))
        # 加载时实际占用约 1.5x 模型文件大小
        return int(base * 1.5)

    def _check_resources(self, model_id: str, device: str) -> Optional[str]:
        """检查资源是否足够，返回错误信息或 None"""
        # 检查是否已加载
        if model_id in self._loaded:
            return None  # 已加载，切换活跃即可

        # 检查加载数量限制
        if len(self._loaded) >= self.max_loaded:
            loaded_names = [self.model_configs.get(m, {}).get("name", m) for m in self._loaded]
            return (f"最多同时加载 {self.max_loaded} 个模型，当前已加载: {', '.join(loaded_names)}。"
                    f"请先卸载一个再加载新模型。")

        # 检查设备显存（仅 CUDA）
        if device.startswith("cuda"):
            available = self._get_device_memory(device)
            if available is not None:
                need = self._estimate_memory_need(model_id)
                if available < need:
                    avail_gb = available / 1024**3
                    need_gb = need / 1024**3
                    return (f"{device} 可用显存 {avail_gb:.1f}GB，"
                            f"加载 {self.model_configs[model_id]['name']} 约需 {need_gb:.1f}GB。"
                            f"请先卸载其他模型释放显存。")

        return None

    def load_model(self, model_id: str, device: str = "cuda") -> dict:
        """加载模型到指定设备

        如果模型已加载，切换为活跃模型。
        如果资源不足，返回错误提示。
        """
        if model_id not in self.model_configs:
            return {"status": "error", "message": f"未知模型: {model_id}"}

        # 已加载 → 切换活跃
        if model_id in self._loaded:
            self.active_model = model_id
            cfg = self.model_configs[model_id]
            return {"status": "success", "message": f"切换到 {cfg['name']}"}

        # 资源检查
        err = self._check_resources(model_id, device)
        if err:
            return {"status": "error", "message": err}

        cfg = self.model_configs[model_id]

        try:
            from funasr import AutoModel

            logger.info(f"加载模型: {cfg['alias']} → {device} (ncpu={self.ncpu if device == 'cpu' else 'N/A'})")
            load_kwargs = {
                "model": cfg["alias"],
                "device": device,
                **cfg.get("model_params", {}),
            }
            # CPU 推理时传递 ncpu 参数，覆盖 FunASR 默认的 4 线程
            if device == "cpu":
                load_kwargs["ncpu"] = self.ncpu
            instance = AutoModel(**load_kwargs)

            self._loaded[model_id] = {
                "instance": instance,
                "device": device,
                "generate_params": cfg.get("generate_params", {}),
            }
            self.active_model = model_id

            logger.info(f"模型 {cfg['name']} 加载完成（已加载 {len(self._loaded)}/{self.max_loaded}）")
            return {
                "status": "success",
                "message": f"模型 {cfg['name']} 已加载（{len(self._loaded)}/{self.max_loaded}）",
            }
        except Exception as e:
            logger.exception(f"加载模型 {model_id} 失败")
            return {"status": "error", "message": f"加载失败: {str(e)}"}

    def unload_model(self, model_id: Optional[str] = None) -> dict:
        """卸载指定模型（默认卸载活跃模型）"""
        target = model_id or self.active_model

        if not target or target not in self._loaded:
            return {"status": "info", "message": "该模型未加载"}

        name = self.model_configs.get(target, {}).get("name", target)
        entry = self._loaded.pop(target)

        del entry["instance"]

        # 如果卸载的是活跃模型，切换到其他已加载的模型
        if self.active_model == target:
            self.active_model = next(iter(self._loaded), None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info(f"模型 {name} 已卸载（剩余 {len(self._loaded)}/{self.max_loaded}）")
        return {"status": "success", "message": f"模型 {name} 已卸载"}

    def get_model_status(self) -> List[dict]:
        """获取所有模型状态"""
        return [
            {
                "id": mid,
                "name": cfg["name"],
                "alias": cfg["alias"],
                "size": cfg["size"],
                "description": cfg["description"],
                "streaming": cfg.get("streaming", False),
                "loaded": mid in self._loaded,
                "active": mid == self.active_model,
                "device": self._loaded[mid]["device"] if mid in self._loaded else None,
            }
            for mid, cfg in self.model_configs.items()
        ]

    def get_loaded_summary(self) -> dict:
        """获取加载摘要"""
        return {
            "active": self.active_model,
            "loaded": [
                {
                    "id": mid,
                    "name": self.model_configs.get(mid, {}).get("name", mid),
                    "device": entry["device"],
                }
                for mid, entry in self._loaded.items()
            ],
            "max": self.max_loaded,
            "count": len(self._loaded),
        }

    def get_device_info(self) -> List[dict]:
        """获取可用设备"""
        devices = [{"id": "cpu", "name": f"CPU ({self.ncpu} threads)", "available": True, "ncpu": self.ncpu}]

        if torch.backends.mps.is_available():
            devices.append({"id": "mps", "name": "Apple Silicon (MPS)", "available": True})

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem = torch.cuda.mem_get_info(i)[0]
                devices.append({
                    "id": f"cuda:{i}",
                    "name": f"GPU {i}: {props.name}",
                    "available": True,
                    "total": f"{props.total_memory / 1024**3:.1f}GB",
                    "free": f"{free_mem / 1024**3:.1f}GB",
                })

        return devices
