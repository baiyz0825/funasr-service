from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 模型配置（参考，实际使用 src/models/model_manager.py）
# FunASR AutoModel 使用官方模型别名，会自动从 ModelScope/HuggingFace 下载
MODEL_CONFIGS = {
    "sensevoice": {
        "name": "SenseVoice-Small",
        "alias": "iic/SenseVoiceSmall",
        "size": "1.5GB",
        "description": "情感+音频事件检测，5种语言，极快",
        "streaming": False,
    },
    "paraformer": {
        "name": "Paraformer-zh",
        "alias": "paraformer-zh",
        "size": "2.0GB",
        "description": "中文生产级识别，适合长音频",
        "streaming": False,
    },
    "paraformer-streaming": {
        "name": "Paraformer-zh-Streaming",
        "alias": "paraformer-zh-streaming",
        "size": "2.0GB",
        "description": "中文流式识别，600ms低延迟",
        "streaming": True,
    },
    "funasr-nano": {
        "name": "Fun-ASR-Nano",
        "alias": "FunAudioLLM/Fun-ASR-Nano-2512",
        "size": "3.0GB",
        "description": "31种语言高精度，自带标点",
        "streaming": False,
    },
}

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860
