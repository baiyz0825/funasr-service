import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional
import io

class AudioProcessor:
    """音频处理器"""

    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        audio, sr = sf.read(file_path)

        # 转换为单声道
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 重采样到目标采样率（如果需要）
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return audio, sr

    @staticmethod
    def save_audio(audio: np.ndarray, sr: int, output_path: str):
        """保存音频文件"""
        sf.write(output_path, audio, sr)

    @staticmethod
    def bytes_to_audio(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """将字节流转换为音频数组"""
        audio, sr = sf.read(io.BytesIO(audio_bytes))

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return audio, sr
