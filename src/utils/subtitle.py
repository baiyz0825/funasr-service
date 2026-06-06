from typing import List, Dict
from datetime import timedelta

class SubtitleGenerator:
    """字幕生成器"""

    @staticmethod
    def seconds_to_srt_time(seconds: float) -> str:
        """将秒数转换为 SRT 时间格式"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def seconds_to_vtt_time(seconds: float) -> str:
        """将秒数转换为 WebVTT 时间格式"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    @classmethod
    def generate_srt(cls, segments: List[Dict]) -> str:
        """生成 SRT 字幕"""
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            start_time = cls.seconds_to_srt_time(segment["start"])
            end_time = cls.seconds_to_srt_time(segment["end"])
            text = segment["text"]

            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(f"{text}")
            srt_lines.append("")

        return "\n".join(srt_lines)

    @classmethod
    def generate_vtt(cls, segments: List[Dict]) -> str:
        """生成 WebVTT 字幕"""
        vtt_lines = ["WEBVTT", ""]

        for segment in segments:
            start_time = cls.seconds_to_vtt_time(segment["start"])
            end_time = cls.seconds_to_vtt_time(segment["end"])
            text = segment["text"]

            vtt_lines.append(f"{start_time} --> {end_time}")
            vtt_lines.append(f"{text}")
            vtt_lines.append("")

        return "\n".join(vtt_lines)

    @classmethod
    def generate_json(cls, segments: List[Dict], full_text: str = "") -> Dict:
        """生成 JSON 格式字幕"""
        return {
            "text": full_text,
            "segments": segments
        }
