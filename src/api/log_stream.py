"""SSE 日志流：将服务端日志实时推送到前端"""

import asyncio
import json
import logging
import time
from collections import deque
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# 存储最近 500 条日志
_log_buffer = deque(maxlen=500)
# 所有活跃的 SSE 连接
_subscribers: list[asyncio.Queue] = []


class SSELogHandler(logging.Handler):
    """自定义日志 Handler，广播到所有 SSE 订阅者"""

    def emit(self, record):
        try:
            entry = {
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname.lower(),
                "name": record.name,
                "msg": self.format(record),
            }
            _log_buffer.append(entry)
            for q in _subscribers:
                try:
                    q.put_nowait(entry)
                except (asyncio.QueueFull, Exception):
                    pass
        except Exception:
            pass


def setup_log_streaming():
    """在 app 启动时调用，挂载 SSE handler 到 root logger"""
    handler = SSELogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)


@router.get(
    "/api/logs/stream",
    summary="实时日志流",
    description="SSE 端点，前端通过 EventSource 连接获取服务端实时日志。连接后先推送历史日志，再持续推送新日志，30 秒无新日志发送心跳。",
    tags=["日志"],
    responses={200: {"description": "SSE 事件流，每条日志为 JSON 格式: {time, level, name, msg}"}},
)
async def log_stream():
    """SSE 端点：前端 EventSource 连接此地址获取实时日志"""

    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.append(queue)

    async def event_generator():
        try:
            # 先发送缓冲区中的历史日志
            for entry in _log_buffer:
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)  # 强制刷新缓冲区

            # 持续推送新日志
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # 强制刷新，避免批量延迟
                except asyncio.TimeoutError:
                    # 心跳保活
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
