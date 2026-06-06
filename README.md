# FunASR 语音转写服务

基于 [FunASR](https://github.com/modelscope/FunASR) 的语音转写服务，支持实时流式转写、离线识别、说话人分离、情感检测。提供 OpenAI 兼容 API、WebSocket 流式协议和 Web 管理界面。

## 功能特性

- **实时转写** - WebSocket 流式识别，延迟约 600ms（Paraformer-Streaming）
- **离线识别** - 上传音频文件，生成文字 / SRT / VTT / JSON 字幕
- **说话人分离** - 自动识别"谁说了什么"（cam++ 模型）
- **情感检测** - 识别语音中的 7 种情感状态（SenseVoice）
- **OpenAI 兼容 API** - `/v1/audio/transcriptions`，可对接 LangChain / Dify 等
- **多模型管理** - 支持同时加载多个模型，按需切换，VRAM 自动检测
- **配置持久化** - 启动自动加载模型、CPU 线程数等配置保存到 `config.json`
- **Web 管理界面** - 7 个场景标签页 + 实时日志面板，一站式操作
- **远程部署** - 一键部署到 Linux 服务器，自动创建 systemd 服务

## 快速开始

### 方式一：本地部署（推荐）

```bash
# 一键安装（自动检测平台、CUDA 版本、配置国内镜像）
chmod +x deploy.sh
./deploy.sh install

# 启动服务
./deploy.sh start-local
```

### 方式二：手动安装

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 PyTorch（根据平台选择）
# Apple Silicon / macOS Intel:
pip install torch torchvision torchaudio
# Linux CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Linux CUDA 11.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt

# 启动
python app.py
```

### 方式三：Docker

```bash
# 使用 docker-compose（推荐）
docker-compose up -d

# 或手动构建
docker build -t funasr-service .
docker run -d -p 7860:7860 -v ./models:/app/models -v ./config.json:/app/config.json funasr-service
```

### 方式四：远程部署

```bash
# 部署到远程 Linux 服务器（自动检测 CUDA、创建 systemd 服务）
chmod +x deploy-remote.sh
./deploy-remote.sh root@192.168.1.100

# 指定 SSH 端口和安装目录
./deploy-remote.sh user@server.com -p 2222 -d /data/funasr
```

### 访问服务

| 地址 | 说明 |
|------|------|
| http://localhost:7860 | Web 管理界面 |
| http://localhost:7860/docs | Swagger API 文档 |
| ws://localhost:7860/ws/stream | WebSocket 流式识别 |
| http://localhost:7860/v1/health | 健康检查 |

## 模型列表

| 模型 ID | 名称 | 大小 | 场景 | 流式 | 说明 |
|---------|------|------|------|------|------|
| `sensevoice` | SenseVoice-Small | 1.5GB | 情感+事件检测 | 否 | 5 种语言，极快，支持情感检测 |
| `paraformer` | Paraformer-zh | 2.0GB | 中文高精度离线 | 否 | 中文生产级，适合长音频，自带标点 |
| `paraformer-streaming` | Paraformer-zh-Streaming | 2.0GB | 实时字幕 | 600ms | 中文流式识别，低延迟 |
| `funasr-nano` | Fun-ASR-Nano | 3.0GB | 多语言 | 否 | 31 种语言高精度，自带标点 |

首次使用需在 Web UI「模型管理」页面加载模型，模型会自动下载到 `models/` 目录。

> **注意**：FunASR 的 `AutoModel` 不支持 `cache_dir` 参数，模型下载目录通过环境变量 `MODELSCOPE_CACHE` 和 `HF_HOME` 控制，服务已自动配置。

## 配置说明

### 配置文件

配置保存在项目根目录 `config.json`，首次启动时自动创建默认配置：

```json
{
  "auto_load_models": ["sensevoice", "paraformer-streaming"],
  "default_device": "cuda",
  "ncpu": 0,
  "max_loaded": 2
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_load_models` | string[] | `[]` | 服务启动时自动加载的模型 ID 列表 |
| `default_device` | string | 自动检测 | 推理设备：`cuda` / `mps` / `cpu`（自动检测优先级：CUDA > MPS > CPU） |
| `ncpu` | int | `0` | CPU 推理线程数，`0` = 自动检测物理核心数（上限 16） |
| `max_loaded` | int | `2` | 最大同时加载模型数（1-4） |

配置可通过 Web UI「模型管理」页面修改并保存，也可通过 API 修改：

```bash
# 读取配置
curl http://localhost:7860/v1/config

# 更新配置
curl -X POST http://localhost:7860/v1/config \
  -H "Content-Type: application/json" \
  -d '{"auto_load_models": ["sensevoice"], "default_device": "cuda"}'
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `MODELSCOPE_CACHE` | ModelScope 模型下载目录（默认：`./models`） |
| `HF_HOME` | HuggingFace 模型缓存目录（默认：`./models/hf`） |
| `CUDA_VISIBLE_DEVICES` | GPU 设备选择，如 `0,1` 使用双卡 |

### CPU 多核加速

CPU 推理时服务会自动检测物理核心数并设置线程数（上限 16），覆盖 FunASR 默认的 4 线程。可通过 `ncpu` 配置项手动指定。

## API 文档

### 转录接口

```
POST /v1/audio/transcriptions
```

上传音频文件进行语音转录，支持多种输出格式和高级功能。

**请求参数（form-data）：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | 必填 | 音频文件（WAV, MP3, FLAC, M4A 等） |
| `model` | string | `sensevoice` | 模型 ID |
| `language` | string | `zh` | 语言代码 |
| `response_format` | string | `json` | 输出格式：`json` / `verbose_json` / `srt` / `vtt` |
| `enable_speaker_diarization` | bool | `false` | 启用说话人分离（使用 cam++ 模型） |
| `enable_emotion_detection` | bool | `false` | 启用情感检测（仅 SenseVoice） |

**响应格式示例：**

```bash
# JSON 格式
curl -X POST http://localhost:7860/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=sensevoice" \
  -F "response_format=json"

# SRT 字幕
curl -X POST http://localhost:7860/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=paraformer" \
  -F "response_format=srt"

# 说话人分离 + 详细 JSON
curl -X POST http://localhost:7860/v1/audio/transcriptions \
  -F "file=@meeting.wav" \
  -F "model=paraformer" \
  -F "enable_speaker_diarization=true" \
  -F "response_format=verbose_json"
```

**verbose_json 响应结构：**

```json
{
  "text": "完整文本",
  "duration": 12.5,
  "language": "zh",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "你好",
      "speaker": "speaker_1",
      "emotions": [{"type": "NEUTRAL", "label": "中性"}]
    }
  ],
  "emotions": [{"type": "NEUTRAL", "label": "中性"}]
}
```

### 模型管理接口

```
GET  /v1/models                    # 列出所有模型及加载状态
GET  /v1/models/loaded             # 列出已加载模型
POST /v1/models/{model_id}/load    # 加载模型（body: {"device": "cuda"}）
POST /v1/models/{model_id}/unload  # 卸载模型
GET  /v1/devices                   # 列出可用设备（CPU/GPU 显存信息）
GET  /v1/health                    # 健康检查
```

### 配置接口

```
GET  /v1/config                    # 读取当前配置
POST /v1/config                    # 更新配置（保存到 config.json）
```

### 日志流

```
GET /api/logs/stream               # SSE 实时日志推送
```

返回 Server-Sent Events 流，每条事件为 JSON：

```json
{"time": "2026-06-06 18:45:55", "level": "INFO", "name": "funasr-service", "msg": "模型加载完成"}
```

### WebSocket 流式识别

**端点：** `ws://host:port/ws/stream`

**协议流程：** 配置 → 音频数据* → 结束

**客户端消息：**

```javascript
// 1. 发送配置（必须第一条）
ws.send(JSON.stringify({
    type: 'config',
    model: 'paraformer-streaming'  // 使用流式模型
}));

// 2. 发送音频数据（16kHz 单声道 PCM int16，base64 编码）
const base64 = btoa(String.fromCharCode(...new Uint8Array(pcmBuffer)));
ws.send(JSON.stringify({type: 'audio', data: base64}));

// 3. 发送结束信号
ws.send(JSON.stringify({type: 'end'}));
```

**服务端消息：**

```json
// 配置确认
{"type": "config_ack", "model": "paraformer-streaming", "streaming": true}

// 识别结果
{"type": "result", "text": "部分文本", "is_final": false, "accumulated": "已确认的全部文本"}

// 最终结果（is_final=true，在发送 end 后返回）
{"type": "result", "text": "最终文本", "is_final": true, "accumulated": "完整文本"}

// 错误
{"type": "error", "message": "错误描述"}
```

**完整 JavaScript 示例：**

```javascript
const ws = new WebSocket('ws://localhost:7860/ws/stream');

ws.onopen = () => {
    ws.send(JSON.stringify({type: 'config', model: 'paraformer-streaming'}));
};

ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'result') {
        console.log('识别文本:', d.accumulated);
        if (d.is_final) console.log('识别完成');
    }
};

// 麦克风采集并发送
const audioContext = new AudioContext({sampleRate: 16000});
// ... 采集 PCM 数据后:
ws.send(JSON.stringify({type: 'audio', data: base64PCM}));

// 结束
ws.send(JSON.stringify({type: 'end'}));
```

### Python 客户端示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7860/v1", api_key="dummy")

# 基础转录
result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("audio.wav", "rb"),
    response_format="verbose_json"
)
print(result.text)

# 说话人分离
import requests
resp = requests.post(
    "http://localhost:7860/v1/audio/transcriptions",
    files={"file": open("meeting.wav", "rb")},
    data={
        "model": "paraformer",
        "enable_speaker_diarization": "true",
        "response_format": "verbose_json"
    }
)
for seg in resp.json()["segments"]:
    print(f"[{seg['speaker']}] {seg['text']}")
```

## Web 管理界面

访问 http://localhost:7860 打开 Web UI，包含 7 个功能标签页：

| 标签页 | 功能 |
|--------|------|
| 首页 | 场景引导、模型推荐、配置说明 |
| 实时转写 | 麦克风录音，WebSocket 流式识别 |
| 离线识别 | 上传音频文件，生成文字/字幕 |
| 说话人分离 | 识别"谁说了什么"，导出带说话人标签的 SRT |
| 情感检测 | 识别语音中的情感状态 |
| 模型管理 | 加载/卸载模型、设置自动加载、查看设备信息 |
| API 文档 | 内嵌 Swagger UI |

右侧面板实时显示服务日志（SSE 推送）。

## 项目结构

```
funasr-service/
├── app.py                      # FastAPI 主应用入口
├── config.py                   # 模型配置和服务器常量
├── config.json                 # 运行时配置（自动创建）
├── deploy.sh                   # 本地/Docker 一键部署脚本
├── deploy-remote.sh            # 远程 Linux 服务器部署脚本
├── Dockerfile                  # Docker 镜像定义
├── docker-compose.yml          # Docker Compose 编排
├── requirements.txt            # Python 依赖
├── static/
│   └── index.html              # Web 管理界面（单文件，深色主题）
├── src/
│   ├── api/
│   │   ├── openai.py           # OpenAI 兼容转录 API
│   │   ├── management.py       # 模型管理 API
│   │   ├── websocket.py        # WebSocket 流式识别
│   │   ├── config.py           # 配置管理 API
│   │   └── log_stream.py       # SSE 日志流
│   ├── models/
│   │   └── model_manager.py    # 多模型生命周期管理器
│   └── utils/
│       ├── audio.py            # 音频加载/重采样
│       └── subtitle.py         # SRT/VTT/JSON 字幕生成
└── models/                     # 模型缓存目录（自动创建）
```

## 部署脚本

### deploy.sh — 本地/Docker 部署

```bash
./deploy.sh install      # 安装本地依赖（自动检测平台和 CUDA）
./deploy.sh start-local  # 启动本地服务
./deploy.sh build        # 构建 Docker 镜像
./deploy.sh start        # Docker Compose 启动
./deploy.sh stop         # 停止服务
./deploy.sh status       # 查看服务状态
./deploy.sh logs         # 查看 Docker 日志
./deploy.sh package      # 打包为 tar.gz 分发
```

特性：
- 自动检测 macOS (Intel/Apple Silicon) 和 Linux
- 自动检测 CUDA 版本并安装对应 PyTorch（cu121 / cu118 / cpu）
- 自动选择最快的国内 PyPI 镜像（清华 / 阿里云 / 中科大）
- Apple Silicon 自动使用 MPS 加速

### deploy-remote.sh — 远程服务器部署

```bash
# 基本用法
./deploy-remote.sh root@192.168.1.100

# 指定 SSH 端口
./deploy-remote.sh user@server.com -p 2222

# 指定安装目录和服务端口
./deploy-remote.sh root@gpu-server -d /data/funasr --service-port 8080
```

自动执行 7 个步骤：
1. 检查 SSH 连接
2. 检测远程环境（OS、架构、Python、CUDA、内存、磁盘）
3. 打包项目文件
4. 上传到远程服务器
5. 安装系统依赖和 Python 依赖（自动检测 CUDA 版本）
6. 创建 systemd 服务（开机自启、自动重启、日志输出到 journal）
7. 启动服务并验证

部署后管理：

```bash
ssh user@host 'systemctl status funasr-service'   # 查看状态
ssh user@host 'journalctl -u funasr-service -f'    # 查看日志
ssh user@host 'systemctl restart funasr-service'    # 重启服务
ssh user@host 'systemctl stop funasr-service'       # 停止服务
```

## 系统要求

- **Python**: 3.10+
- **系统依赖**: ffmpeg, libsndfile
- **内存**: 建议 8GB+（加载模型需要 2-4GB）
- **GPU**（可选）: NVIDIA GPU + CUDA 11.8+ / 12.x，推荐 4GB+ 显存
- **磁盘**: 每个模型 1.5-3GB

## 许可证

MIT
