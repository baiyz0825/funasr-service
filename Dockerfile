# 使用 Python 3.10 作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置国内镜像源（用于非 PyTorch 包）
ARG PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY requirements.txt .
COPY app.py .
COPY config.py .
COPY src/ ./src/
COPY static/ ./static/

# 创建模型缓存目录
RUN mkdir -p /app/models

# 安装 PyTorch（从官方源，国内镜像可能没有 PyTorch 轮子）
RUN pip install --no-cache-dir torch torchvision torchaudio

# 安装其他依赖（torch 已安装会自动跳过）
RUN pip install --no-cache-dir -r requirements.txt \
    -i ${PYPI_MIRROR} --trusted-host pypi.tuna.tsinghua.edu.cn

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]
