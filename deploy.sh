#!/bin/bash

# FunASR 服务一键部署脚本
# 支持: macOS (Intel/Apple Silicon), Linux
# 支持: Docker 部署和本地部署
# 支持: 国内源加速下载

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 服务配置
SERVICE_NAME="funasr-service"
SERVICE_PORT=7860

# 国内 PyPI 镜像
PYPI_MIRRORS=(
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.mirrors.ustc.edu.cn/simple"
)

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux"* ]]; then
        echo "linux"
    else
        echo "unknown"
    fi
}

# 检测架构
detect_arch() {
    arch=$(uname -m)
    if [[ "$arch" == "arm64" ]] || [[ "$arch" == "aarch64" ]]; then
        echo "arm64"
    else
        echo "amd64"
    fi
}

# 检测是否 Apple Silicon
detect_apple_silicon() {
    if [[ "$(uname -m)" == "arm64" ]] && [[ "$(detect_os)" == "macos" ]]; then
        echo "true"
    else
        echo "false"
    fi
}

# 打印信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# 检测 CUDA 版本
# CUDA 13.x/12.x 向下兼容 cu121 轮子，CUDA 11.x 使用 cu118
detect_cuda() {
    if check_command nvidia-smi; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$CUDA_VER" ]]; then
            CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
            if [[ "$CUDA_MAJOR" -ge 12 ]]; then
                echo "cu121"
            elif [[ "$CUDA_MAJOR" == "11" ]]; then
                echo "cu118"
            else
                echo "cu121"
            fi
            return
        fi
    fi
    echo "cpu"
}

# 检查网络连接（测试国内源可用性）
check_network() {
    print_step "检查网络连接..."

    # 测试国内源
    for mirror in "${PYPI_MIRRORS[@]}"; do
        if curl -s --max-time 5 "$mirror" > /dev/null 2>&1; then
            print_info "国内 PyPI 源可用: $mirror"
            return 0
        fi
    done

    print_warn "国内 PyPI 源不可用，将使用默认源"
    return 1
}

# 获取最佳 PyPI 镜像
get_best_mirror() {
    for mirror in "${PYPI_MIRRORS[@]}"; do
        if curl -s --max-time 3 "$mirror" > /dev/null 2>&1; then
            echo "$mirror"
            return
        fi
    done
    echo "https://pypi.org/simple"
}

# 检查环境
check_environment() {
    print_step "检查环境..."

    OS=$(detect_os)
    ARCH=$(detect_arch)
    APPLE_SILICON=$(detect_apple_silicon)

    print_info "操作系统: $OS"
    print_info "架构: $ARCH"
    print_info "Apple Silicon: $APPLE_SILICON"

    if [[ "$OS" == "unknown" ]]; then
        print_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi

    # 检查 Docker
    if ! check_command docker; then
        print_warn "Docker 未安装，Docker 模式不可用"
    fi

    # 检查 Python
    if ! check_command python3; then
        print_error "Python3 未安装，请先安装 Python3"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_info "Python 版本: $PYTHON_VERSION"

    # 检查网络
    check_network

    print_info "环境检查通过"
}

# 安装系统依赖（macOS）
install_macos_deps() {
    print_step "安装 macOS 系统依赖..."

    # 检查 Homebrew
    if ! check_command brew; then
        print_error "Homebrew 未安装，请先安装 Homebrew"
        print_info "安装命令: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    # 安装 ffmpeg 和 libsndfile
    print_info "安装 ffmpeg 和 libsndfile..."
    brew install ffmpeg libsndfile || true

    print_info "macOS 依赖安装完成"
}

# 安装系统依赖（Linux）
install_linux_deps() {
    print_step "安装 Linux 系统依赖..."

    # 检测包管理器
    if check_command apt-get; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y libsndfile1 ffmpeg git
    elif check_command yum; then
        # CentOS/RHEL
        sudo yum install -y libsndfile ffmpeg git
    elif check_command dnf; then
        # Fedora
        sudo dnf install -y libsndfile ffmpeg git
    else
        print_warn "未知的包管理器，请手动安装 libsndfile 和 ffmpeg"
    fi

    print_info "Linux 依赖安装完成"
}

# 安装系统依赖
install_system_deps() {
    OS=$(detect_os)

    if [[ "$OS" == "macos" ]]; then
        install_macos_deps
    elif [[ "$OS" == "linux" ]]; then
        install_linux_deps
    fi
}

# 安装 PyTorch（根据平台选择）
install_pytorch() {
    # 检查是否已安装
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_info "PyTorch 已安装 (版本: $TORCH_VER)，跳过"
        return
    fi

    print_step "安装 PyTorch..."

    OS=$(detect_os)
    ARCH=$(detect_arch)
    APPLE_SILICON=$(detect_apple_silicon)

    # PyTorch 官方源（国内镜像通常没有 PyTorch 轮子）
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"

    if [[ "$APPLE_SILICON" == "true" ]]; then
        # Apple Silicon (M1/M2/M3) - 从 PyPI 安装（自带 MPS 支持）
        print_info "检测到 Apple Silicon，安装 MPS 支持的 PyTorch..."
        pip install torch torchvision torchaudio
    elif [[ "$OS" == "macos" ]] && [[ "$ARCH" == "amd64" ]]; then
        # macOS Intel
        print_info "检测到 macOS Intel，安装 CPU 版本 PyTorch..."
        pip install torch torchvision torchaudio
    elif [[ "$OS" == "linux" ]] && [[ "$ARCH" == "arm64" ]]; then
        # Linux ARM64
        print_info "检测到 Linux ARM64，安装 PyTorch..."
        pip install torch torchvision torchaudio
    else
        # Linux x86_64 - 自动检测 CUDA
        CUDA_TAG=$(detect_cuda)
        if [[ "$CUDA_TAG" != "cpu" ]]; then
            print_info "检测到 NVIDIA GPU (CUDA: $CUDA_TAG)，安装 GPU 版本 PyTorch..."
            pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
        else
            print_info "未检测到 GPU，安装 CPU 版本 PyTorch..."
            pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"
        fi
    fi

    print_info "PyTorch 安装完成"
}

# 安装 Python 依赖
install_python_deps() {
    print_step "安装 Python 依赖..."

    # 检查虚拟环境
    if [ ! -d "venv" ]; then
        print_info "创建虚拟环境..."
        python3 -m venv venv
    fi

    # 激活虚拟环境
    source venv/bin/activate

    # 升级 pip
    pip install --upgrade pip

    # 获取最佳镜像
    MIRROR=$(get_best_mirror)
    print_info "使用 PyPI 镜像: $MIRROR"

    # 安装 PyTorch（平台特定）
    install_pytorch

    # 安装其他依赖
    print_info "安装项目依赖..."
    pip install -r requirements.txt --index-url "$MIRROR"

    print_info "Python 依赖安装完成"
}

# 本地安装
install_local() {
    print_step "开始本地安装..."

    check_environment
    install_system_deps
    install_python_deps

    print_info "本地安装完成！"
    print_info "启动命令: ./deploy.sh start-local"
}

# 构建 Docker 镜像
build_docker() {
    print_step "构建 Docker 镜像..."

    # 检测 Apple Silicon
    if [[ "$(detect_apple_silicon)" == "true" ]]; then
        print_info "检测到 Apple Silicon，使用 linux/arm64 平台构建..."
        docker buildx build --platform linux/arm64 -t ${SERVICE_NAME}:latest .
    else
        docker build -t ${SERVICE_NAME}:latest .
    fi

    print_info "Docker 镜像构建完成"
}

# 启动本地服务
start_local() {
    print_step "启动本地服务..."

    # 检查虚拟环境
    if [ ! -d "venv" ]; then
        print_error "虚拟环境不存在，请先运行: ./deploy.sh install"
        exit 1
    fi

    # 激活虚拟环境
    source venv/bin/activate

    # 启动服务
    print_info "启动 FunASR 服务..."
    print_info "访问地址: http://localhost:${SERVICE_PORT}"
    print_info "按 Ctrl+C 停止服务"
    echo ""

    python app.py
}

# 启动 Docker 服务
start_docker() {
    print_step "启动 Docker 服务..."

    # 检查 Docker
    if ! check_command docker; then
        print_error "Docker 未安装"
        exit 1
    fi

    # 创建模型目录
    mkdir -p models

    # 检测 Apple Silicon
    if [[ "$(detect_apple_silicon)" == "true" ]]; then
        print_info "检测到 Apple Silicon，使用 Docker Compose 启动..."
        docker-compose up -d
    else
        docker-compose up -d
    fi

    print_info "服务已启动，访问 http://localhost:${SERVICE_PORT}"
}

# 停止服务
stop_service() {
    print_step "停止服务..."

    # 停止 Docker 服务
    if check_command docker; then
        docker-compose down 2>/dev/null || true
    fi

    # 停止本地服务（查找并杀死进程）
    pkill -f "python app.py" 2>/dev/null || true

    print_info "服务已停止"
}

# 查看服务状态
status() {
    print_step "查看服务状态..."

    # 查看 Docker 容器状态
    if check_command docker; then
        docker-compose ps 2>/dev/null || true
    fi

    # 查看本地进程
    echo ""
    echo "本地进程:"
    ps aux | grep "python app.py" | grep -v grep || true
}

# 查看日志
logs() {
    print_step "查看服务日志..."

    if check_command docker; then
        docker-compose logs -f
    fi
}

# 打包服务
package() {
    print_step "打包服务..."

    # 创建打包目录
    PACKAGE_NAME="${SERVICE_NAME}-$(date +%Y%m%d).tar.gz"

    # 打包项目
    tar -czf ${PACKAGE_NAME} \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='./models' \
        --exclude='*.tar.gz' \
        .

    print_info "打包完成: ${PACKAGE_NAME}"
    print_info "文件大小: $(du -h ${PACKAGE_NAME} | cut -f1)"
}

# 显示帮助信息
show_help() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           FunASR 服务一键部署脚本                            ║"
    echo "║           支持: macOS (Intel/Apple Silicon) / Linux        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  install       安装本地依赖（推荐首次使用）"
    echo "  build         构建 Docker 镜像"
    echo "  start         启动 Docker 服务"
    echo "  start-local   启动本地服务（需要提前 install）"
    echo "  stop          停止服务"
    echo "  status        查看服务状态"
    echo "  logs          查看服务日志"
    echo "  package       打包服务"
    echo "  help          显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 install        # 首次安装"
    echo "  $0 start-local    # 本地启动"
    echo "  $0 build          # 构建 Docker 镜像"
    echo "  $0 start          # Docker 启动"
    echo "  $0 package        # 打包分发"
    echo ""
    echo "注意:"
    echo "  - macOS 需要安装 Homebrew"
    echo "  - Apple Silicon (M1/M2/M3) 会自动检测并使用 MPS 加速"
    echo "  - 模型文件会下载到 models/ 目录"
    echo "  - 国内源会自动检测并使用"
    echo ""
}

# 主函数
main() {
    case "${1:-help}" in
        install)
            install_local
            ;;
        build)
            check_environment
            build_docker
            ;;
        start)
            start_docker
            ;;
        start-local)
            start_local
            ;;
        stop)
            stop_service
            ;;
        status)
            status
            ;;
        logs)
            logs
            ;;
        package)
            package
            ;;
        help|*)
            show_help
            ;;
    esac
}

# 执行主函数
main "$@"
