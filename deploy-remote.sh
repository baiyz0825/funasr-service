#!/bin/bash

# FunASR 远程部署脚本
# 将服务部署到远程 Linux 服务器，自动创建 systemd 服务
# 支持自动检测 CUDA / CPU，配置 systemd 开机自启

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SERVICE_NAME="funasr-service"
SERVICE_PORT=7860
REMOTE_DIR="/opt/funasr-service"
REMOTE_USER=""
REMOTE_HOST=""
REMOTE_SSH_PORT="22"

print_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_step()    { echo -e "${BLUE}[STEP]${NC} $1"; }
print_remote()  { echo -e "${CYAN}[REMOTE]${NC} $1"; }

usage() {
    echo ""
    echo "用法: $0 <user@host> [选项]"
    echo ""
    echo "将 FunASR 服务部署到远程 Linux 服务器并创建 systemd 服务。"
    echo ""
    echo "参数:"
    echo "  user@host           远程服务器用户名和地址"
    echo ""
    echo "选项:"
    echo "  -p, --port PORT     SSH 端口 (默认: 22)"
    echo "  -d, --dir PATH      远程安装目录 (默认: /opt/funasr-service)"
    echo "  --service-port PORT 服务端口 (默认: 7860)"
    echo "  -h, --help          显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 root@192.168.1.100"
    echo "  $0 user@server.com -p 2222"
    echo "  $0 root@gpu-server -d /data/funasr --service-port 8080"
    echo ""
    echo "部署完成后可使用以下命令管理服务:"
    echo "  ssh user@host 'systemctl start funasr-service'"
    echo "  ssh user@host 'systemctl stop funasr-service'"
    echo "  ssh user@host 'systemctl status funasr-service'"
    echo "  ssh user@host 'journalctl -u funasr-service -f'"
    echo ""
}

# 解析参数
parse_args() {
    if [[ $# -lt 1 ]]; then
        usage
        exit 1
    fi

    REMOTE_HOST="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--port)
                REMOTE_SSH_PORT="$2"; shift 2 ;;
            -d|--dir)
                REMOTE_DIR="$2"; shift 2 ;;
            --service-port)
                SERVICE_PORT="$2"; shift 2 ;;
            -h|--help)
                usage; exit 0 ;;
            *)
                print_error "未知参数: $1"; usage; exit 1 ;;
        esac
    done

    # 分离 user 和 host
    if [[ "$REMOTE_HOST" == *"@"* ]]; then
        REMOTE_USER="${REMOTE_HOST%%@*}"
        REMOTE_HOST="${REMOTE_HOST#*@}"
    else
        REMOTE_USER="root"
    fi
}

SSH_CMD="ssh -p ${REMOTE_SSH_PORT} ${REMOTE_USER}@${REMOTE_HOST}"
SCP_CMD="scp -P ${REMOTE_SSH_PORT}"

# 检查 SSH 连接
check_ssh() {
    print_step "检查 SSH 连接..."
    if ! ${SSH_CMD} "echo ok" > /dev/null 2>&1; then
        print_error "无法连接到 ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SSH_PORT}"
        print_error "请检查 SSH 配置和网络连接"
        exit 1
    fi
    print_info "SSH 连接成功"
}

# 检查远程环境
check_remote_env() {
    print_step "检查远程环境..."

    # 检测操作系统
    REMOTE_OS=$(${SSH_CMD} "uname -s" 2>/dev/null)
    if [[ "$REMOTE_OS" != "Linux" ]]; then
        print_error "远程系统不是 Linux ($REMOTE_OS)，不支持 systemd"
        exit 1
    fi
    print_info "远程系统: Linux"

    # 检测架构
    REMOTE_ARCH=$(${SSH_CMD} "uname -m" 2>/dev/null)
    print_info "架构: $REMOTE_ARCH"

    # 检测 Python
    REMOTE_PYTHON=$(${SSH_CMD} "python3 --version 2>&1 || echo 'NOT_FOUND'" 2>/dev/null)
    if [[ "$REMOTE_PYTHON" == *"NOT_FOUND"* ]]; then
        print_error "远程服务器未安装 Python3"
        exit 1
    fi
    print_info "Python: $REMOTE_PYTHON"

    # 检测 CUDA
    REMOTE_CUDA=$(${SSH_CMD} "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'NO_GPU'" 2>/dev/null)
    if [[ "$REMOTE_CUDA" == *"NO_GPU"* ]] || [[ -z "$REMOTE_CUDA" ]]; then
        REMOTE_DEVICE="cpu"
        print_info "GPU: 未检测到，将使用 CPU 推理"
    else
        REMOTE_CUDA_VERSION=$(${SSH_CMD} "nvidia-smi 2>/dev/null | grep 'CUDA Version' | awk '{print \$9}'" 2>/dev/null)
        REMOTE_DEVICE="cuda"
        print_info "GPU: $REMOTE_CUDA (CUDA $REMOTE_CUDA_VERSION)"
    fi

    # 检测内存
    REMOTE_MEM=$(${SSH_CMD} "free -h 2>/dev/null | awk '/^Mem:/{print \$2}'" 2>/dev/null)
    print_info "内存: $REMOTE_MEM"

    # 检测 systemd
    if ! ${SSH_CMD} "systemctl --version" > /dev/null 2>&1; then
        print_error "远程服务器不支持 systemd"
        exit 1
    fi
    print_info "systemd: 可用"

    # 检测磁盘空间
    REMOTE_DISK=$(${SSH_CMD} "df -h ${REMOTE_DIR%/*} 2>/dev/null | awk 'NR==2{print \$4}'" 2>/dev/null)
    print_info "可用磁盘: $REMOTE_DISK"
}

# 打包项目
package_project() {
    print_step "打包项目文件..."

    PACKAGE_FILE="/tmp/${SERVICE_NAME}-deploy.tar.gz"
    tar -czf "$PACKAGE_FILE" \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.claude' \
        --exclude='./models' \
        --exclude='*.tar.gz' \
        --exclude='funasr-service.log' \
        -C "$(dirname "$0")" .

    PACKAGE_SIZE=$(du -h "$PACKAGE_FILE" | cut -f1)
    print_info "打包完成: $PACKAGE_FILE ($PACKAGE_SIZE)"
}

# 上传到远程
upload_to_remote() {
    print_step "上传文件到远程服务器..."

    # 创建远程目录
    ${SSH_CMD} "sudo mkdir -p ${REMOTE_DIR} && sudo chown ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_DIR}"

    # 上传打包文件
    ${SCP_CMD} "$PACKAGE_FILE" "${REMOTE_USER}@${REMOTE_HOST}:/tmp/${SERVICE_NAME}-deploy.tar.gz"

    # 解压
    ${SSH_CMD} "cd ${REMOTE_DIR} && tar -xzf /tmp/${SERVICE_NAME}-deploy.tar.gz && rm -f /tmp/${SERVICE_NAME}-deploy.tar.gz"

    print_info "文件上传完成"
}

# 远程安装依赖
install_remote_deps() {
    print_step "在远程服务器安装依赖..."

    ${SSH_CMD} bash -s "$REMOTE_DIR" <<'REMOTE_SCRIPT'
set -e
INSTALL_DIR="$1"

echo "[REMOTE] 安装系统依赖..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq libsndfile1 ffmpeg python3-venv python3-pip > /dev/null 2>&1
elif command -v yum &> /dev/null; then
    sudo yum install -y -q libsndfile ffmpeg python3-pip > /dev/null 2>&1
elif command -v dnf &> /dev/null; then
    sudo dnf install -y -q libsndfile ffmpeg python3-pip > /dev/null 2>&1
fi

echo "[REMOTE] 创建虚拟环境..."
cd "$INSTALL_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "[REMOTE] 升级 pip..."
pip install --upgrade pip -q

# 检测 CUDA 并安装对应 PyTorch
# CUDA 13.x/12.x 向下兼容 cu121 轮子，CUDA 11.x 使用 cu118
if command -v nvidia-smi &> /dev/null 2>&1; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
    if [[ "$CUDA_VER" -ge 12 ]]; then
        echo "[REMOTE] 安装 CUDA 12.1 PyTorch (CUDA $CUDA_VER 向下兼容)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    elif [[ "$CUDA_VER" == "11" ]]; then
        echo "[REMOTE] 安装 CUDA 11.8 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    else
        echo "[REMOTE] 安装 CPU PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
    fi
else
    echo "[REMOTE] 安装 CPU PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

echo "[REMOTE] 安装项目依赖..."
pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt

echo "[REMOTE] 依赖安装完成"
REMOTE_SCRIPT

    print_info "远程依赖安装完成"
}

# 创建 systemd 服务
create_systemd_service() {
    print_step "创建 systemd 服务..."

    # 获取远程服务器 IP（用于描述中的访问地址）
    SERVER_IP="${REMOTE_HOST}"

    # 确定 Python 路径和设备信息
    PYTHON_PATH="${REMOTE_DIR}/venv/bin/python"
    DEVICE_INFO=""
    if [[ "$REMOTE_DEVICE" == "cuda" ]]; then
        DEVICE_INFO="CUDA GPU 加速"
    else
        DEVICE_INFO="CPU 推理"
    fi

    # 写入 systemd 服务文件
    ${SSH_CMD} "sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null" <<EOF
[Unit]
Description=FunASR 语音转写服务 - 基于 FunASR 的实时/离线语音识别，支持说话人分离和情感检测 (${DEVICE_INFO}) | Web UI: http://${SERVER_IP}:${SERVICE_PORT} | API Docs: http://${SERVER_IP}:${SERVICE_PORT}/docs | WebSocket: ws://${SERVER_IP}:${SERVICE_PORT}/ws/stream
Documentation=http://${SERVER_IP}:${SERVICE_PORT}/docs
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${REMOTE_USER}
Group=${REMOTE_USER}
WorkingDirectory=${REMOTE_DIR}
ExecStart=${PYTHON_PATH} app.py
Restart=on-failure
RestartSec=10
StartLimitBurst=5
StartLimitIntervalSec=60

# 环境变量
Environment=PYTHONUNBUFFERED=1
Environment=MODELSCOPE_CACHE=${REMOTE_DIR}/models
Environment=HF_HOME=${REMOTE_DIR}/models/hf

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

    # 重载 systemd
    ${SSH_CMD} "sudo systemctl daemon-reload"

    # 创建模型目录
    ${SSH_CMD} "mkdir -p ${REMOTE_DIR}/models"

    print_info "systemd 服务创建完成"
}

# 启动服务
start_remote_service() {
    print_step "启动远程服务..."

    ${SSH_CMD} "sudo systemctl enable ${SERVICE_NAME} && sudo systemctl start ${SERVICE_NAME}"

    # 等待服务启动
    sleep 3

    # 检查服务状态
    STATUS=$(${SSH_CMD} "systemctl is-active ${SERVICE_NAME}" 2>/dev/null || echo "inactive")
    if [[ "$STATUS" == "active" ]]; then
        print_info "服务启动成功！"
    else
        print_warn "服务可能启动失败，请检查日志:"
        print_warn "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'journalctl -u ${SERVICE_NAME} -n 50 --no-pager'"
    fi
}

# 打印部署结果
print_summary() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              FunASR 远程部署完成                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  服务器:     ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SSH_PORT}"
    echo "  安装目录:   ${REMOTE_DIR}"
    echo "  推理设备:   ${REMOTE_DEVICE^^}"
    echo "  服务端口:   ${SERVICE_PORT}"
    echo ""
    echo "  访问地址:"
    echo "    Web UI:    http://${REMOTE_HOST}:${SERVICE_PORT}"
    echo "    API 文档:  http://${REMOTE_HOST}:${SERVICE_PORT}/docs"
    echo "    WebSocket: ws://${REMOTE_HOST}:${SERVICE_PORT}/ws/stream"
    echo "    健康检查:  http://${REMOTE_HOST}:${SERVICE_PORT}/v1/health"
    echo ""
    echo "  管理命令:"
    echo "    查看状态:  ssh ${REMOTE_USER}@${REMOTE_HOST} 'sudo systemctl status ${SERVICE_NAME}'"
    echo "    查看日志:  ssh ${REMOTE_USER}@${REMOTE_HOST} 'sudo journalctl -u ${SERVICE_NAME} -f'"
    echo "    重启服务:  ssh ${REMOTE_USER}@${REMOTE_HOST} 'sudo systemctl restart ${SERVICE_NAME}'"
    echo "    停止服务:  ssh ${REMOTE_USER}@${REMOTE_HOST} 'sudo systemctl stop ${SERVICE_NAME}'"
    echo "    禁用自启:  ssh ${REMOTE_USER}@${REMOTE_HOST} 'sudo systemctl disable ${SERVICE_NAME}'"
    echo ""
    echo "  首次使用:"
    echo "    1. 访问 Web UI 加载模型（首次会自动下载）"
    echo "    2. 模型下载到 ${REMOTE_DIR}/models/"
    echo "    3. 可在「模型管理」页面设置启动自动加载"
    echo ""
}

# 清理临时文件
cleanup() {
    rm -f "/tmp/${SERVICE_NAME}-deploy.tar.gz" 2>/dev/null
}

# 主流程
main() {
    parse_args "$@"

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           FunASR 远程部署脚本                                 ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  目标: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SSH_PORT}"
    echo "  目录: ${REMOTE_DIR}"
    echo ""

    check_ssh
    check_remote_env
    package_project
    upload_to_remote
    install_remote_deps
    create_systemd_service
    start_remote_service
    cleanup
    print_summary
}

main "$@"
