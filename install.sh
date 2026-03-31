#!/usr/bin/env bash
# install.sh — Install ubuntu-face-login
# Usage: sudo ./install.sh [--with-gui]
set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[✗]${NC} $*"; }
die()   { err "$@"; exit 1; }
step()  { echo -e "\n${BOLD}▸ $*${NC}"; }

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
WITH_GUI=0
for arg in "$@"; do
    case "$arg" in
        --with-gui) WITH_GUI=1 ;;
        --help|-h)
            echo "Usage: sudo $0 [--with-gui]"
            echo "  --with-gui   Also install GTK4 dependencies for the enrollment GUI"
            exit 0
            ;;
        *) die "Unknown option: $arg" ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
step "Pre-flight checks"

[[ $EUID -eq 0 ]] || die "Must be run as root. Use: sudo $0"

# Check Ubuntu version
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
        warn "Detected OS: ${ID:-unknown}. This installer targets Ubuntu — proceed at your own risk."
    fi
    case "${VERSION_ID:-}" in
        22.04|24.04) info "Ubuntu ${VERSION_ID} detected" ;;
        *)
            warn "Ubuntu ${VERSION_ID:-unknown} is untested. Tested versions: 22.04, 24.04"
            ;;
    esac
else
    warn "Cannot detect OS version (/etc/os-release missing)"
fi

# Check Python version
PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[[ -n "$PYTHON" ]] || die "Python 3.10+ not found. Install python3 first."

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if (( PY_MAJOR < 3 || PY_MINOR < 10 )); then
    die "Python 3.10+ required, found ${PY_VERSION}"
fi
info "Python ${PY_VERSION} (${PYTHON})"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INSTALL_DIR="/opt/ubuntu-face-login"
CONFIG_DIR="/etc/ubuntu-face-login"
DATA_DIR="/var/lib/ubuntu-face-login"
MODELS_DIR="${INSTALL_DIR}/models"
VENV_DIR="${INSTALL_DIR}/.venv"
WRAPPER="/usr/local/bin/ubuntu-face-login"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model URLs and checksums
FACENET_URL="https://github.com/yonm13/ubuntu-face-login/releases/latest/download/facenet512.onnx"
FACENET_SHA256="9a3ac09681674392ffd739e6c456811c3c05ef87d7b3e4ecfe7d5b50fb077a96"

YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# ---------------------------------------------------------------------------
# apt install
# ---------------------------------------------------------------------------
step "Installing system packages"

APT_PACKAGES=(libpam-python v4l-utils python3-venv)
if (( WITH_GUI )); then
    APT_PACKAGES+=(python3-gi gir1.2-gtk-4.0)
    info "GUI dependencies included"
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq "${APT_PACKAGES[@]}"
info "Installed: ${APT_PACKAGES[*]}"

# ---------------------------------------------------------------------------
# Install directory
# ---------------------------------------------------------------------------
step "Setting up ${INSTALL_DIR}"

mkdir -p "${INSTALL_DIR}"
cp -r "${SCRIPT_DIR}/src" "${INSTALL_DIR}/src"
info "Copied src/ to ${INSTALL_DIR}/src"

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
step "Creating virtual environment"

$PYTHON -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip -q
pip install numpy opencv-python-headless onnxruntime -q

# tomli needed for Python < 3.11
if (( PY_MINOR < 11 )); then
    pip install tomli -q
    info "Installed tomli (Python ${PY_VERSION} lacks tomllib)"
fi

info "Python packages installed in ${VENV_DIR}"
deactivate

# ---------------------------------------------------------------------------
# Download models
# ---------------------------------------------------------------------------
step "Downloading models"

mkdir -p "${MODELS_DIR}"

download_model() {
    local url="$1" dest="$2" expected_sha="$3" name="$4"

    if [[ -f "$dest" ]]; then
        local actual_sha
        actual_sha=$(sha256sum "$dest" | cut -d' ' -f1)
        if [[ "$actual_sha" == "$expected_sha" ]]; then
            info "${name} already present (sha256 OK)"
            return 0
        fi
        warn "${name} exists but sha256 mismatch — re-downloading"
    fi

    echo -e "  Downloading ${name}..."
    if ! curl -fSL --progress-bar -o "$dest" "$url"; then
        die "Failed to download ${name} from ${url}"
    fi

    local actual_sha
    actual_sha=$(sha256sum "$dest" | cut -d' ' -f1)
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        rm -f "$dest"
        die "SHA256 mismatch for ${name}!\n  Expected: ${expected_sha}\n  Got:      ${actual_sha}"
    fi
    info "${name} downloaded and verified"
}

download_model "$FACENET_URL"  "${MODELS_DIR}/facenet512.onnx"  "$FACENET_SHA256"  "facenet512.onnx"
download_model "$YUNET_URL"    "${MODELS_DIR}/yunet.onnx"       "$YUNET_SHA256"    "yunet.onnx"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
step "Configuration"

mkdir -p "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/config.toml" ]]; then
    cp "${SCRIPT_DIR}/config.example.toml" "${CONFIG_DIR}/config.toml"
    # Patch default paths to match installed locations
    sed -i 's|# dir = "/usr/share/ubuntu-face-login/models"|dir = "/opt/ubuntu-face-login/models"|' "${CONFIG_DIR}/config.toml"
    sed -i 's|# dir = "/usr/share/ubuntu-face-login/data"|dir = "/var/lib/ubuntu-face-login"|' "${CONFIG_DIR}/config.toml"
    info "Created ${CONFIG_DIR}/config.toml"
else
    warn "${CONFIG_DIR}/config.toml already exists — not overwriting"
fi

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
mkdir -p "${DATA_DIR}"
mkdir -p "${DATA_DIR}/faces"
chmod 755 "${DATA_DIR}"
chmod 755 "${DATA_DIR}/faces"

# Determine the real calling user.
# - sudo sets SUDO_USER
# - pkexec sets PKEXEC_UID (used when wizard calls pkexec bash install.sh)
REAL_USER=""
if [[ -n "${SUDO_USER:-}" ]]; then
    REAL_USER="${SUDO_USER}"
elif [[ -n "${PKEXEC_UID:-}" ]]; then
    REAL_USER=$(id -un "${PKEXEC_UID}" 2>/dev/null || true)
fi

if [[ -n "${REAL_USER}" ]]; then
    chown -R "${REAL_USER}:${REAL_USER}" "${DATA_DIR}"
    info "Data directory owned by ${REAL_USER}: ${DATA_DIR}"
else
    warn "Could not determine calling user — data directory owned by root."
    warn "Fix manually: sudo chown -R \$(whoami):\$(whoami) ${DATA_DIR}"
fi

# ---------------------------------------------------------------------------
# PAM module
# ---------------------------------------------------------------------------
step "Installing PAM module"
cp "${INSTALL_DIR}/src/facelogin/pam_module.py" "${INSTALL_DIR}/pam_face.py"
chmod 644 "${INSTALL_DIR}/pam_face.py"
info "PAM module installed at ${INSTALL_DIR}/pam_face.py"

# ---------------------------------------------------------------------------
# Wrapper script
# ---------------------------------------------------------------------------
step "Creating wrapper script"

cat > "${WRAPPER}" << 'EOF'
#!/usr/bin/env bash
# ubuntu-face-login — wrapper that activates the venv and dispatches subcommands
set -euo pipefail

INSTALL_DIR="/opt/ubuntu-face-login"
VENV_DIR="${INSTALL_DIR}/.venv"
export FACELOGIN_DIR="${INSTALL_DIR}"
export PYTHONPATH="${INSTALL_DIR}/src"

# Activate venv
source "${VENV_DIR}/bin/activate"

COMMAND="${1:-help}"
shift 2>/dev/null || true

case "$COMMAND" in
    auth)
        exec python -m facelogin.auth "$@"
        ;;
    enroll)
        exec python -m facelogin.enroll "$@"
        ;;
    detect-cameras)
        exec python -m facelogin.camera "$@"
        ;;
    config)
        exec python -m facelogin.config "$@"
        ;;
    version)
        exec python -c "from facelogin import __version__; print(f'ubuntu-face-login {__version__}')"
        ;;
    help|--help|-h)
        cat <<USAGE
Usage: ubuntu-face-login <command> [options]

Commands:
  auth              Authenticate via face recognition (used by PAM)
  enroll            Enroll a user's face
  detect-cameras    List available cameras and their capabilities
  config            Show active configuration
  version           Print version

Options are command-specific. Use: ubuntu-face-login <command> --help
USAGE
        ;;
    *)
        echo "Unknown command: $COMMAND" >&2
        echo "Run 'ubuntu-face-login help' for usage." >&2
        exit 1
        ;;
esac
EOF

chmod 755 "${WRAPPER}"
info "Wrapper installed at ${WRAPPER}"

# ---------------------------------------------------------------------------
# Camera detection
# ---------------------------------------------------------------------------
step "Detecting cameras"

if command -v v4l2-ctl &>/dev/null; then
    echo ""
    v4l2-ctl --list-devices 2>/dev/null || warn "No video devices found"
    echo ""
    info "Run 'ubuntu-face-login detect-cameras' for detailed analysis"
else
    warn "v4l-utils not available — skipping camera detection"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Enroll your face:"
echo "       sudo ubuntu-face-login enroll \$(whoami)"
echo ""
echo "  2. Test authentication:"
echo "       sudo ubuntu-face-login auth --user \$(whoami)"
echo ""
echo "  3. Enable for sudo (interactive, safe):"
echo "       sudo bash scripts/setup-pam.sh"
echo ""
echo "Installed to:  ${INSTALL_DIR}"
echo "Config:        ${CONFIG_DIR}/config.toml"
echo "Embeddings:    ${DATA_DIR}"
echo "Models:        ${MODELS_DIR}"
