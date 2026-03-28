#!/usr/bin/env bash
# setup-pam.sh — Safely configure PAM for face authentication
# Usage: sudo bash scripts/setup-pam.sh
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

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
[[ $EUID -eq 0 ]] || die "Must be run as root. Use: sudo $0"

PAM_MODULE="/opt/ubuntu-face-login/pam_face.py"
DATA_DIR="/var/lib/ubuntu-face-login"
PAM_LINE="auth    sufficient    pam_python.so ${PAM_MODULE}"

echo -e "${BOLD}ubuntu-face-login — PAM setup${NC}"
echo ""

# Check installation
[[ -f "$PAM_MODULE" ]] || die "ubuntu-face-login not installed. Run install.sh first."
[[ -x "/usr/local/bin/ubuntu-face-login" ]] || die "Wrapper script not found. Run install.sh first."

# Check that pam_python.so exists
if ! find /lib/security /lib/*/security -name 'pam_python.so' -print -quit 2>/dev/null | grep -q .; then
    die "pam_python.so not found. Install libpam-python: apt install libpam-python"
fi
info "pam_python.so found"

# Check for enrolled users
ENROLLED=0
if [[ -d "$DATA_DIR" ]]; then
    ENROLLED=$(find "$DATA_DIR" -name '*.npy' 2>/dev/null | wc -l)
fi
if (( ENROLLED == 0 )); then
    die "No enrolled users found in ${DATA_DIR}.\nEnroll first: sudo ubuntu-face-login enroll \$(whoami)"
fi
info "${ENROLLED} embedding(s) found"

# ---------------------------------------------------------------------------
# Helper: add face auth to a PAM file
# ---------------------------------------------------------------------------
add_face_auth() {
    local pam_file="$1"
    local service_name="$2"

    if [[ ! -f "$pam_file" ]]; then
        warn "Skipping ${service_name}: ${pam_file} does not exist"
        return 1
    fi

    # Already configured?
    if grep -qF "pam_face.py" "$pam_file"; then
        warn "${service_name} already has face auth configured"
        return 0
    fi

    # Back up
    local backup="${pam_file}.ubuntu-face-login.bak"
    cp "$pam_file" "$backup"
    info "Backed up ${pam_file} → ${backup}"

    # Insert before @include common-auth
    if grep -q '@include common-auth' "$pam_file"; then
        sed -i "/@include common-auth/i ${PAM_LINE}" "$pam_file"
    else
        # Fallback: insert after the first auth line
        local first_auth_line
        first_auth_line=$(grep -n '^auth' "$pam_file" | head -1 | cut -d: -f1)
        if [[ -n "$first_auth_line" ]]; then
            sed -i "${first_auth_line}i ${PAM_LINE}" "$pam_file"
        else
            # Last resort: prepend
            sed -i "1i ${PAM_LINE}" "$pam_file"
        fi
    fi

    info "Added face auth to ${service_name}"
    return 0
}

# ---------------------------------------------------------------------------
# Helper: restore a PAM backup
# ---------------------------------------------------------------------------
restore_backup() {
    local pam_file="$1"
    local backup="${pam_file}.ubuntu-face-login.bak"

    if [[ -f "$backup" ]]; then
        cp "$backup" "$pam_file"
        info "Restored ${pam_file} from backup"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Configure sudo
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Step 1: Configure /etc/pam.d/sudo${NC}"
echo ""
echo "This adds face authentication before password for sudo."
echo "Face auth is 'sufficient' — if it fails, you still get a password prompt."
echo ""

add_face_auth "/etc/pam.d/sudo" "sudo"

echo ""
echo -e "${YELLOW}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}${BOLD}║  IMPORTANT: Open another terminal and test:             ║${NC}"
echo -e "${YELLOW}${BOLD}║                                                          ║${NC}"
echo -e "${YELLOW}${BOLD}║    sudo whoami                                           ║${NC}"
echo -e "${YELLOW}${BOLD}║                                                          ║${NC}"
echo -e "${YELLOW}${BOLD}║  It should either recognize your face OR ask for your    ║${NC}"
echo -e "${YELLOW}${BOLD}║  password. If it hangs or crashes, come back here.       ║${NC}"
echo -e "${YELLOW}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

read -rp "Did 'sudo whoami' work correctly? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    err "Restoring /etc/pam.d/sudo from backup"
    restore_backup "/etc/pam.d/sudo"
    echo ""
    echo "PAM configuration reverted. No changes were made."
    echo "Check the logs:  journalctl -t pam_face_login --no-pager -n 20"
    exit 1
fi

info "sudo face auth confirmed working"

# ---------------------------------------------------------------------------
# Step 2: Offer additional services
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Step 2: Additional PAM services (optional)${NC}"
echo ""

SERVICES_ADDED=("/etc/pam.d/sudo")

declare -A EXTRA_SERVICES=(
    ["/etc/pam.d/sudo-i"]="sudo -i (root shell)"
    ["/etc/pam.d/gdm-password"]="GDM login screen"
    ["/etc/pam.d/polkit-1"]="Polkit (GUI privilege prompts)"
)

for pam_file in "/etc/pam.d/sudo-i" "/etc/pam.d/gdm-password" "/etc/pam.d/polkit-1"; do
    desc="${EXTRA_SERVICES[$pam_file]}"

    if [[ ! -f "$pam_file" ]]; then
        continue
    fi

    read -rp "Also enable face auth for ${desc}? (y/n): " ANSWER
    if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
        if add_face_auth "$pam_file" "$desc"; then
            SERVICES_ADDED+=("$pam_file")
        fi
    fi
done

# ---------------------------------------------------------------------------
# Step 3: Check for howdy
# ---------------------------------------------------------------------------
echo ""

HOWDY_FOUND=0
for pam_file in /etc/pam.d/sudo /etc/pam.d/sudo-i /etc/pam.d/gdm-password /etc/pam.d/polkit-1; do
    if [[ -f "$pam_file" ]] && grep -q 'howdy' "$pam_file"; then
        HOWDY_FOUND=1
        break
    fi
done

if (( HOWDY_FOUND )); then
    echo -e "${YELLOW}Howdy PAM lines detected.${NC}"
    echo "Running both face auth systems can cause conflicts."
    echo ""
    read -rp "Comment out howdy lines in PAM files? (y/n): " HOWDY_ANSWER

    if [[ "$HOWDY_ANSWER" == "y" || "$HOWDY_ANSWER" == "Y" ]]; then
        for pam_file in /etc/pam.d/sudo /etc/pam.d/sudo-i /etc/pam.d/gdm-password /etc/pam.d/polkit-1; do
            if [[ -f "$pam_file" ]] && grep -q 'howdy' "$pam_file"; then
                # Back up if not already backed up
                local_backup="${pam_file}.ubuntu-face-login.bak"
                if [[ ! -f "$local_backup" ]]; then
                    cp "$pam_file" "$local_backup"
                fi
                sed -i '/howdy/s/^/#/' "$pam_file"
                info "Commented out howdy lines in ${pam_file}"
            fi
        done
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}PAM setup complete!${NC}"
echo ""
echo "Services configured:"
for svc in "${SERVICES_ADDED[@]}"; do
    echo -e "  ${GREEN}✓${NC} ${svc}"
done

echo ""
echo "Backup files (restore on problems):"
for svc in "${SERVICES_ADDED[@]}"; do
    bak="${svc}.ubuntu-face-login.bak"
    if [[ -f "$bak" ]]; then
        echo "  ${bak}"
    fi
done

echo ""
echo "To undo all changes:"
echo "  sudo bash $(dirname "$0")/../uninstall.sh"
echo ""
echo "Troubleshooting:"
echo "  journalctl -t pam_face_login --no-pager -n 20"
