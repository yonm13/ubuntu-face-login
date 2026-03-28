#!/usr/bin/env bash
# uninstall.sh — Remove ubuntu-face-login
# Usage: sudo ./uninstall.sh
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

[[ $EUID -eq 0 ]] || die "Must be run as root. Use: sudo $0"

echo -e "${BOLD}ubuntu-face-login — Uninstall${NC}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Remove face auth from PAM files
# ---------------------------------------------------------------------------
PAM_FILES=(/etc/pam.d/sudo /etc/pam.d/sudo-i /etc/pam.d/gdm-password /etc/pam.d/polkit-1)

for pam_file in "${PAM_FILES[@]}"; do
    if [[ ! -f "$pam_file" ]]; then
        continue
    fi

    if grep -qF "pam_face.py" "$pam_file"; then
        sed -i '/pam_face\.py/d' "$pam_file"
        info "Removed face auth from ${pam_file}"
    fi

    # Restore backup if present
    backup="${pam_file}.ubuntu-face-login.bak"
    if [[ -f "$backup" ]]; then
        # Uncomment any howdy lines we may have commented
        # (restoring backup achieves this, but only if user wants to revert fully)
        rm -f "$backup"
        info "Removed backup ${backup}"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Remove installation directory
# ---------------------------------------------------------------------------
if [[ -d /opt/ubuntu-face-login ]]; then
    rm -rf /opt/ubuntu-face-login
    info "Removed /opt/ubuntu-face-login"
else
    warn "/opt/ubuntu-face-login not found — skipping"
fi

# ---------------------------------------------------------------------------
# Step 3: Remove config directory
# ---------------------------------------------------------------------------
if [[ -d /etc/ubuntu-face-login ]]; then
    rm -rf /etc/ubuntu-face-login
    info "Removed /etc/ubuntu-face-login"
else
    warn "/etc/ubuntu-face-login not found — skipping"
fi

# ---------------------------------------------------------------------------
# Step 4: User data (ask first)
# ---------------------------------------------------------------------------
DATA_DIR="/var/lib/ubuntu-face-login"
if [[ -d "$DATA_DIR" ]]; then
    EMBEDDING_COUNT=$(find "$DATA_DIR" -name '*.npy' 2>/dev/null | wc -l)
    echo ""
    warn "${DATA_DIR} contains ${EMBEDDING_COUNT} embedding file(s)."
    read -rp "Remove user face data? This cannot be undone. (y/n): " ANSWER
    if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
        rm -rf "$DATA_DIR"
        info "Removed ${DATA_DIR}"
    else
        warn "Kept ${DATA_DIR}"
    fi
else
    warn "${DATA_DIR} not found — skipping"
fi

# ---------------------------------------------------------------------------
# Step 5: Remove wrapper script
# ---------------------------------------------------------------------------
WRAPPER="/usr/local/bin/ubuntu-face-login"
if [[ -f "$WRAPPER" ]]; then
    rm -f "$WRAPPER"
    info "Removed ${WRAPPER}"
else
    warn "${WRAPPER} not found — skipping"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}Uninstall complete.${NC}"
echo ""
echo "Note: libpam-python and other apt packages were not removed."
echo "To remove them: sudo apt remove libpam-python"
