#!/usr/bin/env bash
# install_systemd_timer.sh — install/refresh the nightly replication timer on Razer.
# Idempotent. Run as the regular user (it sudos where needed).
#
# Usage:
#   bash scripts/systemd/install_systemd_timer.sh
#   bash scripts/systemd/install_systemd_timer.sh --uninstall

set -euo pipefail

UNIT_DIR=/etc/systemd/system
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "[timer] disabling + removing units"
    sudo systemctl disable --now jeffi-replicate.timer 2>/dev/null || true
    sudo rm -f "$UNIT_DIR/jeffi-replicate.timer" "$UNIT_DIR/jeffi-replicate.service"
    sudo systemctl daemon-reload
    echo "[timer] uninstalled"
    exit 0
fi

echo "[timer] installing units to $UNIT_DIR"
sudo install -m 0644 "$HERE/jeffi-replicate.service" "$UNIT_DIR/jeffi-replicate.service"
sudo install -m 0644 "$HERE/jeffi-replicate.timer"   "$UNIT_DIR/jeffi-replicate.timer"

echo "[timer] reloading systemd"
sudo systemctl daemon-reload

echo "[timer] enabling + starting timer"
sudo systemctl enable --now jeffi-replicate.timer

echo "[timer] status"
systemctl status jeffi-replicate.timer --no-pager --lines=0
echo
echo "[timer] next scheduled run:"
systemctl list-timers jeffi-replicate.timer --no-pager
echo
echo "Next steps:"
echo "  - Tail logs:        journalctl -u jeffi-replicate.service -f"
echo "  - Run on demand:    sudo systemctl start jeffi-replicate.service"
echo "  - Last run summary: systemctl status jeffi-replicate.service --no-pager"
