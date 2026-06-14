#!/usr/bin/env bash
# bootstrap_ubuntu.sh — set up jeffistores-labs on the Razer/Ubuntu GPU box.
# Idempotent — safe to re-run. Tested on Ubuntu 22.04 with RTX 4080 Laptop (CUDA 12.x).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/AloysJehwin/jeffistores-labs/main/scripts/bootstrap_ubuntu.sh | bash
# OR after cloning:
#   bash scripts/bootstrap_ubuntu.sh

set -euo pipefail

REPO_URL="https://github.com/AloysJehwin/jeffistores-labs.git"
REPO_DIR="${REPO_DIR:-$HOME/Documents/GitHub-Personal/jeffistores-labs}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log()  { printf '\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

# 1. APT prerequisites ---------------------------------------------------------
log "Checking apt prerequisites (git, curl, build-essential)..."
need_apt=()
for pkg in git curl ca-certificates build-essential; do
    dpkg -s "$pkg" >/dev/null 2>&1 || need_apt+=("$pkg")
done
if [ ${#need_apt[@]} -gt 0 ]; then
    log "Installing: ${need_apt[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${need_apt[@]}"
fi
ok "apt prerequisites present"

# 2. NVIDIA driver / CUDA visibility -------------------------------------------
log "Checking NVIDIA driver..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found. Install the NVIDIA driver first (e.g. 'sudo ubuntu-drivers autoinstall'), reboot, and re-run."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ok "NVIDIA driver visible"

# 3. uv (fast Python package manager) ------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin — make sure it's on PATH for this shell
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q '\.local/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
fi
ok "uv $(uv --version)"

# 4. Clone or update the repo --------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    log "Cloning $REPO_URL into $REPO_DIR"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "Repo already exists at $REPO_DIR — pulling latest"
    git -C "$REPO_DIR" pull --ff-only || warn "pull failed (uncommitted changes?) — continuing"
fi
cd "$REPO_DIR"

# 5. Pin Python version via uv -------------------------------------------------
log "Pinning Python $PYTHON_VERSION via uv..."
uv python install "$PYTHON_VERSION"
echo "$PYTHON_VERSION" > .python-version
ok "Python $PYTHON_VERSION pinned"

# 6. Sync deps -----------------------------------------------------------------
# By default pyproject.toml asks for torch>=2.4. uv will pick a CUDA wheel
# automatically on Linux x86_64 (PyTorch's default index ships CUDA wheels).
log "Installing base dependencies (this may take a few minutes the first time)..."
uv sync
ok "base deps installed"

# 7. GPU sanity check ----------------------------------------------------------
log "Verifying PyTorch sees the GPU..."
uv run python - <<'PY'
import torch
cuda = torch.cuda.is_available()
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA build  : {torch.version.cuda}")
print(f"CUDA avail  : {cuda}")
if cuda:
    print(f"Device 0    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Tiny matmul to make sure kernels actually run
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"Matmul OK   : tensor shape {tuple(y.shape)} on cuda:0")
else:
    raise SystemExit("CUDA not available — check driver / reboot / nvidia-cuda-toolkit")
PY
ok "PyTorch + CUDA working"

# 8. Optional HF stack (uncomment when starting Week 4) ------------------------
# log "Installing Hugging Face stack (transformers, peft, trl)..."
# uv sync --extra hf --extra track

# 9. Done ----------------------------------------------------------------------
cat <<EOF

\033[1;32mAll set.\033[0m

Repo: $REPO_DIR
Next:
  cd $REPO_DIR
  make jupyter            # or: uv run jupyter lab
  # then open experiments/01_micrograd/01_value_and_autograd.ipynb

Tip — VSCode Remote-SSH from your Mac:
  - Cmd+Shift+P → "Remote-SSH: Connect to Host" → razer-gpu
  - Open folder: $REPO_DIR

EOF
