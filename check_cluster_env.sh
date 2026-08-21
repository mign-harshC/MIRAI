#!/usr/bin/env bash
# Modified for the MIRAI project, 2026.

set -euo pipefail

echo "MIRAI environment check"
python3 --version

python3 - <<'PY'
try:
    import torch
except ImportError:
    print("PyTorch: not installed")
else:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
PY

if [[ -n "${MIRAI_HOSTS:-}" ]]; then
  IFS=',' read -r -a hosts <<< "$MIRAI_HOSTS"
  for host in "${hosts[@]}"; do
    host=${host//[[:space:]]/}
    [[ -n "$host" ]] || continue
    echo "Checking SSH connectivity to $host"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
      "python3 --version; command -v nvidia-smi >/dev/null && nvidia-smi -L || true"
  done
else
  echo "MIRAI_HOSTS is unset; remote SSH checks skipped."
fi

echo "This script never creates or copies SSH keys. Configure remote access separately."
