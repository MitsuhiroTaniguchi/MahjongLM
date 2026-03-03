#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# On macOS + Homebrew GCC, pymahjong's setup.py uses g++ by default.
# Override CMake compiler selection to clang/clang++ for stable builds.
if command -v xcrun >/dev/null 2>&1; then
  export SDKROOT="$(xcrun --show-sdk-path)"
fi
export CMAKE_ARGS="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ${CMAKE_ARGS:-}"

python -m pip install "git+https://github.com/MitsuhiroTaniguchi/pymahjong.git"

echo "Setup complete. Activate with: source .venv/bin/activate"
