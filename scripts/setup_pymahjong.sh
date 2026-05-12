#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "python3 or python is required" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

PYMAHJONG_REPO="https://github.com/MitsuhiroTaniguchi/pymahjong.git"
PYMAHJONG_REF="${PYMAHJONG_REF:-main}"
PYMAHJONG_PIP_SPEC="git+${PYMAHJONG_REPO}@${PYMAHJONG_REF}"

python -m pip install --force-reinstall "${PYMAHJONG_PIP_SPEC}"

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Then run scripts with either: python ... or python3 ..."
