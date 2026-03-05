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

PYMAHJONG_REPO="https://github.com/MitsuhiroTaniguchi/pymahjong.git"
PYMAHJONG_REF="${PYMAHJONG_REF:-main}"
PYMAHJONG_PIP_SPEC="git+${PYMAHJONG_REPO}@${PYMAHJONG_REF}"

python -m pip install --force-reinstall "${PYMAHJONG_PIP_SPEC}"

echo "Setup complete. Activate with: source .venv/bin/activate"
