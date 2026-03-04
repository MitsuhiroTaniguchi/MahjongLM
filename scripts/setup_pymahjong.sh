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

PYMAHJONG_REPO="https://github.com/MitsuhiroTaniguchi/pymahjong.git"
PYMAHJONG_REF="0b80bafa99c5b4b8bc1ade256c5a86b679238576"
PATCH_FILE="${ROOT_DIR}/scripts/patches/pymahjong-batchapi.patch"
PYMAHJONG_BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pymahjong-build.XXXXXX")"
trap 'rm -rf "${PYMAHJONG_BUILD_DIR}"' EXIT

# Base fast APIs were merged upstream:
# https://github.com/MitsuhiroTaniguchi/pymahjong/pull/4
# Additional batch APIs are temporarily applied from local patch until upstreamed.
# https://github.com/MitsuhiroTaniguchi/pymahjong/pull/5

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

git clone --depth 1 "${PYMAHJONG_REPO}" "${PYMAHJONG_BUILD_DIR}"
(
  cd "${PYMAHJONG_BUILD_DIR}"
  git checkout "${PYMAHJONG_REF}"
  git apply "${PATCH_FILE}"
)

python -m pip install --force-reinstall "${PYMAHJONG_BUILD_DIR}"

echo "Setup complete. Activate with: source .venv/bin/activate"
