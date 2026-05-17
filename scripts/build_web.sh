#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMSDK_DIR="${EMSDK_DIR:-/Users/janosmeny/Projects/emsdk}"

source "${EMSDK_DIR}/emsdk_env.sh" >/dev/null

mkdir -p "${REPO_ROOT}/web/dist"

em++ \
  --use-port=emdawnwebgpu \
  -O3 \
  -std=c++20 \
  -sALLOW_MEMORY_GROWTH=1 \
  -sASSERTIONS=1 \
  -sENVIRONMENT=web \
  -sEXIT_RUNTIME=0 \
  "-sEXPORTED_FUNCTIONS=['_main','_curve_set_turn','_curve_reset_game']" \
  "-sEXPORTED_RUNTIME_METHODS=['ccall']" \
  "${REPO_ROOT}/cpp/curve_webgpu_web.cpp" \
  -o "${REPO_ROOT}/web/dist/curve_web.js"

cp "${REPO_ROOT}/web/index.html" "${REPO_ROOT}/web/dist/index.html"
touch "${REPO_ROOT}/web/dist/.nojekyll"
