#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEMP_PARENT="${TMPDIR:-/tmp}"
TEMP_ROOT="$(mktemp -d "${TEMP_PARENT%/}/llmservingsim-astra-conformance.XXXXXX")"

cleanup() {
    case "${TEMP_ROOT}" in
        "${TEMP_PARENT%/}"/llmservingsim-astra-conformance.*)
            rm -rf -- "${TEMP_ROOT}"
            ;;
        *)
            echo "Refusing to remove unexpected temporary path: ${TEMP_ROOT}" >&2
            return 1
            ;;
    esac
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

"${PYTHON_BIN}" -m venv "${TEMP_ROOT}/venv"
PIP_CACHE_DIR="${TEMP_ROOT}/pip-cache" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    "${TEMP_ROOT}/venv/bin/python" -m pip install \
    --no-input \
    --no-deps \
    --only-binary=:all: \
    "protobuf==7.35.0" >&2

cd "${REPO_ROOT}"
"${TEMP_ROOT}/venv/bin/python" \
    "${REPO_ROOT}/scripts/astra_operation_conformance_runner.py"
