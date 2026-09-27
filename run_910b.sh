#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH is not set"
    exit 1
fi
source "${ASCEND_HOME_PATH}/set_env.sh"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_910b}"
NPU_ARCH="${NPU_ARCH:-dav-2201}"
CAST_OPTION=()
if [[ "${ROPE_USE_CAST_ROUND:-0}" == "1" ]]; then
    CAST_OPTION+=("-DROPE_USE_CAST_ROUND=ON")
fi
if [[ "${ROPE_USE_POWER_INV_FREQ:-0}" == "1" ]]; then
    CAST_OPTION+=("-DROPE_USE_POWER_INV_FREQ=ON")
fi
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DNPU_ARCH="${NPU_ARCH}" "${CAST_OPTION[@]}"
cmake --build "${BUILD_DIR}" -j"${JOBS:-4}"

python3 "${SCRIPT_DIR}/scripts/run_device_matrix.py" \
    --exe "${BUILD_DIR}/rotary_pos_emb_eval" \
    --workdir "${SCRIPT_DIR}" \
    --kernel "${SCRIPT_DIR}/kernel.asc"
