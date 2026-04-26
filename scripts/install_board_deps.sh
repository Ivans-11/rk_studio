#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

rknn_runtime="${repo_root}/third_party/rknn/runtime/aarch64/librknnrt.so"
zenoh_deb_dir="${repo_root}/third_party/zenohc/debian/arm64"
zenoh_runtime_deb="${zenoh_deb_dir}/libzenohc_1.9.0_arm64.deb"
zenoh_dev_deb="${zenoh_deb_dir}/libzenohc-dev_1.9.0_arm64.deb"

if [[ ${EUID} -eq 0 ]]; then
  sudo_cmd=()
else
  sudo_cmd=(sudo)
fi

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
}

require_file "${rknn_runtime}"
require_file "${zenoh_runtime_deb}"
require_file "${zenoh_dev_deb}"

echo "[deps] installing RKNN runtime: ${rknn_runtime}"
"${sudo_cmd[@]}" install -m 0644 "${rknn_runtime}" /usr/lib/librknnrt.so

echo "[deps] installing zenoh-c packages"
"${sudo_cmd[@]}" dpkg -i "${zenoh_runtime_deb}" "${zenoh_dev_deb}"

echo "[deps] refreshing dynamic linker cache"
"${sudo_cmd[@]}" ldconfig

echo "[deps] verifying RKNN runtime"
nm -D /usr/lib/librknnrt.so | grep -q rknn_mem_sync

echo "[deps] verifying zenoh-c CMake package"
test -f /usr/lib/cmake/zenohc/zenohcConfig.cmake

echo "[deps] board runtime dependencies are ready"
