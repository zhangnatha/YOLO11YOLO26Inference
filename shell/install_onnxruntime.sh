#!/bin/bash
# Download and install ONNX Runtime 1.17.3 (CPU + CUDA 12 GPU) into 3rdparty/.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
INSTALL_GPU_DIR="${PROJECT_ROOT}/3rdparty/onnxruntime-cuda12-1.17.3"
INSTALL_CPU_DIR="${PROJECT_ROOT}/3rdparty/onnxruntime"
ORT_VERSION="1.17.3"
CPU_TGZ="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
GPU_TGZ="onnxruntime-linux-x64-gpu-cuda12-${ORT_VERSION}.tgz"
CPU_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${CPU_TGZ}"
GPU_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${GPU_TGZ}"

cd "${PROJECT_ROOT}"
mkdir -p 3rdparty

download_if_needed() {
    local url="$1"
    local file="$2"
    if [[ ! -s "${file}" ]]; then
        echo "Downloading ${file} ..."
        wget -c "${url}" -O "${file}"
    else
        echo "Found local ${file}, skip download."
    fi
}

extract_into() {
    local archive="$1"
    local dest="$2"
    local tmp
    tmp="$(mktemp -d "${PROJECT_ROOT}/.ort-extract.XXXXXX")"
    tar -xzf "${archive}" -C "${tmp}"
    # Official archives contain a single top-level directory; accept either
    # onnxruntime-linux-x64-1.17.3 or onnxruntime-linux-x64-gpu-1.17.3.
    local top
    top="$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [[ -z "${top}" ]]; then
        echo "Failed to locate extracted ONNX Runtime directory in ${archive}" >&2
        rm -rf "${tmp}"
        exit 1
    fi
    rm -rf "${dest}"
    mkdir -p "${dest}"
    # Prefer rsync-like move of contents so dest path stays stable for CMake.
    mv "${top}"/* "${dest}/"
    rm -rf "${tmp}"
}

echo "=========================================="
echo "Installing ONNX Runtime ${ORT_VERSION}"
echo "  CPU  -> ${INSTALL_CPU_DIR}"
echo "  GPU  -> ${INSTALL_GPU_DIR}"
echo "=========================================="

download_if_needed "${CPU_URL}" "${CPU_TGZ}"
download_if_needed "${GPU_URL}" "${GPU_TGZ}"

echo "Extracting CPU package..."
extract_into "${CPU_TGZ}" "${INSTALL_CPU_DIR}"
echo "Extracting GPU (CUDA 12) package..."
extract_into "${GPU_TGZ}" "${INSTALL_GPU_DIR}"

rm -f "${CPU_TGZ}" "${GPU_TGZ}"

if [[ ! -f "${INSTALL_CPU_DIR}/include/onnxruntime_cxx_api.h" ]] || \
   [[ ! -f "${INSTALL_CPU_DIR}/lib/libonnxruntime.so" ]]; then
    echo "CPU ONNX Runtime install looks incomplete." >&2
    exit 1
fi
if [[ ! -f "${INSTALL_GPU_DIR}/include/onnxruntime_cxx_api.h" ]] || \
   [[ ! -f "${INSTALL_GPU_DIR}/lib/libonnxruntime.so" ]]; then
    echo "GPU ONNX Runtime install looks incomplete." >&2
    exit 1
fi

echo "ONNX Runtime ${ORT_VERSION} installed."
echo "  CPU:  ${INSTALL_CPU_DIR}"
echo "  CUDA: ${INSTALL_GPU_DIR}"
echo "Build with -DYOLO_USE_CUDA_ORT=ON to link the CUDA package."
