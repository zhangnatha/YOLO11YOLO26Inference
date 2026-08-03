#!/bin/bash
# Install CUDA toolkit + matching cuDNN for ONNX Runtime GPU builds.
# Default is CUDA 12.1 to match the bundled onnxruntime-cuda12-1.17.3 package.
# Override with: CUDA_CHOICE=11.8 ./shell/install_cuda.sh
set -euo pipefail

CUDA_CHOICE="${CUDA_CHOICE:-12.1}"
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${CUDA_CHOICE}" == "12.1" ]]; then
    CUDA_VERSION="12.1"
    CUDA_RUN_FILE="cuda_12.1.0_530.30.02_linux.run"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run"
    CUDNN_TAR_FILE="cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"
    CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz"
    CUDNN_DIR="cudnn-linux-x86_64-8.9.7.29_cuda12-archive"
elif [[ "${CUDA_CHOICE}" == "11.8" ]]; then
    CUDA_VERSION="11.8"
    CUDA_RUN_FILE="cuda_11.8.0_520.61.05_linux.run"
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run"
    CUDNN_TAR_FILE="cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz"
    CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz"
    CUDNN_DIR="cudnn-linux-x86_64-8.9.7.29_cuda11-archive"
else
    echo "Unsupported CUDA_CHOICE='${CUDA_CHOICE}'. Use 12.1 or 11.8." >&2
    exit 1
fi

CUDA_INSTALL_PATH="/usr/local/cuda-${CUDA_VERSION}"

echo "=========================================="
echo "Installing CUDA ${CUDA_VERSION} + cuDNN"
echo "Working directory: ${PROJECT_ROOT}"
echo "=========================================="
echo "Note: NVIDIA driver must already be installed."
echo "The CUDA .run installer is interactive; follow on-screen prompts."
echo "Recommended: install toolkit only (skip driver) if a driver exists."

if [[ ! -f "${CUDA_RUN_FILE}" ]]; then
    echo "Downloading CUDA installer..."
    wget -c "${CUDA_URL}" -O "${CUDA_RUN_FILE}"
else
    echo "Found local ${CUDA_RUN_FILE}, skip download."
fi

if [[ ! -f "${CUDNN_TAR_FILE}" ]]; then
    echo "Downloading cuDNN archive..."
    wget -c "${CUDNN_URL}" -O "${CUDNN_TAR_FILE}"
else
    echo "Found local ${CUDNN_TAR_FILE}, skip download."
fi

echo "Installing CUDA ${CUDA_VERSION}..."
chmod +x "${CUDA_RUN_FILE}"
sudo "./${CUDA_RUN_FILE}"

echo "Installing cuDNN into ${CUDA_INSTALL_PATH}..."
tar -xJf "${CUDNN_TAR_FILE}"
sudo cp "${CUDNN_DIR}"/include/cudnn*.h "${CUDA_INSTALL_PATH}/include/"
sudo cp "${CUDNN_DIR}"/lib/libcudnn* "${CUDA_INSTALL_PATH}/lib64/"
sudo chmod a+r "${CUDA_INSTALL_PATH}"/include/cudnn*.h
sudo chmod a+r "${CUDA_INSTALL_PATH}"/lib64/libcudnn*

rm -rf "${CUDNN_DIR}"
rm -f "${CUDNN_TAR_FILE}" "${CUDA_RUN_FILE}"

if ! grep -q "CUDA-${CUDA_VERSION} Config" "${HOME}/.bashrc"; then
    cat << EOF >> "${HOME}/.bashrc"

# CUDA-${CUDA_VERSION} Config
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
EOF
fi

echo "=========================================="
echo "CUDA ${CUDA_VERSION} and cuDNN installed."
echo "Run: source ~/.bashrc"
echo "Then build with: -DYOLO_USE_CUDA_ORT=ON"
echo "=========================================="
