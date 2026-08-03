#!/bin/bash
# Build yaml-cpp 0.8.0 into 3rdparty/yaml-cpp.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${SOURCE_DIR}/3rdparty/yaml-cpp"
YAML_CPP_VERSION="0.8.0"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

echo "=========================================="
echo "Building yaml-cpp ${YAML_CPP_VERSION} -> ${INSTALL_DIR}"
echo "=========================================="

echo "Installing build dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential cmake git

cd "${SOURCE_DIR}"
if [[ ! -d "yaml-cpp-${YAML_CPP_VERSION}" ]]; then
    echo "Downloading yaml-cpp ${YAML_CPP_VERSION}..."
    wget -c "https://github.com/jbeder/yaml-cpp/archive/refs/tags/${YAML_CPP_VERSION}.tar.gz" \
        -O "yaml-cpp-${YAML_CPP_VERSION}.tar.gz"
    tar -xzf "yaml-cpp-${YAML_CPP_VERSION}.tar.gz"
fi

cd "yaml-cpp-${YAML_CPP_VERSION}"
rm -rf build
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=ON \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF \
    -DYAML_CPP_BUILD_CONTRIB=ON

cmake --build . --parallel "${BUILD_JOBS}"
rm -rf "${INSTALL_DIR}"
cmake --install .

cd "${SOURCE_DIR}"
rm -rf "yaml-cpp-${YAML_CPP_VERSION}" "yaml-cpp-${YAML_CPP_VERSION}.tar.gz"

if [[ ! -f "${INSTALL_DIR}/lib/libyaml-cpp.so" ]]; then
    echo "yaml-cpp install failed: libyaml-cpp.so not found under ${INSTALL_DIR}" >&2
    exit 1
fi

echo "yaml-cpp ${YAML_CPP_VERSION} installed to ${INSTALL_DIR}"
