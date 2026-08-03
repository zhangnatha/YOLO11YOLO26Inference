#!/bin/bash
# Build OpenCV 4.5.5 (core modules only) into 3rdparty/opencv.
# Matches the modules this project links: core, imgproc, imgcodecs.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${SOURCE_DIR}/3rdparty/opencv"
OPENCV_VERSION="4.5.5"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

echo "=========================================="
echo "Building OpenCV ${OPENCV_VERSION} -> ${INSTALL_DIR}"
echo "=========================================="

echo "Installing build dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libjpeg-dev libpng-dev libtiff-dev zlib1g-dev \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev

cd "${SOURCE_DIR}"
if [[ ! -d "opencv-${OPENCV_VERSION}" ]]; then
    echo "Downloading OpenCV ${OPENCV_VERSION}..."
    wget -c "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz" \
        -O "opencv-${OPENCV_VERSION}.tar.gz"
    tar -xzf "opencv-${OPENCV_VERSION}.tar.gz"
fi

cd "opencv-${OPENCV_VERSION}"
rm -rf build
mkdir -p build && cd build

# Only the modules required by this project. Skipping contrib keeps the
# install smaller and avoids unused shared libraries.
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_GTK=ON \
    -DWITH_OPENGL=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_QT=OFF \
    -DWITH_CUDA=OFF

cmake --build . --parallel "${BUILD_JOBS}"
rm -rf "${INSTALL_DIR}"
cmake --install .

cd "${SOURCE_DIR}"
rm -rf "opencv-${OPENCV_VERSION}" "opencv-${OPENCV_VERSION}.tar.gz"

if [[ ! -f "${INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake" ]] && \
   [[ ! -f "${INSTALL_DIR}/lib64/cmake/opencv4/OpenCVConfig.cmake" ]]; then
    echo "OpenCV install failed: OpenCVConfig.cmake not found under ${INSTALL_DIR}" >&2
    exit 1
fi

echo "OpenCV ${OPENCV_VERSION} installed to ${INSTALL_DIR}"
