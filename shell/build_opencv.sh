#!/bin/bash
# Build OpenCV 4.5.5 (core modules only) into 3rdparty/opencv.
# Matches the modules this project links: core, imgproc, imgcodecs.
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${SOURCE_DIR}/3rdparty/opencv"
OPENCV_VERSION="4.5.5"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

# Do not let an activated Conda environment leak newer TIFF/Lerc/libstdc++
# libraries into an Ubuntu 18-compatible OpenCV build.
clean_env() {
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_EXE \
        -u CMAKE_PREFIX_PATH -u LD_LIBRARY_PATH -u LIBRARY_PATH \
        -u CPATH -u CPLUS_INCLUDE_PATH -u C_INCLUDE_PATH \
        -u PKG_CONFIG_PATH \
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        "$@"
}

echo "=========================================="
echo "Building OpenCV ${OPENCV_VERSION} -> ${INSTALL_DIR}"
echo "=========================================="

echo "Installing build dependencies (requires sudo)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config wget \
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
clean_env /usr/bin/cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_LIST=core,imgproc,imgcodecs \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GTK=OFF \
    -DWITH_OPENGL=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_QT=OFF \
    -DWITH_CUDA=OFF

clean_env /usr/bin/cmake --build . -- -j"${BUILD_JOBS}"
rm -rf "${INSTALL_DIR}"
clean_env /usr/bin/cmake --build . --target install

cd "${SOURCE_DIR}"
rm -rf "opencv-${OPENCV_VERSION}" "opencv-${OPENCV_VERSION}.tar.gz"

if [[ ! -f "${INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake" ]] && \
   [[ ! -f "${INSTALL_DIR}/lib64/cmake/opencv4/OpenCVConfig.cmake" ]]; then
    echo "OpenCV install failed: OpenCVConfig.cmake not found under ${INSTALL_DIR}" >&2
    exit 1
fi

echo "OpenCV ${OPENCV_VERSION} installed to ${INSTALL_DIR}"
