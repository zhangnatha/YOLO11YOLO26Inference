#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/ubuntu18-release"
DIST_ROOT="${PROJECT_ROOT}/dist"
PACKAGE_NAME="YOLO11YOLO26Inference-ubuntu18-x86_64"
SKIP_BUILD=0
FORCE_NATIVE=0
FORCE_DOCKER=0
SKIP_ABI_CHECK=0

usage() {
    echo "Usage: $0 [--native|--docker] [--skip-build] [--skip-abi-check]"
    echo ""
    echo "Default: build natively on Ubuntu 18.04; use Docker on other hosts."
    echo "--native          Force building on the current host."
    echo "--docker          Force building in the Ubuntu 18.04 Docker container."
    echo "--skip-build      Package an existing build/ubuntu18-release/YOLO_seg."
    echo "--skip-abi-check  Allow GLIBC symbols newer than 2.27 (not Ubuntu 18 compatible)."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native) FORCE_NATIVE=1 ;;
        --docker) FORCE_DOCKER=1 ;;
        --skip-build) SKIP_BUILD=1; FORCE_NATIVE=1 ;;
        --skip-abi-check) SKIP_ABI_CHECK=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

if [[ ${FORCE_NATIVE} -eq 1 && ${FORCE_DOCKER} -eq 1 ]]; then
    echo "--native and --docker cannot be used together." >&2
    exit 2
fi

HOST_OS_ID=""
HOST_VERSION_ID=""
if [[ -r /etc/os-release ]]; then
    HOST_OS_ID="$(. /etc/os-release; printf '%s' "${ID:-}")"
    HOST_VERSION_ID="$(. /etc/os-release; printf '%s' "${VERSION_ID:-}")"
fi
IS_UBUNTU_18=0
if [[ "${HOST_OS_ID}" == "ubuntu" &&
      "${HOST_VERSION_ID}" == 18.04* ]]; then
    IS_UBUNTU_18=1
fi

USE_DOCKER=0
if [[ ${FORCE_DOCKER} -eq 1 ]]; then
    USE_DOCKER=1
elif [[ ${FORCE_NATIVE} -eq 1 ]]; then
    USE_DOCKER=0
elif [[ ${IS_UBUNTU_18} -eq 1 ]]; then
    echo "Ubuntu 18.04 detected; using native packaging (Docker is not required)."
    USE_DOCKER=0
else
    USE_DOCKER=1
fi

if [[ ${USE_DOCKER} -eq 1 ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required on ${HOST_OS_ID:-this host} ${HOST_VERSION_ID:-} to produce an Ubuntu 18-compatible package." >&2
        echo "Run on Ubuntu 18.04, install Docker, or explicitly use --native for diagnostics." >&2
        exit 5
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "Docker is installed but the current user cannot access the Docker daemon." >&2
        echo "Fix Docker permissions, use --docker with a working daemon, or run this script directly on Ubuntu 18.04." >&2
        exit 5
    fi
    echo "Building the offline package in an Ubuntu 18.04 container..."
    docker build \
        --file "${SCRIPT_DIR}/Dockerfile.ubuntu18" \
        --tag inference-ui-builder:ubuntu18 \
        "${SCRIPT_DIR}"
    docker run --rm \
        -v "${PROJECT_ROOT}:/src" \
        -w /src \
        -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        inference-ui-builder:ubuntu18 \
        bash -lc 'shell/package/package_ubuntu18.sh --native && chown -R "${HOST_UID}:${HOST_GID}" /src/build /src/dist'
    exit 0
fi

if [[ ${IS_UBUNTU_18} -ne 1 ]]; then
    echo "Warning: native packaging on ${HOST_OS_ID:-unknown} ${HOST_VERSION_ID:-unknown}; the GLIBC compatibility check must pass." >&2
fi

REQUIRED_COMMANDS=(cmake ldd awk strings tar patchelf)
if [[ ${SKIP_BUILD} -eq 0 ]]; then
    REQUIRED_COMMANDS+=(c++)
fi
for REQUIRED_COMMAND in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "${REQUIRED_COMMAND}" >/dev/null 2>&1; then
        echo "Required command not found: ${REQUIRED_COMMAND}" >&2
        echo "Install the Ubuntu 18.04 build dependencies listed in README.md." >&2
        exit 6
    fi
done

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    (
        cd "${BUILD_DIR}"
        cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/ \
            -DYOLO_USE_BUNDLED_LINUX_DEPS=ON \
            -DYOLO_USE_CUDA_ORT=OFF \
            "${PROJECT_ROOT}"
    )
    cmake --build "${BUILD_DIR}" --config Release -- -j"$(nproc)"
fi

BUILD_EXECUTABLE="${BUILD_DIR}/YOLO_seg"
if [[ ! -x "${BUILD_EXECUTABLE}" ]]; then
    echo "Executable not found: ${BUILD_EXECUTABLE}" >&2
    exit 3
fi

PACKAGE_DIR="${DIST_ROOT}/${PACKAGE_NAME}"
rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/lib" \
         "${PACKAGE_DIR}/results" "${PACKAGE_DIR}/original"

# Also fixes the install prefix when packaging an older --skip-build tree.
(
    cd "${BUILD_DIR}"
    cmake -DCMAKE_INSTALL_PREFIX=/ "${PROJECT_ROOT}"
)
DESTDIR="${PACKAGE_DIR}" cmake --build "${BUILD_DIR}" --target install -- -j"$(nproc)"
EXECUTABLE="${PACKAGE_DIR}/YOLO_seg"
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Installed executable not found: ${EXECUTABLE}" >&2
    exit 3
fi

cp -a "${PROJECT_ROOT}/model" "${PACKAGE_DIR}/model"
cp -a "${PROJECT_ROOT}/images" "${PACKAGE_DIR}/images"
cp "${PROJECT_ROOT}/README.md" "${PACKAGE_DIR}/README.md"

# Recursively collect non-glibc shared objects. Keeping the build inside Ubuntu
# 18 prevents newer system libraries from introducing incompatible symbols.
declare -A SEEN
QUEUE=("${EXECUTABLE}")
SEARCH_PATH="${PROJECT_ROOT}/3rdparty/opencv/lib:${PROJECT_ROOT}/3rdparty/onnxruntime/lib:${PROJECT_ROOT}/3rdparty/yaml-cpp/lib:${PACKAGE_DIR}/lib"
while [[ ${#QUEUE[@]} -gt 0 ]]; do
    CURRENT="${QUEUE[0]}"
    QUEUE=("${QUEUE[@]:1}")
    while IFS= read -r DEPENDENCY; do
        [[ -f "${DEPENDENCY}" ]] || continue
        BASE_NAME="$(basename "${DEPENDENCY}")"
        case "${BASE_NAME}" in
            libc.so.*|libpthread.so.*|libdl.so.*|libm.so.*|librt.so.*|ld-linux-*.so.*) continue ;;
            libGL.so.*|libGLX.so.*|libGLdispatch.so.*|libOpenGL.so.*) continue ;;
            libX*.so.*|libxcb*.so.*|libwayland-*.so.*|libbsd.so.*) continue ;;
        esac
        [[ -n "${SEEN[${BASE_NAME}]:-}" ]] && continue
        SEEN["${BASE_NAME}"]=1
        cp -L "${DEPENDENCY}" "${PACKAGE_DIR}/lib/${BASE_NAME}"
        QUEUE+=("${PACKAGE_DIR}/lib/${BASE_NAME}")
    done < <(LD_LIBRARY_PATH="${SEARCH_PATH}" ldd "${CURRENT}" 2>/dev/null | awk '/=> \// {print $3} /^[[:space:]]*\/.* \(/ {print $1}')
done

# Some checked-in libraries contain a RUNPATH from their original build host.
# Normalize every bundled object so its own transitive dependencies resolve
# from this archive's lib directory without LD_LIBRARY_PATH or a wrapper.
while IFS= read -r -d '' LIBRARY_FILE; do
    patchelf --set-rpath '$ORIGIN' "${LIBRARY_FILE}"
done < <(find "${PACKAGE_DIR}/lib" -type f -print0)

if [[ ${SKIP_ABI_CHECK} -eq 0 ]]; then
    NEWEST_GLIBC="$(find "${EXECUTABLE}" "${PACKAGE_DIR}/lib" -type f -print0 | xargs -0 strings 2>/dev/null | grep -Eo 'GLIBC_[0-9]+\.[0-9]+' | sort -V | tail -n 1 || true)"
    if [[ -n "${NEWEST_GLIBC}" ]]; then
        HIGHEST="$(printf '%s\n%s\n' GLIBC_2.27 "${NEWEST_GLIBC}" | sort -V | tail -n 1)"
        if [[ "${HIGHEST}" != "GLIBC_2.27" ]]; then
            echo "ABI check failed: package requires ${NEWEST_GLIBC}; Ubuntu 18 supports GLIBC_2.27." >&2
            echo "Run the default Docker build or use --skip-abi-check only for diagnostics." >&2
            exit 4
        fi
    fi
fi

ARCHIVE="${DIST_ROOT}/${PACKAGE_NAME}.tar.gz"
rm -f "${ARCHIVE}"
tar -C "${DIST_ROOT}" -czf "${ARCHIVE}" "${PACKAGE_NAME}"
echo "Offline Ubuntu package: ${ARCHIVE}"
