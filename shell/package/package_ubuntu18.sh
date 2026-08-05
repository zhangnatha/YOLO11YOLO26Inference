#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build/ubuntu18-release"
DIST_ROOT="${PROJECT_ROOT}/dist"
PACKAGE_NAME="YOLO11YOLO26Inference-ubuntu18-x86_64"
SKIP_BUILD=0

usage() {
    echo "Usage: $0 [--skip-build]"
    echo ""
    echo "Build a GLIBC 2.27 portable package natively on Ubuntu 18.04."
    echo "--skip-build      Package an existing build/ubuntu18-release/YOLO_seg."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --native) ;; # Kept as a no-op for compatibility with older commands.
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

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

if [[ ${IS_UBUNTU_18} -ne 1 ]]; then
    echo "Portable packaging must run natively on Ubuntu 18.04 (GLIBC 2.27)." >&2
    echo "Detected: ${HOST_OS_ID:-unknown} ${HOST_VERSION_ID:-unknown}." >&2
    echo "A binary built on a newer Ubuntu release cannot be made backward compatible by copying libraries." >&2
    echo "Use an Ubuntu 18.04 build machine or virtual machine." >&2
    exit 5
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "An active Conda environment would contaminate the portable build: ${CONDA_PREFIX}" >&2
    echo "Run 'conda deactivate' until CONDA_PREFIX is empty, then package again." >&2
    exit 8
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
        CMAKE_ARGS=(
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=/
            -DYOLO_USE_BUNDLED_LINUX_DEPS=ON
            -DYOLO_USE_CUDA_ORT=OFF
        )
        cmake "${CMAKE_ARGS[@]}" "${PROJECT_ROOT}"
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

for OPTIONAL_RESOURCE_DIR in model images; do
    if [[ -d "${PROJECT_ROOT}/${OPTIONAL_RESOURCE_DIR}" ]]; then
        cp -a "${PROJECT_ROOT}/${OPTIONAL_RESOURCE_DIR}" \
              "${PACKAGE_DIR}/${OPTIONAL_RESOURCE_DIR}"
    else
        mkdir -p "${PACKAGE_DIR}/${OPTIONAL_RESOURCE_DIR}"
        echo "Optional directory not found; packaging an empty ${OPTIONAL_RESOURCE_DIR}/ directory."
    fi
done
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

UNRESOLVED="$(LD_LIBRARY_PATH="${PACKAGE_DIR}/lib" ldd "${EXECUTABLE}" 2>/dev/null | awk '/not found/ {print}')"
if [[ -n "${UNRESOLVED}" ]]; then
    echo "Package has unresolved shared-library dependencies:" >&2
    echo "${UNRESOLVED}" >&2
    exit 7
fi

NEWEST_GLIBC="$(find "${EXECUTABLE}" "${PACKAGE_DIR}/lib" -type f -print0 | xargs -0 strings 2>/dev/null | grep -Eo 'GLIBC_[0-9]+\.[0-9]+' | sort -V | tail -n 1 || true)"
if [[ -n "${NEWEST_GLIBC}" ]]; then
    HIGHEST="$(printf '%s\n%s\n' GLIBC_2.27 "${NEWEST_GLIBC}" | sort -V | tail -n 1)"
    if [[ "${HIGHEST}" != "GLIBC_2.27" ]]; then
        echo "ABI check failed: package requires ${NEWEST_GLIBC}; Ubuntu 18 supports GLIBC_2.27." >&2
        exit 4
    fi
fi

ARCHIVE="${DIST_ROOT}/${PACKAGE_NAME}.tar.gz"
rm -f "${ARCHIVE}"
tar -C "${DIST_ROOT}" -czf "${ARCHIVE}" "${PACKAGE_NAME}"
echo "Offline Ubuntu package: ${ARCHIVE}"
