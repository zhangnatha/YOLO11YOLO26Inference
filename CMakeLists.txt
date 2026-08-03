cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

project(Inference_UI VERSION 1.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YOLO_USE_BUNDLED_LINUX_DEPS
       "Prefer dependencies already stored in 3rdparty on Linux" ON)
option(YOLO_USE_CUDA_ORT
       "Link the bundled CUDA ONNX Runtime instead of the portable CPU runtime" OFF)

set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

# OpenCV: Linux releases use the repository bundle; Windows uses vcpkg or an
# explicitly supplied OpenCV_DIR.
if(UNIX AND YOLO_USE_BUNDLED_LINUX_DEPS AND
   EXISTS "${THIRDPARTY_DIR}/opencv/lib/cmake/opencv4/OpenCVConfig.cmake")
    set(OpenCV_DIR "${THIRDPARTY_DIR}/opencv/lib/cmake/opencv4" CACHE PATH "" FORCE)
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}")

if(UNIX AND YOLO_USE_BUNDLED_LINUX_DEPS)
    # The checked-in OpenCV was built on Ubuntu 18 and records libtiff.so.5.
    # Link it explicitly so modern build hosts cannot silently substitute the
    # incompatible libtiff.so.6 ABI.
    find_library(YOLO_TIFF5_LIBRARY NAMES libtiff.so.5)
    if(NOT YOLO_TIFF5_LIBRARY)
        message(FATAL_ERROR
            "Bundled OpenCV requires libtiff.so.5. Build with the Ubuntu 18 "
            "packaging container or install/provide the Ubuntu 18 TIFF runtime.")
    endif()
endif()

# ONNX Runtime 1.17.3. ONNXRUNTIME_DIR may point to an unpacked official SDK.
if(NOT ONNXRUNTIME_DIR)
    if(UNIX AND YOLO_USE_CUDA_ORT AND
       EXISTS "${THIRDPARTY_DIR}/onnxruntime-cuda12-1.17.3")
        set(ONNXRUNTIME_DIR "${THIRDPARTY_DIR}/onnxruntime-cuda12-1.17.3")
    elseif(UNIX AND EXISTS "${THIRDPARTY_DIR}/onnxruntime")
        set(ONNXRUNTIME_DIR "${THIRDPARTY_DIR}/onnxruntime")
    elseif(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}")
    endif()
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS "${ONNXRUNTIME_DIR}/include")
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
             HINTS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/lib64")
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime SDK not found. Set ONNXRUNTIME_DIR to the unpacked 1.17.3 SDK.")
endif()
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)

if(WIN32)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(yaml-cpp CONFIG REQUIRED)
    set(GLFW_LINK_TARGET glfw)
    set(YAML_CPP_LINK_TARGET yaml-cpp::yaml-cpp)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW REQUIRED IMPORTED_TARGET glfw3)
    set(GLFW_LINK_TARGET PkgConfig::GLFW)

    if(YOLO_USE_BUNDLED_LINUX_DEPS AND
       EXISTS "${THIRDPARTY_DIR}/yaml-cpp/lib/libyaml-cpp.so")
        add_library(YOLO_yaml_cpp SHARED IMPORTED)
        set_target_properties(YOLO_yaml_cpp PROPERTIES
            IMPORTED_LOCATION "${THIRDPARTY_DIR}/yaml-cpp/lib/libyaml-cpp.so"
            INTERFACE_INCLUDE_DIRECTORIES "${THIRDPARTY_DIR}/yaml-cpp/include")
        set(YAML_CPP_LINK_TARGET YOLO_yaml_cpp)
    else()
        find_package(yaml-cpp CONFIG REQUIRED)
        set(YAML_CPP_LINK_TARGET yaml-cpp::yaml-cpp)
    endif()
endif()

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set(GUI_SOURCES
    main.cpp
    SegmentationAnalysis.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/imgui_impl_opengl3.cpp)

add_executable(YOLO_seg ${GUI_SOURCES})
add_custom_command(TARGET YOLO_seg POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:YOLO_seg>/results"
            "$<TARGET_FILE_DIR:YOLO_seg>/original"
    COMMENT "Creating result and original output directories")
target_include_directories(YOLO_seg PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
    "${ONNXRUNTIME_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(YOLO_seg PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_LIBRARY}"
    ${GLFW_LINK_TARGET}
    OpenGL::GL
    ${YAML_CPP_LINK_TARGET}
    Threads::Threads)
if(YOLO_TIFF5_LIBRARY)
    target_link_libraries(YOLO_seg PRIVATE "${YOLO_TIFF5_LIBRARY}")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(YOLO_seg PRIVATE dl)
    # DT_RPATH is intentionally used instead of DT_RUNPATH: the portable
    # archive keeps every direct and transitive shared dependency in lib/ and
    # must start without an LD_LIBRARY_PATH wrapper.
    set_property(TARGET YOLO_seg APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,--disable-new-dtags")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        target_link_libraries(YOLO_seg PRIVATE stdc++fs)
    endif()
    set_target_properties(YOLO_seg PROPERTIES
        BUILD_RPATH "${THIRDPARTY_DIR}/opencv/lib;${ONNXRUNTIME_DIR}/lib;${THIRDPARTY_DIR}/yaml-cpp/lib"
        INSTALL_RPATH "$ORIGIN/lib")
endif()

if(MSVC)
    target_compile_options(YOLO_seg PRIVATE /W4 /utf-8 /permissive-)
    target_compile_definitions(YOLO_seg PRIVATE
        _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(YOLO_seg PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MINGW)
    # ONNX Runtime 1.17.3's official Windows headers use the MSVC spelling
    # `_stdcall`.  GCC 15's MinGW frontend only accepts `__stdcall`, so map
    # the spelling without modifying the downloaded SDK headers.
    target_compile_definitions(YOLO_seg PRIVATE
        NOMINMAX
        _WIN32_WINNT=0x0A00
        _stdcall=__stdcall)
    set_property(TARGET YOLO_seg APPEND_STRING PROPERTY
        LINK_FLAGS " -static-libgcc -static-libstdc++")
endif()

if(WIN32)
    find_file(ONNXRUNTIME_DLL NAMES onnxruntime.dll
              HINTS "${ONNXRUNTIME_DIR}/lib" "${ONNXRUNTIME_DIR}/bin"
              NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_DLL)
        message(FATAL_ERROR "onnxruntime.dll was not found in ONNXRUNTIME_DIR")
    endif()
    add_custom_command(TARGET YOLO_seg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_DLL}" "$<TARGET_FILE_DIR:YOLO_seg>"
        COMMENT "Copying ONNX Runtime DLL beside YOLO_seg.exe")
endif()

if(BUILD_TESTING)
    add_executable(roi_filter_test tests/RoiFilterTest.cpp)
    target_include_directories(roi_filter_test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(roi_filter_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME roi_filter_test COMMAND roi_filter_test)
endif()

install(TARGETS YOLO_seg RUNTIME DESTINATION .)
