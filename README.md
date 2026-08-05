#  🚀 YOLO11YOLO26Inference

[![Ubuntu](https://img.shields.io/badge/Ubuntu-18.04%20%7C%2020.04%20%7C%2022.04%20%7C%2024.04-orange.svg)](https://ubuntu.com/) [![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011%20x64-blue.svg)](https://microsoft.com/) [![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17) [![Dependency](https://img.shields.io/badge/Dependency-ONNX__Runtime__1.17.3-orange.svg)](https://onnxruntime.ai/) [![Dependency](https://img.shields.io/badge/Dependency-OpenCV__4%2B-green.svg)](https://opencv.org/) [![Dependency](https://img.shields.io/badge/Dependency-ImGui%20%7C%20GLFW-brightgreen.svg)](https://github.com/ocornut/imgui)

基于 ONNX Runtime 1.17.3 的跨平台 C++17 YOLO11/YOLO26 实例分割推理与人工复核客户端。支持 Ubuntu 18.04+ 及 Windows 10+ (x64) 系统，提供与 X-AnyLabeling 标注数据的真值与推理对比分析，以及一键打包部署功能。  

## 💡 功能与绘制约定

- **文件夹推理**：一键批量推理 PNG、JPG/JPEG、BMP、TIF/TIFF 格式图片。  
- **多边形 ROI**：在左侧原图上鼠标逐点绘制并持久化 ROI；ROI 外目标的 mask、轮廓、box/编号和预测 JSON 记录会同时过滤。
- **图形绘制**：GT（真值）轮廓固定显示为绿色；PRED（预测）根据类别分配固定颜色（不与 GT 绿色冲突）。  
- **信息标注**：目标仅显示同色数字编号，左上角显示详细信息（编号、类别、score、PRED/GT 颜色）。  
- **自动显示/保存**：
  - 勾选 `Show top-left text` 可控制左上角文字显示，设置自动保存。  
  - 若原图旁存在同名 X-AnyLabeling JSON，自动加载并绘制真值轮廓。  
  - 推理图追加 `_r` 后缀保存（如 `sample_r.png`），并生成兼容 X-AnyLabeling 的 JSON 文件（保留类别与 score）。  
- **人工复核界面**：
  - 左右分屏对比原图与结果图，支持按键（左右方向键）或按钮快速切换。  
  - 点击 `Save Original Images` 或按 `S` 键一键提取保存当前原图与结果图。  
- **IoU 分析工具**：点击 `Open Analysis UI` 可对比预测 JSON 与 GT JSON，统计逐类别的漏检、误检、Precision、Recall、F1 及可视化效果图。  

## 📁 自动创建的输出目录

构建成功后，在可执行文件同级自动生成以下目录（配置文件、日志及 imgui 设置均写在运行目录下，不污染源码）：

```bash
results/     # 存放批量推理结果图及预测 JSON
original/    # 存放人工选中的原图及对应 _r 结果图
```

- **单配置构建**：`build/results/` 和 `build/original/`

- **Windows MinGW**：`build/windows-mingw64/results/` 和 `build/windows-mingw64/original/`

- **离线包运行**：位于解压后的离线包根目录  

## 📦 第三方库安装教程

本项目依赖如下库（版本建议）：

| 依赖 | 版本 | 用途 | Linux 位置 / 获取方式 |
|------|------|------|------------------------|
| ONNX Runtime | **1.17.3** | 模型推理 | `3rdparty/onnxruntime`（CPU）或 `3rdparty/onnxruntime-cuda12-1.17.3`（GPU） |
| OpenCV | **4.5.5+**（模块：core / imgproc / imgcodecs） | 图像读写与处理 | `3rdparty/opencv` 或系统包 |
| yaml-cpp | **0.8.x** | 配置读写 | `3rdparty/yaml-cpp` 或系统包 |
| GLFW3 | 系统包 | ImGui 窗口 | `libglfw3-dev` / vcpkg |
| OpenGL | 系统包 | ImGui 渲染 | `libgl1-mesa-dev` |
| ImGui | 仓库内 `imgui/` | GUI | 已随源码提供，无需单独安装 |
| CUDA + cuDNN（可选） | CUDA **12.1**（推荐）或 11.8 | GPU 推理 | 见下方 GPU 小节 |

> **说明**：`3rdparty/` 默认被 `.gitignore` 忽略，需在本机安装或通过脚本生成。Ubuntu 18.04 离线打包依赖仓库内预置的 OpenCV / yaml-cpp / ONNX Runtime CPU 包。

### 一键脚本总览（Linux）

在项目根目录执行：

```bash
# 1) ONNX Runtime 1.17.3（同时安装 CPU 与 CUDA12 GPU 包到 3rdparty/）
./shell/install_onnxruntime.sh

# 2) OpenCV 4.5.5（仅编译 core/imgproc/imgcodecs，安装到 3rdparty/opencv）
./shell/build_opencv.sh

# 3) yaml-cpp 0.8.0（安装到 3rdparty/yaml-cpp）
./shell/build_yaml_cpp.sh

# 4) 可选：CUDA Toolkit + cuDNN（默认 CUDA 12.1，供 GPU ORT 使用）
./shell/install_cuda.sh
# 或指定 11.8：
# CUDA_CHOICE=11.8 ./shell/install_cuda.sh
```

脚本会把产物放到 `3rdparty/` 下，CMake 在 `YOLO_USE_BUNDLED_LINUX_DEPS=ON` 时自动优先使用这些路径。

### 1. ONNX Runtime 1.17.3

**方式 A：使用仓库脚本（推荐）**

```bash
./shell/install_onnxruntime.sh
```

安装结果：

- CPU：`3rdparty/onnxruntime/`
- GPU（CUDA 12）：`3rdparty/onnxruntime-cuda12-1.17.3/`

**方式 B：手动下载官方包**

```bash
# CPU
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
tar -xzf onnxruntime-linux-x64-1.17.3.tgz
mv onnxruntime-linux-x64-1.17.3 3rdparty/onnxruntime

# GPU (CUDA 12)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-cuda12-1.17.3.tgz
tar -xzf onnxruntime-linux-x64-gpu-cuda12-1.17.3.tgz
# 解压目录名通常为 onnxruntime-linux-x64-gpu-1.17.3
mv onnxruntime-linux-x64-gpu-1.17.3 3rdparty/onnxruntime-cuda12-1.17.3
```

Windows 可使用官方 `onnxruntime-win-x64-1.17.3.zip`；`shell/package/package_windows.ps1` 会在首次构建时自动下载并缓存到 `3rdparty/windows/`。

### 2. OpenCV 4.x

**方式 A：仓库脚本（Ubuntu 18 兼容、与打包一致）**

```bash
./shell/build_opencv.sh
# 产物：3rdparty/opencv ，含 OpenCVConfig.cmake
```

**方式 B：系统包（Ubuntu 20.04+ 源码构建时可用）**

```bash
sudo apt-get install -y libopencv-dev
# 构建时关闭捆绑依赖：
# cmake ... -DYOLO_USE_BUNDLED_LINUX_DEPS=OFF
```

**方式 C：Windows**

由 `vcpkg.json` 声明依赖，打包脚本通过 vcpkg 安装 `opencv4`（静态）。

> 注意：仓库预置的 OpenCV 在 Ubuntu 18 上构建，运行时需要 `libtiff.so.5`（包名 `libtiff5`）。在较新系统上若使用捆绑 OpenCV，请自行提供该 ABI，或改用系统 OpenCV（`YOLO_USE_BUNDLED_LINUX_DEPS=OFF`）。

### 3. yaml-cpp

```bash
# 方式 A：脚本安装到 3rdparty/yaml-cpp
./shell/build_yaml_cpp.sh

# 方式 B：系统包（Ubuntu 20.04+）
sudo apt-get install -y libyaml-cpp-dev
```

Windows 同样由 vcpkg 提供 `yaml-cpp`。

### 4. GLFW3 / OpenGL / 基础工具链

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake pkg-config \
    libgl1-mesa-dev libglfw3-dev \
    libtiff5 patchelf wget
```

- **ImGui**：已 vendoring 在 `imgui/`，无需额外安装。  
- **Windows**：GLFW / OpenCV / yaml-cpp 均由 vcpkg 根据 `vcpkg.json` 安装；另需 MinGW `x86_64-w64-mingw32` **GCC 15.2.0**。

### 5. CUDA + cuDNN（仅 GPU 推理需要）

GPU 推理需要：

1. 已安装的 NVIDIA 驱动  
2. CUDA Toolkit（推荐 **12.1**，与 `onnxruntime-cuda12-1.17.3` 匹配）  
3. 对应版本 cuDNN  
4. 使用 GPU 版 ONNX Runtime 链接  

```bash
# 安装 CUDA 12.1 + cuDNN（交互式 runfile，需 sudo）
./shell/install_cuda.sh
source ~/.bashrc

# 确认已安装 GPU ORT
ls 3rdparty/onnxruntime-cuda12-1.17.3/lib/libonnxruntime.so

# 构建时启用 CUDA ORT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DYOLO_USE_BUNDLED_LINUX_DEPS=ON \
  -DYOLO_USE_CUDA_ORT=ON
cmake --build build --parallel
```

无 GPU 或不需要加速时，保持 `-DYOLO_USE_CUDA_ORT=OFF`（默认），仅使用 CPU ORT 即可。

### 6. 依赖就绪后的目录示意

```text
3rdparty/
├── onnxruntime/                      # CPU ORT 1.17.3
│   ├── include/
│   └── lib/libonnxruntime.so*
├── onnxruntime-cuda12-1.17.3/        # 可选 GPU ORT
│   ├── include/
│   └── lib/
├── opencv/                           # OpenCV 4.5.5
│   ├── include/opencv4/
│   └── lib/cmake/opencv4/OpenCVConfig.cmake
└── yaml-cpp/                         # yaml-cpp 0.8
    ├── include/
    └── lib/libyaml-cpp.so*
```

### 7. CMake 相关开关

| 选项 | 默认 | 含义 |
|------|------|------|
| `YOLO_USE_BUNDLED_LINUX_DEPS` | `ON` | Linux 优先使用 `3rdparty/` 内 OpenCV / yaml-cpp |
| `YOLO_USE_CUDA_ORT` | `OFF` | `ON` 时链接 `3rdparty/onnxruntime-cuda12-1.17.3` |
| `ONNXRUNTIME_DIR` | 自动探测 | 可手动指定解压后的 ORT 根目录 |
| `OpenCV_DIR` | 自动探测 | 可手动指定 OpenCVConfig.cmake 所在目录 |

## 🛠️ Ubuntu 源码构建与运行

### 1. Ubuntu 18.04（使用仓库内预置 / 脚本安装的依赖）

```bash
conda deactivate  # 如已启用 Conda，请先退出（可能需要执行多次）
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libgl1-mesa-dev libglfw3-dev libtiff5 patchelf

# 若曾在 Conda 环境中构建 OpenCV，退出 Conda 后必须重新构建
./shell/build_opencv.sh

# 其余 3rdparty 依赖按需执行上一节安装脚本
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DYOLO_USE_BUNDLED_LINUX_DEPS=ON -DYOLO_USE_CUDA_ORT=OFF
cmake --build build --parallel
```

### 2. Ubuntu 20.04/22.04/24.04（使用系统依赖）

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libgl1-mesa-dev libglfw3-dev libopencv-dev libyaml-cpp-dev

# 仍需提供 ONNX Runtime 1.17.3（脚本或手动放入 3rdparty/onnxruntime）
./shell/install_onnxruntime.sh   # 可只保留 CPU 目录

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DYOLO_USE_BUNDLED_LINUX_DEPS=OFF \
  -DONNXRUNTIME_DIR="$PWD/3rdparty/onnxruntime"
cmake --build build --parallel
```

### 3. 运行程序

- **GUI 模式**：

  ```bash
  ./build/YOLO_seg
  ```
  
- **CLI 命令行模式**：

  ```bash
  ./build/YOLO_seg \
    --batch-images images \
    --model model/suidong_20260727YOLO26m.dmmodel/segmentation.onnx \
    --output build/results \
    --conf 0.60 --iou 0.45 --cpu
  ```

## 📦 Ubuntu 18/20/22/24 通用离线包

- **本机编译（Ubuntu 18.04 直接运行）**：

  ```bash
  ./shell/package/package_ubuntu18.sh --native
  ```
  
- 打包脚本为纯本机流程，请在 Ubuntu 18.04 x86_64 实机或虚拟机中执行；
  在更新系统上编译产生的 GLIBC 引用无法向下兼容 Ubuntu 18.04，因此脚本会拒绝
  在 Ubuntu 20/22/24 上生成通用包。脚本会递归收集非系统动态库，检查不存在
  `not found` 项，并拒绝高于 GLIBC 2.27 的产物。

  成功后在 `dist/` 目录生成
  `YOLO11YOLO26Inference-ubuntu18-x86_64.tar.gz`。该包以 Ubuntu 18.04
  为最低 ABI 基线，可在 Ubuntu 18.04、20.04、22.04、24.04 x86_64
  上直接解压运行 `./YOLO_seg`；目标机仍需图形驱动/X11（桌面版 Ubuntu
  默认具备）。

## 💻 Windows 10/11 源码构建与离线包

### 环境准备

- 64 位 Windows 10/11，需要安装 CMake、Git、PowerShell 5.1+。  
- **GCC 工具链**：必须精确使用 `mingw64_x86_64-15.2.0`（Target: `x86_64-w64-mingw32`）。  
- 依赖由 `vcpkg.json` + 打包脚本自动处理（glfw3 / opencv4 / yaml-cpp），ONNX Runtime 1.17.3 首次构建时自动下载。

### 编译与打包脚本

配置环境变量并运行打包脚本（首次运行会自动下载并缓存 vcpkg 和 ONNX Runtime）：  

```PowerShell
$env:MINGW_ROOT = "D:\mingw64_x86_64-15.2.0"

# 仅构建源码
.\shell\package\package_windows.ps1 -BuildOnly

# 构建并打包离线包（输出至 dist\YOLO11YOLO26Inference-windows-x64.zip）
.\shell\package\package_windows.ps1

# 纯离线编译（禁止重新下载依赖）
.\shell\package\package_windows.ps1 -Offline
```

## 🖥️ GUI 使用说明

1. 选择 ONNX 模型及图片目录。  
2. 设置阈值（Confidence、IoU）和运行设备（CPU/GPU）。  
3. 如需限制检测区域，点击 **Draw / Redraw ROI**，在左侧预览图中左键逐点绘制、右键撤销上一点；至少三个点后点击 **Finish ROI**。
4. `Enable ROI` 控制是否应用已完成的 ROI，`Clear ROI` 恢复整图推理。ROI 使用归一化坐标保存，可应用到文件夹内不同分辨率图片。
5. 点击 **Load Model** 加载模型，再点击 **Run Folder Inference** 开始推理。
6. 使用键盘 **← / →** 快捷键逐张查看对比图。
7. 遇到不理想样本，点击 **Save Original Images** 或直接按 **S** 键保存至 `original/` 目录供后续迭代。

只要检测 mask 与 ROI 有至少一个像素交集就保留该目标，并将 mask 裁剪到 ROI 内、按裁剪后的 mask 重新计算 box；完全没有交集的目标同时从轮廓、box/编号和预测 JSON 中移除。没有 mask 时使用 box 与 ROI 的像素交集更新 box。

## 📊 预测与标注 IoU 分析

对推理结果与真值（Ground Truth）进行量化评估：  

### 图形化 UI 分析

点击主界面 **Open Analysis UI**：

1. 设置 `Prediction directory`（含 `_r.json` 目录）与 `Ground-truth directory`（含原始图片及同名 JSON 目录）。  
2. 设定匹配阈值 `Match IoU`，点击 **Run Analysis**。  
3. 可视化效果说明：
   - 🟩 **绿色**：已匹配的 GT  
   - 🟥 **红色**：漏检 GT (Missed)  
   - 🟦 **蓝色**：已匹配的预测  
   - 🟪 **品红**：误检预测 (False Pos)  

### CLI 命令行分析

```bash
./YOLO_seg --analyze-predictions results --ground-truth images --analysis-output analysis --match-iou 0.50 --hide-overlay-text
```

输出产物包含统计表 `analysis_report.csv` / `json` 及对比图。  

## 🧹 清理构建产物

- **Linux**：

  ```bash
  rm -rf build dist runtime
  ```
  
- **Windows PowerShell**：

  ```powershell
  Remove-Item -Recurse -Force build, dist, runtime -ErrorAction SilentlyContinue
  ```

如需重新生成 Linux 第三方库：

```bash
rm -rf 3rdparty/opencv 3rdparty/yaml-cpp 3rdparty/onnxruntime 3rdparty/onnxruntime-cuda12-1.17.3
./shell/install_onnxruntime.sh
./shell/build_opencv.sh
./shell/build_yaml_cpp.sh
```

## 📂 源码目录结构

```text
YOLO11YOLO26Inference/
├── 3rdparty/                 # Linux 第三方库（脚本安装，默认不入 git）
├── images/                   # 示例图片与可选同名真值 JSON
├── imgui/                    # ImGui GUI 源码（已内置）
├── include/                  # YOLO11/YOLO26 推理核心与工具头文件
├── model/                    # ONNX 模型与 classes.names
├── shell/
│   ├── install_onnxruntime.sh
│   ├── build_opencv.sh
│   ├── build_yaml_cpp.sh
│   ├── install_cuda.sh
│   └── package/              # Ubuntu / Windows 一键打包
├── tests/                    # 单元测试（ROI 过滤等）
├── CMakeLists.txt
├── main.cpp                  # GUI / 批量推理 / 复核入口
├── SegmentationAnalysis.cpp  # 预测与 GT 的 IoU 分析
├── vcpkg.json                # Windows vcpkg 依赖清单
└── README.md
```
