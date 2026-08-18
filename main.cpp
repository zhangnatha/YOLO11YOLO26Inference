#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cctype>
#include <limits>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#elif defined(__linux__)
  #include <limits.h>
  #include <unistd.h>
#endif

// ============================================================================
// 兼容 Ubuntu 18.04 (GCC 7) 的 Filesystem 处理
// ============================================================================
#if defined(__has_include)
  #if __has_include(<filesystem>) && __cplusplus >= 201703L && \
      !(defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8)
    #include <filesystem>
    namespace fs = std::filesystem;
  #elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
  #else
    #error "Compiling requires filesystem support!"
  #endif
#else
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#endif

// OpenCV
#include <opencv2/opencv.hpp>

// GLFW & OpenGL
#include <GLFW/glfw3.h>

// Windows' legacy OpenGL header exposes only OpenGL 1.1 constants. The
// texture parameter is supported by all runtime versions used by GLFW, but
// MinGW may not declare its OpenGL 1.2 token.
#ifndef GL_CLAMP_TO_EDGE
  #define GL_CLAMP_TO_EDGE 0x812F
#endif

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// yaml-cpp (用于保存和加载参数配置)
#include <yaml-cpp/yaml.h>

// YOLO Segmentation (Supports YOLO11 & YOLO26 via Factory pattern)
#include "YOLOSegFactory.hpp"
#include "RoiFilter.hpp"
#include "SegmentationAnalysis.hpp"

// ============================================================================
// Global States and Data Structures
// ============================================================================

char modelPath[512] = "";
char imageDirectoryPath[512] = "";
char originalDirectoryPath[512] = "";
char resultPath[512] = "";
fs::path applicationDataDirectory;
std::string imguiIniPath;
float confThreshold = 0.60f;
float iouThreshold = 0.45f;
int deviceChoice = 1; // 0: CPU, 1: GPU
bool showOverlayText = true;
bool roiEnabled = false;
bool roiEditing = false;
roi_filter::Polygon roiNormalizedPoints;
char analysisPredictionDirectory[512] = "";
char analysisGroundTruthDirectory[512] = "";
char analysisOutputDirectory[512] = "";
float analysisIouThreshold = 0.50f;
bool showAnalysisWindow = false;

std::vector<std::string> selectedImages;
std::vector<std::string> logs;
std::mutex logMutex;
std::vector<std::string> pendingLogs;

std::unique_ptr<IYOLOSegDetector> detector = nullptr;
std::atomic<bool> isModelLoaded(false);
std::atomic<bool> isLoadingModel(false);
std::atomic<bool> isDetecting(false);
std::atomic<bool> shouldExit(false); // 安全退出的原子控制信号

std::thread loadModelThread;
std::thread detectionThread;
std::thread analysisThread;
std::atomic<bool> isAnalyzing(false);
std::atomic<bool> shouldStopAnalysis(false);
std::mutex analysisMutex;
segmentation_analysis::Report analysisReport;
std::atomic<unsigned long long> analysisVersion(0);
std::atomic<int> analysisCurrentProgress(0);
std::atomic<int> analysisTotalProgress(0);
std::atomic<long long> analysisElapsedMs(0);
std::atomic<bool> analysisHasProgress(false);

void copyPathArgument(char* destination, size_t capacity,
                      const std::string& value);

struct ReviewItem {
    std::string originalPath;
    std::string resultPath;
};

std::vector<ReviewItem> reviewItems;
std::mutex reviewMutex;
std::atomic<unsigned long long> reviewVersion(0);

// ============================================================================
// Logging System (支持本地存储与隔天自动重命名文件)
// ============================================================================
void addLog(const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* local_tm = std::localtime(&in_time_t);

    std::stringstream ss_time;
    ss_time << std::put_time(local_tm, "%H:%M:%S");
    std::string fullLogLine = ss_time.str() + ": " + msg;
    
    std::stringstream ss_date;
    ss_date << std::put_time(local_tm, "%Y%m%d") << ".log";
    const fs::path logDirectory =
        applicationDataDirectory.empty()
            ? fs::path("logs")
            : applicationDataDirectory / "logs";
    std::error_code directoryError;
    fs::create_directories(logDirectory, directoryError);
    fs::path logFilename = logDirectory / ss_date.str();

    std::lock_guard<std::mutex> lock(logMutex);
    
    pendingLogs.push_back(fullLogLine);

    std::cout << fullLogLine << std::endl;
    std::ofstream logFile(logFilename.string(), std::ios::app);
    if (logFile.is_open()) {
        logFile << fullLogLine << std::endl;
        logFile.close();
    }
}

void flushLogs() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (!pendingLogs.empty()) {
        for (const auto& log : pendingLogs) {
            logs.push_back(log);
        }
        pendingLogs.clear();
    }
}

// ============================================================================
// Configuration Load and Save (YAML)
// ============================================================================
void saveConfig() {
    try {
        YAML::Node config;
        config["modelPath"] = std::string(modelPath);
        config["imageDirectoryPath"] = std::string(imageDirectoryPath);
        config["originalDirectoryPath"] =
            std::string(originalDirectoryPath);
        config["resultPath"] = std::string(resultPath);
        config["confThreshold"] = confThreshold;
        config["iouThreshold"] = iouThreshold;
        config["deviceChoice"] = deviceChoice;
        config["showOverlayText"] = showOverlayText;
        config["roiEnabled"] = roiEnabled &&
            roi_filter::isValid(roiNormalizedPoints);
        YAML::Node roiPoints(YAML::NodeType::Sequence);
        for (const auto& point : roiNormalizedPoints) {
            YAML::Node yamlPoint(YAML::NodeType::Sequence);
            yamlPoint.push_back(roi_filter::clampNormalized(point.x));
            yamlPoint.push_back(roi_filter::clampNormalized(point.y));
            roiPoints.push_back(yamlPoint);
        }
        config["roiNormalizedPoints"] = roiPoints;
        config["analysisPredictionDirectory"] =
            std::string(analysisPredictionDirectory);
        config["analysisGroundTruthDirectory"] =
            std::string(analysisGroundTruthDirectory);
        config["analysisOutputDirectory"] =
            std::string(analysisOutputDirectory);
        config["analysisIouThreshold"] = analysisIouThreshold;

        const fs::path configPath =
            applicationDataDirectory.empty()
                ? fs::path("config.yaml")
                : applicationDataDirectory / "config.yaml";
        std::ofstream fout(configPath.string());
        fout << config;
        fout.close();
        addLog("Configuration saved to " + configPath.string());
    } catch (const std::exception& e) {
        addLog("Failed to save config: " + std::string(e.what()));
    }
}

void loadConfig() {
    const fs::path configPath =
        applicationDataDirectory.empty()
            ? fs::path("config.yaml")
            : applicationDataDirectory / "config.yaml";
    if (!fs::exists(configPath)) {
        addLog("No previous configuration found. Using defaults.");
        return;
    }
    try {
        YAML::Node config = YAML::LoadFile(configPath.string());
        if (config["modelPath"]) copyPathArgument(modelPath, sizeof(modelPath), config["modelPath"].as<std::string>());
        if (config["imageDirectoryPath"]) copyPathArgument(imageDirectoryPath, sizeof(imageDirectoryPath), config["imageDirectoryPath"].as<std::string>());
        if (config["originalDirectoryPath"]) {
            copyPathArgument(originalDirectoryPath,
                             sizeof(originalDirectoryPath),
                             config["originalDirectoryPath"].as<std::string>());
        } else if (config["savePath"]) {
            copyPathArgument(originalDirectoryPath,
                             sizeof(originalDirectoryPath),
                             config["savePath"].as<std::string>());
        }
        if (config["resultPath"]) copyPathArgument(resultPath, sizeof(resultPath), config["resultPath"].as<std::string>());
        if (config["confThreshold"]) confThreshold = config["confThreshold"].as<float>();
        if (config["iouThreshold"]) iouThreshold = config["iouThreshold"].as<float>();
        if (config["deviceChoice"]) deviceChoice = config["deviceChoice"].as<int>();
        if (config["showOverlayText"]) {
            showOverlayText = config["showOverlayText"].as<bool>();
        }
        roiNormalizedPoints.clear();
        const YAML::Node savedRoiPoints = config["roiNormalizedPoints"];
        if (savedRoiPoints && savedRoiPoints.IsSequence()) {
            for (const auto& point : savedRoiPoints) {
                if (!point.IsSequence() || point.size() < 2) continue;
                roiNormalizedPoints.emplace_back(
                    roi_filter::clampNormalized(point[0].as<float>()),
                    roi_filter::clampNormalized(point[1].as<float>()));
            }
        }
        roiEnabled = config["roiEnabled"] &&
            config["roiEnabled"].as<bool>() &&
            roi_filter::isValid(roiNormalizedPoints);
        if (config["analysisPredictionDirectory"]) {
            copyPathArgument(analysisPredictionDirectory,
                sizeof(analysisPredictionDirectory),
                config["analysisPredictionDirectory"].as<std::string>());
        }
        if (config["analysisGroundTruthDirectory"]) {
            copyPathArgument(analysisGroundTruthDirectory,
                sizeof(analysisGroundTruthDirectory),
                config["analysisGroundTruthDirectory"].as<std::string>());
        }
        if (config["analysisOutputDirectory"]) {
            copyPathArgument(analysisOutputDirectory,
                sizeof(analysisOutputDirectory),
                config["analysisOutputDirectory"].as<std::string>());
        }
        if (config["analysisIouThreshold"]) {
            analysisIouThreshold =
                config["analysisIouThreshold"].as<float>();
        }
        addLog("Configuration loaded successfully from " +
               configPath.string());
    } catch (const std::exception& e) {
        addLog("Failed to load config: " + std::string(e.what()));
    }
}

// ============================================================================
// Convert OpenCV Mat to OpenGL Texture
// ============================================================================
GLuint MatToTexture(const cv::Mat& mat) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    cv::Mat rgbMat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgbMat, cv::COLOR_BGRA2RGBA);
    } else {
        cv::cvtColor(mat, rgbMat, cv::COLOR_GRAY2RGB);
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgbMat.cols, rgbMat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgbMat.data);
    return textureID;
}

void releaseTexture(GLuint& textureId) {
    if (textureId != 0) {
        glDeleteTextures(1, &textureId);
        textureId = 0;
    }
}

void clearReviewItems() {
    std::lock_guard<std::mutex> lock(reviewMutex);
    reviewItems.clear();
    ++reviewVersion;
}

void publishReviewItem(const std::string& originalPath,
                       const std::string& visualizationPath) {
    std::lock_guard<std::mutex> lock(reviewMutex);
    reviewItems.push_back({originalPath, visualizationPath});
    ++reviewVersion;
}

size_t getReviewItemCount() {
    std::lock_guard<std::mutex> lock(reviewMutex);
    return reviewItems.size();
}

bool getReviewItem(size_t index, ReviewItem& item) {
    std::lock_guard<std::mutex> lock(reviewMutex);
    if (index >= reviewItems.size()) return false;
    item = reviewItems[index];
    return true;
}

bool loadReviewTextures(size_t index, GLuint& originalTexture,
                        GLuint& resultTexture, cv::Size& originalSize,
                        cv::Size& resultSize) {
    ReviewItem item;
    if (!getReviewItem(index, item)) return false;

    const cv::Mat original = cv::imread(item.originalPath);
    const cv::Mat result = cv::imread(item.resultPath);
    if (original.empty() || result.empty()) {
        addLog("Cannot load review pair: " + item.originalPath);
        return false;
    }

    releaseTexture(originalTexture);
    releaseTexture(resultTexture);
    originalTexture = MatToTexture(original);
    resultTexture = MatToTexture(result);
    originalSize = original.size();
    resultSize = result.size();
    return true;
}

bool copyFileBytes(const fs::path& source, const fs::path& destination) {
    std::error_code equivalentError;
    if (fs::exists(destination) &&
        fs::equivalent(source, destination, equivalentError) &&
        !equivalentError) {
        return true;
    }
    std::ifstream input(source.string(), std::ios::binary);
    if (!input.is_open()) return false;
    std::ofstream output(destination.string(), std::ios::binary | std::ios::trunc);
    if (!output.is_open()) return false;
    output << input.rdbuf();
    return (input.good() || input.eof()) ? output.good() : false;
}

bool saveReviewPair(size_t index) {
    ReviewItem item;
    if (!getReviewItem(index, item)) return false;
    try {
        const fs::path destinationDirectory(originalDirectoryPath);
        fs::create_directories(destinationDirectory);
        const fs::path originalPath(item.originalPath);
        const fs::path resultPathValue(item.resultPath);
        const fs::path originalDestination =
            destinationDirectory / originalPath.filename();
        const fs::path resultName = originalPath.stem().string() + "_r" +
                                    originalPath.extension().string();
        const fs::path resultDestination = destinationDirectory / resultName;
        const bool originalSaved = copyFileBytes(originalPath, originalDestination);
        const bool resultSaved = copyFileBytes(resultPathValue, resultDestination);
        if (!originalSaved || !resultSaved) return false;
        addLog("Original/result pair saved: " + originalDestination.string() +
               " + " + resultDestination.string());
        return true;
    } catch (const std::exception& e) {
        addLog("Failed to save original/result pair: " +
               std::string(e.what()));
        return false;
    }
}

void drawReviewImagePanel(const char* id, const char* title, GLuint texture,
                          const cv::Size& imageSize, const ImVec2& panelSize,
                          roi_filter::Polygon* roiPoints = nullptr,
                          bool roiActive = false,
                          bool roiCanEdit = false) {
    ImGui::BeginChild(id, panelSize, true,
                      ImGuiWindowFlags_NoScrollbar |
                      ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::TextUnformatted(title);
    ImGui::Separator();
    const ImVec2 available = ImGui::GetContentRegionAvail();
    if (texture == 0 || imageSize.width <= 0 || imageSize.height <= 0) {
        const char* placeholder = "No image available";
        const ImVec2 textSize = ImGui::CalcTextSize(placeholder);
        ImGui::SetCursorPos(ImVec2(
            std::max(0.0f, (available.x - textSize.x) * 0.5f),
            ImGui::GetCursorPosY() + std::max(0.0f, (available.y - textSize.y) * 0.5f)));
        ImGui::TextDisabled("%s", placeholder);
        ImGui::EndChild();
        return;
    }

    const float scale = std::min(available.x / imageSize.width,
                                 available.y / imageSize.height);
    const ImVec2 displaySize(imageSize.width * scale, imageSize.height * scale);
    const float offsetX = std::max(0.0f, (available.x - displaySize.x) * 0.5f);
    const float offsetY = std::max(0.0f, (available.y - displaySize.y) * 0.5f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offsetY);
    ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)),
                 displaySize);

    if (roiPoints != nullptr) {
        const ImVec2 imageMin = ImGui::GetItemRectMin();
        const ImVec2 imageMax = ImGui::GetItemRectMax();
        const bool imageHovered = ImGui::IsItemHovered();
        if (roiCanEdit && imageHovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            const ImVec2 mouse = ImGui::GetIO().MousePos;
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                roiPoints->emplace_back(
                    roi_filter::clampNormalized(
                        (mouse.x - imageMin.x) / displaySize.x),
                    roi_filter::clampNormalized(
                        (mouse.y - imageMin.y) / displaySize.y));
            } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) &&
                       !roiPoints->empty()) {
                roiPoints->pop_back();
            }
        }

        std::vector<ImVec2> screenPoints;
        screenPoints.reserve(roiPoints->size());
        for (const auto& point : *roiPoints) {
            screenPoints.emplace_back(
                imageMin.x + roi_filter::clampNormalized(point.x) *
                                 displaySize.x,
                imageMin.y + roi_filter::clampNormalized(point.y) *
                                 displaySize.y);
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        const ImU32 lineColor = roiCanEdit
            ? IM_COL32(255, 190, 0, 255)
            : (roiActive ? IM_COL32(0, 210, 255, 255)
                         : IM_COL32(150, 150, 150, 230));
        if (screenPoints.size() >= 2) {
            const ImDrawFlags flags =
                !roiCanEdit && screenPoints.size() >= 3
                    ? ImDrawFlags_Closed : ImDrawFlags_None;
            drawList->AddPolyline(screenPoints.data(),
                                  static_cast<int>(screenPoints.size()),
                                  lineColor, flags, 3.0f);
        }
        for (const auto& point : screenPoints) {
            drawList->AddCircleFilled(point, 4.5f, lineColor);
            drawList->AddCircle(point, 6.0f, IM_COL32(20, 20, 20, 230),
                                0, 1.0f);
        }
        if (roiCanEdit && imageHovered && !screenPoints.empty()) {
            const ImVec2 mouse = ImGui::GetIO().MousePos;
            const ImVec2 clampedMouse(
                std::max(imageMin.x, std::min(imageMax.x, mouse.x)),
                std::max(imageMin.y, std::min(imageMax.y, mouse.y)));
            drawList->AddLine(screenPoints.back(), clampedMouse,
                              lineColor, 1.5f);
        }
    }
    ImGui::EndChild();
}

// ============================================================================
// Custom ImGui File/Folder Browser
// ============================================================================
struct FileBrowser {
    bool opened = false;
    bool folderMode = false;
    bool multiSelect = false;
    std::string title = "Select File";
    fs::path currentPath = fs::current_path();
    std::string selectedPath = "";
    std::vector<std::string> selectedPaths;
    std::string filterExtension = ""; 

    struct FileEntry {
        std::string name;
        bool isDir;
        bool isChecked;
        FileEntry(std::string n, bool d, bool c = false)
            : name(std::move(n)), isDir(d), isChecked(c) {}
    };
    std::vector<FileEntry> entries;

    void open(const std::string& dialogTitle, bool isFolder, bool isMulti = false, const std::string& filter = "") {
        title = dialogTitle;
        folderMode = isFolder;
        multiSelect = isMulti;
        filterExtension = filter;
        opened = true;
        selectedPath = "";
        selectedPaths.clear();
        refresh();
    }

    void refresh() {
        entries.clear();
        if (currentPath.has_parent_path()) {
            entries.push_back({"..", true, false});
        }
        try {
            for (const auto& entry : fs::directory_iterator(currentPath)) {
                std::string name = entry.path().filename().string();
                bool isDir = fs::is_directory(entry.path()); 

                if (!isDir && !filterExtension.empty() && !folderMode) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    std::string lowerFilter = filterExtension;
                    std::transform(lowerFilter.begin(), lowerFilter.end(), lowerFilter.begin(), ::tolower);

                    if (lowerFilter.find(ext) == std::string::npos) {
                        continue;
                    }
                }
                entries.push_back({name, isDir, false});
            }
        } catch (...) {}

        std::sort(entries.begin(), entries.end(), [](const FileEntry& a, const FileEntry& b) {
            if (a.name == "..") return true;
            if (b.name == "..") return false;
            if (a.isDir != b.isDir) return a.isDir > b.isDir;
            return a.name < b.name;
        });
    }

    bool show() {
        if (!opened) return false;

        bool clickedOK = false;
        ImGui::SetNextWindowSize(ImVec2(700, 450), ImGuiCond_FirstUseEver);
        if (ImGui::Begin(title.c_str(), &opened, ImGuiWindowFlags_NoCollapse)) {
            char manualPath[512] = "";
            copyPathArgument(manualPath, sizeof(manualPath), currentPath.string());
            ImGui::PushItemWidth(-120);
            if (ImGui::InputText("Current Path", manualPath, sizeof(manualPath), ImGuiInputTextFlags_EnterReturnsTrue)) {
                fs::path newPath(manualPath);
                if (fs::exists(newPath) && fs::is_directory(newPath)) {
                    currentPath = newPath;
                    refresh();
                }
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();
            if (ImGui::Button("Refresh", ImVec2(100, 0))) {
                refresh();
            }

            ImGui::Spacing();

            ImGui::BeginChild("FileList", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() - 10), true);
            for (size_t i = 0; i < entries.size(); ++i) {
                auto& entry = entries[i];
                if (entry.isDir) {
                    std::string label = "[Dir] " + entry.name;
                    if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                        if (ImGui::IsMouseDoubleClicked(0)) {
                            if (entry.name == "..") {
                                currentPath = currentPath.parent_path();
                            } else {
                                currentPath /= entry.name;
                            }
                            refresh();
                            break;
                        }
                    }
                } else {
                    std::string label = "[File] " + entry.name;
                    if (multiSelect) {
                        ImGui::Checkbox(label.c_str(), &entry.isChecked);
                    } else {
                        std::string fullFilePath = (currentPath / entry.name).string();
                        if (ImGui::Selectable(label.c_str(), selectedPath == fullFilePath)) {
                            selectedPath = fullFilePath;
                        }
                    }
                }
            }
            ImGui::EndChild();

            if (folderMode) {
                if (ImGui::Button("Select Current Folder", ImVec2(180, 0))) {
                    selectedPath = currentPath.string();
                    opened = false;
                    clickedOK = true;
                }
                ImGui::SameLine();
            } else if (multiSelect) {
                if (ImGui::Button("OK", ImVec2(100, 0))) {
                    selectedPaths.clear();
                    for (const auto& entry : entries) {
                        if (!entry.isDir && entry.isChecked) {
                            selectedPaths.push_back((currentPath / entry.name).string());
                        }
                    }
                    opened = false;
                    clickedOK = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Select All", ImVec2(100, 0))) {
                    for (auto& entry : entries) {
                        if (!entry.isDir) {
                            entry.isChecked = true;
                        }
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Deselect All", ImVec2(110, 0))) {
                    for (auto& entry : entries) {
                        if (!entry.isDir) {
                            entry.isChecked = false;
                        }
                    }
                }
                ImGui::SameLine();
            } else {
                if (ImGui::Button("OK", ImVec2(100, 0))) {
                    if (!selectedPath.empty()) {
                        opened = false;
                        clickedOK = true;
                    }
                }
                ImGui::SameLine();
            }

            if (ImGui::Button("Cancel", ImVec2(100, 0))) {
                opened = false;
            }
        }
        ImGui::End();
        return clickedOK;
    }
};

FileBrowser fileBrowser;

// ============================================================================
// ONNX model labels resolver
// ============================================================================
std::string findLabelsPath(const std::string& modelPathStr) {
    fs::path modelPath(modelPathStr);
    std::string base = modelPath.stem().string();
    fs::path dir = modelPath.parent_path();

    std::vector<fs::path> candidates = {
        dir / (base + ".names"),
        dir / (base + ".txt"),
        dir / "classes.names",
        dir / "labels.txt",
        fs::current_path() / "models/classes.names"
    };

    for (const auto& c : candidates) {
        if (fs::exists(c)) {
            return c.string();
        }
    }
    return "models/classes.names"; 
}

// ============================================================================
// Background model loading and inference threads
// ============================================================================

void loadModelWorker() {
    isLoadingModel = true;
    const std::string requestedModelPath(modelPath);
    const int requestedDeviceChoice = deviceChoice;
    addLog("Loading model: " + requestedModelPath);
    try {
        std::string labelsFile = findLabelsPath(requestedModelPath);
        addLog("Matched labels file: " + labelsFile);
        
        bool requestGPU = (requestedDeviceChoice == 1);
        if (requestGPU) {
            try {
                addLog("Attempting to load model with GPU support...");
                detector = YOLOSegFactory::createDetector(
                    requestedModelPath, labelsFile, true);
                isModelLoaded = true;
                addLog("ONNX segmentation model loaded successfully on GPU!");
            } catch (const std::exception& gpu_err) {
                addLog("GPU loading failed: " + std::string(gpu_err.what()));
                addLog("Falling back to CPU mode...");
                detector = YOLOSegFactory::createDetector(
                    requestedModelPath, labelsFile, false);
                isModelLoaded = true;
                addLog("ONNX segmentation model loaded successfully on CPU!");
            }
        } else {
            addLog("Attempting to load model on CPU (user choice)...");
            detector = YOLOSegFactory::createDetector(
                requestedModelPath, labelsFile, false);
            isModelLoaded = true;
            addLog("ONNX segmentation model loaded successfully on CPU!");
        }
    } catch (const std::exception& e) {
        addLog("Failed to load model: " + std::string(e.what()));
        isModelLoaded = false;
    }
    isLoadingModel = false;
}

void startLoadingModel() {
    if (isLoadingModel) return;
    if (loadModelThread.joinable()) {
        loadModelThread.join();
    }
    isLoadingModel = true;
    isModelLoaded = false;
    loadModelThread = std::thread(loadModelWorker);
}

// Draw X-AnyLabeling sidecar annotations as green contours. JSON is a YAML
// subset, so the existing yaml-cpp dependency can parse these files directly.
struct GroundTruthItem {
    std::string label;
    cv::Rect bounds;
};

std::vector<GroundTruthItem> drawGroundTruthContours(
    cv::Mat& image, const std::string& imagePath) {
    fs::path jsonPath(imagePath);
    jsonPath.replace_extension(".json");
    if (!fs::exists(jsonPath)) return {};

    const YAML::Node root = YAML::LoadFile(jsonPath.string());
    const YAML::Node shapes = root["shapes"];
    if (!shapes || !shapes.IsSequence()) return {};

    const cv::Scalar groundTruthColor(0, 255, 0);  // OpenCV BGR: green
    std::vector<GroundTruthItem> items;
    for (const auto& shape : shapes) {
        const YAML::Node pointNodes = shape["points"];
        if (!pointNodes || !pointNodes.IsSequence()) continue;

        std::vector<cv::Point> points;
        for (const auto& pointNode : pointNodes) {
            if (!pointNode.IsSequence() || pointNode.size() < 2) continue;
            const int x = std::max(0, std::min(image.cols - 1,
                cvRound(pointNode[0].as<double>())));
            const int y = std::max(0, std::min(image.rows - 1,
                cvRound(pointNode[1].as<double>())));
            points.emplace_back(x, y);
        }

        const std::string shapeType = shape["shape_type"]
            ? shape["shape_type"].as<std::string>() : "polygon";
        if (shapeType == "rectangle" && points.size() >= 2) {
            const cv::Point a = points[0];
            const cv::Point b = points[1];
            points = {cv::Point(a.x, a.y), cv::Point(b.x, a.y),
                      cv::Point(b.x, b.y), cv::Point(a.x, b.y)};
        }
        if (points.size() < 3) continue;

        std::vector<std::vector<cv::Point>> contourList{points};
        cv::drawContours(image, contourList, -1, groundTruthColor, 3,
                         cv::LINE_AA);
        items.push_back({shape["label"]
            ? shape["label"].as<std::string>() : "unknown",
            cv::boundingRect(points)});
    }
    return items;
}

struct LegendEntry {
    std::string text;
    cv::Scalar color;
};

void drawResultsLegend(
    cv::Mat& image,
    const std::vector<GroundTruthItem>& groundTruthItems,
    const std::vector<SegmentedDetection>& predictions,
    const std::vector<std::string>& classNames) {
    if (image.empty()) return;

    const cv::Scalar groundTruthColor(0, 255, 0);
    const cv::Scalar predictionHeaderColor(255, 80, 0);
    std::vector<LegendEntry> entries;

    entries.push_back({"GT - green contour", groundTruthColor});
    if (groundTruthItems.empty()) {
        entries.push_back({"  no annotation JSON", groundTruthColor});
    } else {
        for (size_t i = 0; i < groundTruthItems.size(); ++i) {
            entries.push_back({"  #" + std::to_string(i + 1) + "  " +
                               groundTruthItems[i].label, groundTruthColor});
        }
    }

    entries.push_back({"PRED - class colors", predictionHeaderColor});
    if (predictions.empty()) {
        entries.push_back({"  no prediction", predictionHeaderColor});
    }
    for (size_t i = 0; i < predictions.size(); ++i) {
        const auto& prediction = predictions[i];
        const bool knownClass = prediction.classId >= 0 &&
            static_cast<size_t>(prediction.classId) < classNames.size();
        const std::string className = knownClass
            ? classNames[static_cast<size_t>(prediction.classId)]
            : "class_" + std::to_string(prediction.classId);
        std::ostringstream line;
        line << "  #" << (i + 1) << "  " << className << "  "
             << std::fixed << std::setprecision(3) << prediction.conf;
        entries.push_back(
            {line.str(), predictionColorForClass(prediction.classId)});
    }

    const int margin = 8;
    const int padding = 8;
    const int swatchSize = 11;
    const int lineHeight = 19;
    const double fontScale = 0.45;
    const int fontThickness = 1;
    const int maxRows = std::max(1,
        (image.rows - 2 * margin - 2 * padding) / lineHeight);
    if (static_cast<int>(entries.size()) > maxRows) {
        entries.resize(static_cast<size_t>(maxRows));
    }

    int maxTextWidth = 0;
    for (const auto& entry : entries) {
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            entry.text, cv::FONT_HERSHEY_SIMPLEX, fontScale,
            fontThickness, &baseline);
        maxTextWidth = std::max(maxTextWidth, textSize.width);
    }
    const int panelWidth = std::min(image.cols - 2 * margin,
        3 * padding + swatchSize + maxTextWidth);
    const int panelHeight = std::min(image.rows - 2 * margin,
        2 * padding + lineHeight * static_cast<int>(entries.size()));
    if (panelWidth <= 0 || panelHeight <= 0) return;

    const cv::Rect panel(margin, margin, panelWidth, panelHeight);
    cv::Mat panelRoi = image(panel);
    cv::Mat darkPanel(panelRoi.size(), panelRoi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(panelRoi, 0.25, darkPanel, 0.75, 0.0, panelRoi);

    for (size_t i = 0; i < entries.size(); ++i) {
        const int centerY = margin + padding +
            static_cast<int>(i) * lineHeight + lineHeight / 2;
        const int swatchX = margin + padding;
        cv::rectangle(image,
                      cv::Point(swatchX, centerY - swatchSize / 2),
                      cv::Point(swatchX + swatchSize,
                                centerY + swatchSize / 2),
                      entries[i].color, cv::FILLED);
        cv::putText(image, entries[i].text,
                    cv::Point(swatchX + swatchSize + padding,
                              centerY + 5),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale,
                    cv::Scalar(255, 255, 255), fontThickness, cv::LINE_AA);
    }
}

void drawInstanceTag(cv::Mat& image, const std::string& id,
                     cv::Point anchor, const cv::Scalar& color,
                     bool darkText) {
    int baseline = 0;
    const double fontScale = 0.42;
    const cv::Size textSize = cv::getTextSize(
        id, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
    const int width = textSize.width + 8;
    const int height = textSize.height + 8;
    int x = std::max(0, std::min(anchor.x, image.cols - width));
    int y = std::max(0, std::min(anchor.y - height, image.rows - height));
    cv::rectangle(image, cv::Rect(x, y, width, height), color, cv::FILLED);
    cv::rectangle(image, cv::Rect(x, y, width, height),
                  cv::Scalar(20, 20, 20), 1);
    cv::putText(image, id, cv::Point(x + 4, y + textSize.height + 3),
                cv::FONT_HERSHEY_SIMPLEX, fontScale,
                darkText ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255),
                1, cv::LINE_AA);
}

void drawInstanceTags(cv::Mat& image,
                      const std::vector<GroundTruthItem>& groundTruthItems,
                      const std::vector<SegmentedDetection>& predictions) {
    for (size_t i = 0; i < groundTruthItems.size(); ++i) {
        const cv::Rect& bounds = groundTruthItems[i].bounds;
        drawInstanceTag(image, std::to_string(i + 1),
                        cv::Point(bounds.x, bounds.y), cv::Scalar(0, 255, 0),
                        true);
    }
    for (size_t i = 0; i < predictions.size(); ++i) {
        const cv::Scalar predictionColor =
            predictionColorForClass(predictions[i].classId);
        const double brightness =
            0.114 * predictionColor[0] +
            0.587 * predictionColor[1] +
            0.299 * predictionColor[2];
        const auto& box = predictions[i].box;
        drawInstanceTag(image, std::to_string(i + 1),
                        cv::Point(box.x + box.width, box.y),
                        predictionColor, brightness > 150.0);
    }
}

void detectionWorker() {
    isDetecting = true;
    addLog("---------- Start Batch Detection ----------");
    const float currentConfThreshold = confThreshold;
    const float currentIouThreshold = iouThreshold;
    const bool currentShowOverlayText = showOverlayText;
    const roi_filter::Polygon currentRoiPoints = roiNormalizedPoints;
    const bool currentRoiEnabled =
        roiEnabled && roi_filter::isValid(currentRoiPoints);
    if (currentRoiEnabled) {
        addLog("ROI filtering enabled (mask/box intersection clipping).");
    }
    
    if (!detector) {
        addLog("Error: Model not loaded! Please click [1_Start Model] first.");
        isDetecting = false;
        return;
    }

    try {
        fs::create_directories(resultPath);
    } catch (...) {
        addLog("Error: Cannot create result directory: " +
               std::string(resultPath));
        isDetecting = false;
        return;
    }

    clearReviewItems();
    const int totalImages = static_cast<int>(selectedImages.size());
    int completedCount = 0;
    int failedCount = 0;

    for (int i = 0; i < totalImages; ++i) {
        if (shouldExit) break;

        const std::string imgPath = selectedImages[static_cast<size_t>(i)];
        addLog("Inference on image [" + std::to_string(i + 1) + "/" + std::to_string(totalImages) + "]: " + imgPath);

        cv::Mat rawImage = cv::imread(imgPath);
        if (rawImage.empty()) {
            addLog("  Error: Cannot read image " + imgPath);
            ++failedCount;
            continue;
        }

        std::vector<SegmentedDetection> results;
        try {
            results = detector->segment(rawImage, currentConfThreshold, currentIouThreshold);
        } catch (const std::exception& e) {
            addLog("  Error: Inference failed: " + std::string(e.what()));
            ++failedCount;
            continue;
        }

        if (currentRoiEnabled) {
            const size_t unfilteredCount = results.size();
            results = roi_filter::filter(
                results, rawImage.size(), currentRoiPoints);
            addLog("  ROI kept " + std::to_string(results.size()) + "/" +
                   std::to_string(unfilteredCount) + " prediction(s)");
        }

        const auto& classNames = detector->getClassNames();
        cv::Mat visualImg = rawImage.clone();
        std::vector<GroundTruthItem> groundTruthItems;
        try {
            groundTruthItems = drawGroundTruthContours(visualImg, imgPath);
            if (!groundTruthItems.empty()) {
                addLog("  Ground truth: " +
                       std::to_string(groundTruthItems.size()) +
                       " green contour(s)");
            }
        } catch (const std::exception& e) {
            addLog("  Warning: cannot parse annotation JSON: " +
                   std::string(e.what()));
        }
        detector->drawSegmentationsAndBoxes(visualImg, results);
        drawInstanceTags(visualImg, groundTruthItems, results);
        if (currentShowOverlayText) {
            drawResultsLegend(visualImg, groundTruthItems, results, classNames);
        }
        
        const fs::path p(imgPath);
        const fs::path visualName =
            p.stem().string() + "_r" + p.extension().string();
        const fs::path visualDest = fs::path(resultPath) / visualName;
        if (cv::imwrite(visualDest.string(), visualImg)) {
            addLog("  Visualization saved to: " + visualDest.string());
            const fs::path jsonDest = fs::path(resultPath) /
                (p.stem().string() + "_r.json");
            std::string jsonError;
            std::vector<segmentation_analysis::Prediction> jsonPredictions;
            jsonPredictions.reserve(results.size());
            for (const auto& result : results) {
                jsonPredictions.push_back(
                    {result.conf, result.classId, result.mask});
            }
            if (segmentation_analysis::writePredictionJson(
                    jsonDest.string(), visualName.string(), rawImage.size(),
                    jsonPredictions, classNames, &jsonError)) {
                addLog("  Prediction JSON saved to: " + jsonDest.string());
            } else {
                addLog("  Warning: cannot save prediction JSON: " + jsonError);
            }
            publishReviewItem(imgPath, visualDest.string());
            ++completedCount;
        } else {
            addLog("  Failed to save visualization to: " + visualDest.string());
            ++failedCount;
        }
    }

    addLog("Detection finished. Total: " + std::to_string(totalImages) +
           ", completed: " + std::to_string(completedCount) +
           ", failed: " + std::to_string(failedCount) +
           ". Use the review client to save selected originals manually.");
    isDetecting = false;
}

void startDetection() {
    if (isDetecting) return;
    if (detectionThread.joinable()) {
        detectionThread.join();
    }
    detectionThread = std::thread(detectionWorker);
}

segmentation_analysis::Report getAnalysisReportSnapshot() {
    std::lock_guard<std::mutex> lock(analysisMutex);
    return analysisReport;
}

void analysisWorker() {
    {
        std::lock_guard<std::mutex> lock(analysisMutex);
        analysisReport = segmentation_analysis::Report();
        ++analysisVersion;
    }
    analysisCurrentProgress = 0;
    analysisTotalProgress = 0;
    analysisElapsedMs = 0;
    analysisHasProgress = false;
    shouldStopAnalysis = false;

    const std::string predictionDirectory(analysisPredictionDirectory);
    const std::string groundTruthDirectory(analysisGroundTruthDirectory);
    const std::string outputDirectory(analysisOutputDirectory);
    const float matchThreshold = analysisIouThreshold;
    const bool includeOverlayText = showOverlayText;
    addLog("---------- Start Prediction/GT Analysis ----------");
    try {
        auto report = segmentation_analysis::analyzeDirectories(
            predictionDirectory, groundTruthDirectory, outputDirectory,
            matchThreshold, includeOverlayText,
            [](const std::string& message) { addLog(message); },
            [](int current, int total, long long elapsedMs) {
                analysisCurrentProgress = current;
                analysisTotalProgress = total;
                analysisElapsedMs = elapsedMs;
                analysisHasProgress = true;
            },
            [&]() {
                return shouldStopAnalysis.load() || shouldExit.load();
            });
        {
            std::lock_guard<std::mutex> lock(analysisMutex);
            analysisReport = std::move(report);
            ++analysisVersion;
        }
        const auto snapshot = getAnalysisReportSnapshot();
        if (shouldStopAnalysis.load()) {
            addLog("Analysis stopped by user. Processed images: " +
                   std::to_string(snapshot.images) + ", missed: " +
                   std::to_string(snapshot.total.falseNegative) +
                   ", false positives: " +
                   std::to_string(snapshot.total.falsePositive));
        } else {
            addLog("Analysis finished. Images: " +
                   std::to_string(snapshot.images) + ", missed: " +
                   std::to_string(snapshot.total.falseNegative) +
                   ", false positives: " +
                   std::to_string(snapshot.total.falsePositive));
        }
        addLog("Analysis CSV: " + snapshot.csvPath);
        addLog("Analysis JSON: " + snapshot.jsonPath);
    } catch (const std::exception& error) {
        addLog("Analysis failed: " + std::string(error.what()));
        ++analysisVersion;
    }
    isAnalyzing = false;
}

void startAnalysis() {
    if (isAnalyzing.exchange(true)) return;
    shouldStopAnalysis = false;
    if (analysisThread.joinable()) analysisThread.join();
    analysisThread = std::thread(analysisWorker);
}

void stopAnalysis() {
    if (isAnalyzing.load()) {
        shouldStopAnalysis = true;
        addLog("Stopping analysis...");
    }
}

bool loadAnalysisTextures(size_t index,
                          GLuint& originalTexture,
                          GLuint& visualizationTexture,
                          cv::Size& originalSize,
                          cv::Size& visualizationSize) {
    const auto snapshot = getAnalysisReportSnapshot();
    if (index >= snapshot.reviewItems.size()) return false;
    const auto& item = snapshot.reviewItems[index];
    const cv::Mat original = cv::imread(item.originalPath);
    const cv::Mat visualization = cv::imread(item.visualizationPath);
    if (original.empty() || visualization.empty()) {
        addLog("Cannot load analysis comparison: " + item.visualizationPath);
        return false;
    }
    releaseTexture(originalTexture);
    releaseTexture(visualizationTexture);
    originalTexture = MatToTexture(original);
    visualizationTexture = MatToTexture(visualization);
    originalSize = original.size();
    visualizationSize = visualization.size();
    return true;
}

void copyPathArgument(char* destination, size_t capacity,
                      const std::string& value) {
    if (capacity == 0) return;
    std::strncpy(destination, value.c_str(), capacity - 1);
    destination[capacity - 1] = '\0';
}

bool isSupportedImage(const fs::path& path) {
    if (!fs::is_regular_file(path)) return false;
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".bmp" ||
           extension == ".tif" || extension == ".tiff";
}

std::vector<std::string> findImagesInDirectory(const fs::path& directory) {
    std::vector<std::string> images;
    if (!fs::exists(directory) || !fs::is_directory(directory)) return images;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!isSupportedImage(entry.path())) continue;
        const std::string stem = entry.path().stem().string();
        if (stem.size() >= 2 && stem.compare(stem.size() - 2, 2, "_r") == 0) {
            continue;
        }
        images.push_back(entry.path().string());
    }
    std::sort(images.begin(), images.end());
    return images;
}

fs::path resolveExecutablePath(const char* argumentZero) {
#if defined(_WIN32)
    std::vector<wchar_t> pathBuffer(32768);
    const DWORD length = GetModuleFileNameW(
        nullptr, pathBuffer.data(), static_cast<DWORD>(pathBuffer.size()));
    if (length > 0 && length < pathBuffer.size()) {
        return fs::path(std::wstring(pathBuffer.data(), length));
    }
#elif defined(__linux__)
    std::vector<char> pathBuffer(PATH_MAX + 1, '\0');
    const ssize_t length = readlink(
        "/proc/self/exe", pathBuffer.data(), pathBuffer.size() - 1);
    if (length > 0) {
        pathBuffer[static_cast<size_t>(length)] = '\0';
        return fs::path(pathBuffer.data());
    }
#endif
    return fs::absolute(fs::path(argumentZero ? argumentZero : "yolo_seg"));
}

void initializeDefaultOutputDirectories(const char* executablePath) {
    fs::path executable = resolveExecutablePath(executablePath);
    fs::path executableDirectory = executable.parent_path();
    applicationDataDirectory =
        executableDirectory.filename() == "bin"
            ? executableDirectory.parent_path()
            : executableDirectory;
    if (applicationDataDirectory.empty()) {
        applicationDataDirectory = fs::current_path() / "build";
    }

    fs::path resourceDirectory;
    fs::path candidateDirectory = applicationDataDirectory;
    for (int depth = 0; depth < 4; ++depth) {
        if (fs::exists(candidateDirectory / "model") &&
            fs::exists(candidateDirectory / "images")) {
            resourceDirectory = candidateDirectory;
            break;
        }
        if (!candidateDirectory.has_parent_path()) break;
        candidateDirectory = candidateDirectory.parent_path();
    }
    if (resourceDirectory.empty()) {
        resourceDirectory = fs::current_path();
    }

    const fs::path resultsDirectory =
        applicationDataDirectory / "results";
    const fs::path originalsDirectory =
        applicationDataDirectory / "original";
    std::error_code directoryError;
    fs::create_directories(resultsDirectory, directoryError);
    directoryError.clear();
    fs::create_directories(originalsDirectory, directoryError);
    copyPathArgument(resultPath, sizeof(resultPath),
                     resultsDirectory.string());
    copyPathArgument(originalDirectoryPath,
                     sizeof(originalDirectoryPath),
                     originalsDirectory.string());
    copyPathArgument(
        modelPath, sizeof(modelPath),
        (resourceDirectory / "model" / "suidong_20260727YOLO26m.dmmodel" /
         "segmentation.onnx").string());
    copyPathArgument(imageDirectoryPath, sizeof(imageDirectoryPath),
                     (resourceDirectory / "images").string());
    copyPathArgument(analysisPredictionDirectory,
                     sizeof(analysisPredictionDirectory),
                     resultsDirectory.string());
    copyPathArgument(analysisGroundTruthDirectory,
                     sizeof(analysisGroundTruthDirectory),
                     (resourceDirectory / "images").string());
    copyPathArgument(analysisOutputDirectory,
                     sizeof(analysisOutputDirectory),
                     (applicationDataDirectory / "analysis").string());
    imguiIniPath = (applicationDataDirectory / "imgui.ini").string();
}

void printCommandLineHelp(const char* executable) {
    std::cout
        << "Usage:\n"
        << "  " << executable << "                         Launch GUI\n"
        << "  " << executable << " --batch-images DIR [options]  Batch inference\n"
        << "  " << executable << " --analyze-predictions DIR --ground-truth DIR [options]\n\n"
        << "Options:\n"
        << "  --model FILE       ONNX model path\n"
        << "  --output DIR       Result directory (default: executable/results)\n"
        << "  --conf FLOAT       Confidence threshold (default: 0.60)\n"
        << "  --iou FLOAT        NMS IoU threshold (default: 0.45)\n"
        << "  --cpu | --gpu      Execution provider (GPU falls back to CPU)\n"
        << "  --analysis-output DIR  Analysis report/image directory\n"
        << "  --match-iou FLOAT      Prediction/GT matching IoU (default: 0.50)\n"
        << "  --hide-overlay-text    Hide top-left text in generated images\n"
        << "  --help              Show this message\n";
}

// Returns -1 when GUI mode should continue, otherwise a process exit code.
int runBatchCommand(int argc, char** argv) {
    if (argc <= 1) return -1;

    std::string imageDirectory;
    bool analysisMode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto nextValue = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + option);
            }
            return argv[++i];
        };

        try {
            if (argument == "--help" || argument == "-h") {
                printCommandLineHelp(argv[0]);
                return 0;
            } else if (argument == "--batch-images") {
                imageDirectory = nextValue(argument);
            } else if (argument == "--analyze-predictions") {
                analysisMode = true;
                copyPathArgument(analysisPredictionDirectory,
                    sizeof(analysisPredictionDirectory), nextValue(argument));
            } else if (argument == "--ground-truth") {
                copyPathArgument(analysisGroundTruthDirectory,
                    sizeof(analysisGroundTruthDirectory), nextValue(argument));
            } else if (argument == "--analysis-output") {
                copyPathArgument(analysisOutputDirectory,
                    sizeof(analysisOutputDirectory), nextValue(argument));
            } else if (argument == "--match-iou") {
                analysisIouThreshold = std::stof(nextValue(argument));
            } else if (argument == "--hide-overlay-text") {
                showOverlayText = false;
            } else if (argument == "--model") {
                copyPathArgument(modelPath, sizeof(modelPath), nextValue(argument));
            } else if (argument == "--output") {
                copyPathArgument(resultPath, sizeof(resultPath), nextValue(argument));
            } else if (argument == "--conf") {
                confThreshold = std::stof(nextValue(argument));
            } else if (argument == "--iou") {
                iouThreshold = std::stof(nextValue(argument));
            } else if (argument == "--cpu") {
                deviceChoice = 0;
            } else if (argument == "--gpu") {
                deviceChoice = 1;
            } else {
                std::cerr << "Unknown option: " << argument << "\n";
                printCommandLineHelp(argv[0]);
                return 2;
            }
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 2;
        }
    }

    if (analysisMode) {
        if (!imageDirectory.empty()) {
            std::cerr << "Batch inference and analysis modes cannot be combined.\n";
            return 2;
        }
        try {
            const auto report = segmentation_analysis::analyzeDirectories(
                analysisPredictionDirectory,
                analysisGroundTruthDirectory,
                analysisOutputDirectory,
                analysisIouThreshold,
                showOverlayText,
                [](const std::string& message) {
                    std::cout << message << '\n';
                });
            std::cout << "Analysis complete: images=" << report.images
                      << ", missed=" << report.total.falseNegative
                      << ", false_positive=" << report.total.falsePositive
                      << "\nCSV: " << report.csvPath
                      << "\nJSON: " << report.jsonPath << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "Analysis failed: " << error.what() << '\n';
            return 5;
        }
    }

    if (imageDirectory.empty()) {
        std::cerr << "--batch-images is required in command-line mode.\n";
        return 2;
    }
    const fs::path directory(imageDirectory);
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Image directory does not exist: " << imageDirectory << "\n";
        return 2;
    }

    selectedImages = findImagesInDirectory(directory);
    if (selectedImages.empty()) {
        std::cerr << "No supported images found in: " << imageDirectory << "\n";
        return 3;
    }

    addLog("Batch mode selected " + std::to_string(selectedImages.size()) +
           " image(s) from " + imageDirectory);
    loadModelWorker();
    if (!isModelLoaded || !detector) return 4;
    detectionWorker();
    detector.reset();
    isModelLoaded = false;
    return 0;
}

// ============================================================================
// Main Function & Interface Rendering
// ============================================================================

static void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

int main(int argc, char** argv) {
    initializeDefaultOutputDirectories(argc > 0 ? argv[0] : "yolo_seg");
    const int batchResult = runBatchCommand(argc, argv);
    if (batchResult >= 0) return batchResult;

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    const char* glslVersion = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* window = glfwCreateWindow(
        1440, 850, "YOLO Segmentation Review Client", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = imguiIniPath.c_str();
    ImGui::StyleColorsLight();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.PopupRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.ItemSpacing = ImVec2(8, 7);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.92f, 0.92f, 0.92f, 1.00f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.92f, 0.94f, 0.97f, 1.00f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.85f, 0.88f, 0.92f, 1.00f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.84f, 0.87f, 0.92f, 1.00f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.70f, 0.78f, 0.90f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.60f, 0.70f, 0.86f, 1.00f);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glslVersion);

    addLog("-------- Segmentation review client started --------");
    loadConfig();
    selectedImages = findImagesInDirectory(fs::path(imageDirectoryPath));
    if (!selectedImages.empty()) {
        addLog("Loaded " + std::to_string(selectedImages.size()) +
               " image(s) from configured directory.");
    }

    GLuint originalTexture = 0;
    GLuint resultTexture = 0;
    cv::Size originalSize;
    cv::Size resultSize;
    GLuint roiPreviewTexture = 0;
    cv::Size roiPreviewSize;
    GLuint analysisOriginalTexture = 0;
    GLuint analysisVisualizationTexture = 0;
    cv::Size analysisOriginalSize;
    cv::Size analysisVisualizationSize;
    size_t currentAnalysisIndex = 0;
    size_t displayedAnalysisIndex = std::numeric_limits<size_t>::max();
    unsigned long long observedAnalysisVersion = analysisVersion.load();
    size_t currentReviewIndex = 0;
    size_t displayedReviewIndex = std::numeric_limits<size_t>::max();
    unsigned long long observedReviewVersion = reviewVersion.load();

    auto displayReviewIndex = [&](size_t index) {
        if (loadReviewTextures(index, originalTexture, resultTexture,
                               originalSize, resultSize)) {
            currentReviewIndex = index;
            displayedReviewIndex = index;
            return true;
        }
        return false;
    };
    auto loadRoiPreview = [&]() {
        releaseTexture(roiPreviewTexture);
        roiPreviewSize = cv::Size();
        if (selectedImages.empty()) return false;
        const cv::Mat preview = cv::imread(selectedImages.front());
        if (preview.empty()) {
            addLog("Cannot load ROI preview: " + selectedImages.front());
            return false;
        }
        roiPreviewTexture = MatToTexture(preview);
        roiPreviewSize = preview.size();
        return true;
    };
    auto saveCurrentReview = [&]() {
        if (!saveReviewPair(currentReviewIndex)) {
            addLog("Failed to save the current original/result pair.");
            return false;
        }
        return true;
    };
    loadRoiPreview();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        flushLogs();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        size_t reviewCount = getReviewItemCount();
        const unsigned long long currentVersion = reviewVersion.load();
        if (currentVersion != observedReviewVersion) {
            observedReviewVersion = currentVersion;
            if (reviewCount == 0) {
                releaseTexture(originalTexture);
                releaseTexture(resultTexture);
                originalSize = cv::Size();
                resultSize = cv::Size();
                currentReviewIndex = 0;
                displayedReviewIndex = std::numeric_limits<size_t>::max();
            } else if (displayedReviewIndex ==
                       std::numeric_limits<size_t>::max()) {
                displayReviewIndex(0);
            }
        }

        const unsigned long long currentAnalysisVersion =
            analysisVersion.load();
        if (currentAnalysisVersion != observedAnalysisVersion) {
            observedAnalysisVersion = currentAnalysisVersion;
            const auto snapshot = getAnalysisReportSnapshot();
            if (snapshot.reviewItems.empty()) {
                releaseTexture(analysisOriginalTexture);
                releaseTexture(analysisVisualizationTexture);
                analysisOriginalSize = cv::Size();
                analysisVisualizationSize = cv::Size();
                currentAnalysisIndex = 0;
                displayedAnalysisIndex = std::numeric_limits<size_t>::max();
            } else if (loadAnalysisTextures(
                           0, analysisOriginalTexture,
                           analysisVisualizationTexture,
                           analysisOriginalSize,
                           analysisVisualizationSize)) {
                currentAnalysisIndex = 0;
                displayedAnalysisIndex = 0;
            }
        }

        bool requestPrevious = false;
        bool requestNext = false;
        bool requestSave = false;
        if (reviewCount > 0 && !roiEditing && !showAnalysisWindow &&
            !fileBrowser.opened &&
            !io.WantTextInput && !ImGui::IsAnyItemActive()) {
            requestPrevious = ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false);
            requestNext = ImGui::IsKeyPressed(ImGuiKey_RightArrow, false);
            requestSave = ImGui::IsKeyPressed(ImGuiKey_S, false);
        }
        if (requestPrevious && currentReviewIndex > 0) {
            displayReviewIndex(currentReviewIndex - 1);
        }
        if (requestNext && currentReviewIndex + 1 < reviewCount) {
            displayReviewIndex(currentReviewIndex + 1);
        }
        if (requestSave &&
            displayedReviewIndex != std::numeric_limits<size_t>::max()) {
            saveCurrentReview();
        }

        int windowWidth = 0;
        int windowHeight = 0;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(
            ImVec2(static_cast<float>(windowWidth),
                   static_cast<float>(windowHeight)));
        ImGui::Begin(
            "YOLO Review Client", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        ImGui::Text("Inference and Manual Review");
        ImGui::SameLine();
        if (ImGui::Button("Open Analysis UI")) {
            showAnalysisWindow = true;
        }
        ImGui::SameLine();
        if (isLoadingModel) {
            ImGui::TextColored(ImVec4(0.85f, 0.55f, 0.0f, 1.0f),
                               "Model: loading");
        } else if (isModelLoaded) {
            ImGui::TextColored(ImVec4(0.0f, 0.60f, 0.15f, 1.0f),
                               "Model: ready");
        } else {
            ImGui::TextDisabled("Model: not loaded");
        }
        ImGui::SameLine();
        if (isDetecting) {
            ImGui::TextColored(ImVec4(0.1f, 0.4f, 0.9f, 1.0f),
                               "Inference running...");
        }
        if (isAnalyzing) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.1f, 0.4f, 0.9f, 1.0f),
                               "Analysis running...");
        }
        ImGui::Separator();

        ImGui::BeginChild("Settings", ImVec2(0, 215), true);
        const bool settingsLocked = isLoadingModel || isDetecting;
        if (settingsLocked) ImGui::BeginDisabled();
        auto drawPathInput = [&](const char* label, const char* inputId,
                                 char* buffer, size_t bufferSize,
                                 const char* browseId,
                                 const std::string& browserTitle,
                                 bool folderMode,
                                 const std::string& extension) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label);
            ImGui::SameLine(155);
            ImGui::SetNextItemWidth(
                std::max(180.0f, ImGui::GetContentRegionAvail().x - 92.0f));
            const bool changed = ImGui::InputText(inputId, buffer, bufferSize);
            ImGui::SameLine();
            if (ImGui::Button(browseId)) {
                fileBrowser.open(browserTitle, folderMode, false, extension);
            }
            return changed;
        };

        const bool modelPathChanged =
            drawPathInput("ONNX model", "##ModelPath", modelPath,
                          sizeof(modelPath), "Browse##Model",
                          "Select ONNX Model", false, ".onnx");
        const bool imageDirectoryChanged =
            drawPathInput("Image directory", "##ImageDirectory",
                          imageDirectoryPath, sizeof(imageDirectoryPath),
                          "Browse##Images", "Select Image Directory", true, "");
        drawPathInput("Result directory", "##ResultPath", resultPath,
                      sizeof(resultPath), "Browse##Result",
                      "Select Result Directory", true, "");
        drawPathInput("Original directory", "##OriginalPath",
                      originalDirectoryPath,
                      sizeof(originalDirectoryPath), "Browse##Original",
                      "Select Original Directory", true, "");

        ImGui::SetNextItemWidth(180);
        ImGui::SliderFloat("Confidence", &confThreshold, 0.0f, 1.0f, "%.2f");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(180);
        ImGui::SliderFloat("IoU", &iouThreshold, 0.0f, 1.0f, "%.2f");
        ImGui::SameLine();
        ImGui::RadioButton("CPU", &deviceChoice, 0);
        ImGui::SameLine();
        ImGui::RadioButton("GPU", &deviceChoice, 1);
        ImGui::SameLine();
        ImGui::Checkbox("Show top-left text", &showOverlayText);
        ImGui::SameLine();

        const bool canLoad = !isLoadingModel && !isDetecting;
        if (!canLoad) ImGui::BeginDisabled();
        if (ImGui::Button(isModelLoaded ? "Reload Model" : "Load Model",
                          ImVec2(115, 0))) {
            startLoadingModel();
        }
        if (!canLoad) ImGui::EndDisabled();
        ImGui::SameLine();

        const bool canDetect =
            isModelLoaded && !selectedImages.empty() && !isDetecting &&
            !roiEditing;
        if (!canDetect) ImGui::BeginDisabled();
        if (ImGui::Button("Run Folder Inference", ImVec2(155, 0))) {
            startDetection();
        }
        if (!canDetect) ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::Text("Input: %d image(s)", static_cast<int>(selectedImages.size()));

        const bool roiValid = roi_filter::isValid(roiNormalizedPoints);
        const bool roiEnableDisabled = !roiValid || roiEditing;
        if (roiEnableDisabled) ImGui::BeginDisabled();
        ImGui::Checkbox("Enable ROI", &roiEnabled);
        if (roiEnableDisabled) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Draw / Redraw ROI", ImVec2(145, 0))) {
            roiNormalizedPoints.clear();
            roiEnabled = false;
            roiEditing = true;
            addLog("ROI drawing started: left-click points, right-click undo.");
        }
        ImGui::SameLine();
        const bool finishRoiDisabled = !roiEditing || !roiValid;
        if (finishRoiDisabled) ImGui::BeginDisabled();
        if (ImGui::Button("Finish ROI", ImVec2(95, 0))) {
            roiEditing = false;
            roiEnabled = true;
            addLog("ROI completed with " +
                   std::to_string(roiNormalizedPoints.size()) +
                   " point(s) and enabled.");
        }
        if (finishRoiDisabled) ImGui::EndDisabled();
        ImGui::SameLine();
        const bool clearRoiDisabled = roiNormalizedPoints.empty();
        if (clearRoiDisabled) ImGui::BeginDisabled();
        if (ImGui::Button("Clear ROI", ImVec2(85, 0))) {
            roiNormalizedPoints.clear();
            roiEnabled = false;
            roiEditing = false;
            addLog("ROI cleared; inference will use the full image.");
        }
        if (clearRoiDisabled) ImGui::EndDisabled();
        ImGui::SameLine();
        if (roiEditing) {
            ImGui::TextColored(ImVec4(0.85f, 0.55f, 0.0f, 1.0f),
                               "Drawing: left-click add, right-click undo");
        } else if (roiEnabled && roiValid) {
            ImGui::TextColored(ImVec4(0.0f, 0.60f, 0.15f, 1.0f),
                               "ROI active (%d points)",
                               static_cast<int>(roiNormalizedPoints.size()));
        } else {
            ImGui::TextDisabled("ROI disabled: full image inference");
        }
        if (settingsLocked) ImGui::EndDisabled();
        ImGui::EndChild();

        if (modelPathChanged && isModelLoaded) {
            detector.reset();
            isModelLoaded = false;
            addLog("Model path changed; load the model again.");
        }
        if (imageDirectoryChanged) {
            selectedImages =
                findImagesInDirectory(fs::path(imageDirectoryPath));
            clearReviewItems();
            loadRoiPreview();
        }

        reviewCount = getReviewItemCount();
        const bool hasReview = reviewCount > 0 &&
            displayedReviewIndex != std::numeric_limits<size_t>::max();
        const bool previousReviewDisabled =
            !hasReview || currentReviewIndex == 0;
        if (previousReviewDisabled) ImGui::BeginDisabled();
        if (ImGui::Button("< Previous", ImVec2(105, 0))) {
            displayReviewIndex(currentReviewIndex - 1);
        }
        if (previousReviewDisabled) ImGui::EndDisabled();
        ImGui::SameLine();

        const bool nextReviewDisabled =
            !hasReview || currentReviewIndex + 1 >= reviewCount;
        if (nextReviewDisabled) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Next >", ImVec2(105, 0))) {
            displayReviewIndex(currentReviewIndex + 1);
        }
        if (nextReviewDisabled) {
            ImGui::EndDisabled();
        }
        ImGui::SameLine();
        if (hasReview) {
            ReviewItem currentItem;
            getReviewItem(currentReviewIndex, currentItem);
            ImGui::Text("%d / %d    %s",
                        static_cast<int>(currentReviewIndex + 1),
                        static_cast<int>(reviewCount),
                        fs::path(currentItem.originalPath).filename()
                            .string().c_str());
        } else {
            ImGui::TextDisabled("0 / 0    Run inference to begin review");
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Keyboard: Left / Right / S");

        const float saveButtonWidth = 205.0f;
        ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 10.0f,
            ImGui::GetWindowContentRegionMax().x - saveButtonWidth));
        if (!hasReview) ImGui::BeginDisabled();
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImVec4(0.80f, 0.28f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(0.90f, 0.34f, 0.28f, 1.0f));
        if (ImGui::Button("Save Original Images",
                          ImVec2(saveButtonWidth, 0))) {
            saveCurrentReview();
        }
        ImGui::PopStyleColor(2);
        if (!hasReview) ImGui::EndDisabled();

        const float logHeight = 105.0f;
        const ImVec2 remaining = ImGui::GetContentRegionAvail();
        const float panelHeight = std::max(
            160.0f, remaining.y - logHeight - style.ItemSpacing.y);
        const float panelWidth = std::max(
            160.0f, (remaining.x - style.ItemSpacing.x) * 0.5f);
        const GLuint displayedOriginalTexture =
            hasReview ? originalTexture : roiPreviewTexture;
        const cv::Size displayedOriginalSize =
            hasReview ? originalSize : roiPreviewSize;
        drawReviewImagePanel(
            "OriginalPanel",
            hasReview ? "Original Image" : "ROI Preview (first image)",
            displayedOriginalTexture, displayedOriginalSize,
            ImVec2(panelWidth, panelHeight), &roiNormalizedPoints,
            roiEnabled && roi_filter::isValid(roiNormalizedPoints),
            roiEditing && !settingsLocked);
        ImGui::SameLine();
        drawReviewImagePanel("ResultPanel", "Inference Result",
                             resultTexture, resultSize,
                             ImVec2(panelWidth, panelHeight));

        ImGui::BeginChild("LogsBox", ImVec2(0, 0), true);
        for (const auto& log : logs) ImGui::TextUnformatted(log.c_str());
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
            ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();
        ImGui::End();

        if (showAnalysisWindow) {
            ImGui::SetNextWindowSize(ImVec2(1220, 780),
                                     ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowPos(ImVec2(55, 35),
                                    ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Prediction / Ground Truth Analysis",
                             &showAnalysisWindow,
                             ImGuiWindowFlags_NoCollapse)) {
                ImGui::TextUnformatted(
                    "Class-aware polygon IoU matching: unmatched GT = missed, "
                    "unmatched prediction = false positive");
                if (isAnalyzing) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.1f, 0.4f, 0.9f, 1.0f),
                                       "Analyzing...");
                }
                ImGui::Separator();

                const bool analysisWasRunning = isAnalyzing.load();
                if (analysisWasRunning) ImGui::BeginDisabled();
                auto drawAnalysisPath = [&](const char* label,
                                             const char* inputId,
                                             char* buffer,
                                             size_t bufferSize,
                                             const char* browseId,
                                             const char* browserTitle) {
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted(label);
                    ImGui::SameLine(175);
                    ImGui::SetNextItemWidth(
                        std::max(220.0f,
                            ImGui::GetContentRegionAvail().x - 95.0f));
                    ImGui::InputText(inputId, buffer, bufferSize);
                    ImGui::SameLine();
                    if (ImGui::Button(browseId)) {
                        fileBrowser.open(browserTitle, true, false, "");
                    }
                };
                drawAnalysisPath(
                    "Prediction directory", "##AnalysisPrediction",
                    analysisPredictionDirectory,
                    sizeof(analysisPredictionDirectory),
                    "Browse##AnalysisPrediction",
                    "Select Prediction JSON Directory");
                drawAnalysisPath(
                    "Ground-truth directory", "##AnalysisGroundTruth",
                    analysisGroundTruthDirectory,
                    sizeof(analysisGroundTruthDirectory),
                    "Browse##AnalysisGroundTruth",
                    "Select Ground Truth JSON Directory");
                drawAnalysisPath(
                    "Analysis output", "##AnalysisOutput",
                    analysisOutputDirectory,
                    sizeof(analysisOutputDirectory),
                    "Browse##AnalysisOutput",
                    "Select Analysis Output Directory");

                ImGui::SetNextItemWidth(210);
                ImGui::SliderFloat("Match IoU", &analysisIouThreshold,
                                   0.0f, 1.0f, "%.2f");
                ImGui::SameLine();
                ImGui::Checkbox("Show top-left text in generated images",
                                &showOverlayText);
                ImGui::SameLine();
                const bool analysisPathsReady =
                    analysisPredictionDirectory[0] != '\0' &&
                    analysisGroundTruthDirectory[0] != '\0' &&
                    analysisOutputDirectory[0] != '\0';
                if (!analysisPathsReady) ImGui::BeginDisabled();
                if (ImGui::Button("Run Analysis", ImVec2(120, 0))) {
                    startAnalysis();
                }
                if (!analysisPathsReady) ImGui::EndDisabled();
                if (analysisWasRunning) ImGui::EndDisabled();

                ImGui::SameLine();
                const bool canStopAnalysis = isAnalyzing.load();
                if (!canStopAnalysis) ImGui::BeginDisabled();
                if (ImGui::Button("Stop Analysis", ImVec2(120, 0))) {
                    stopAnalysis();
                }
                if (!canStopAnalysis) ImGui::EndDisabled();

                const auto currentReport = getAnalysisReportSnapshot();
                ImGui::Separator();
                if (analysisHasProgress.load() && analysisTotalProgress.load() > 0) {
                    ImGui::Text(
                        "Progress: %d/%d - %lldms    Processed images: %d    Skipped: %d    "
                        "Report: %s",
                        analysisCurrentProgress.load(),
                        analysisTotalProgress.load(),
                        analysisElapsedMs.load(),
                        currentReport.images, currentReport.skippedImages,
                        currentReport.csvPath.empty()
                            ? (isAnalyzing ? "generating..." : "not generated")
                            : currentReport.csvPath.c_str());
                } else {
                    ImGui::Text(
                        "Progress: 0/0    Processed images: %d    Skipped: %d    "
                        "Report: %s",
                        currentReport.images, currentReport.skippedImages,
                        currentReport.csvPath.empty()
                            ? "not generated" : currentReport.csvPath.c_str());
                }

                const float tableHeight = std::min(
                    190.0f, std::max(100.0f,
                        ImGui::GetContentRegionAvail().y * 0.30f));
                if (ImGui::BeginTable(
                        "AnalysisMetrics", 9,
                        ImGuiTableFlags_Borders |
                        ImGuiTableFlags_RowBg |
                        ImGuiTableFlags_ScrollY |
                        ImGuiTableFlags_Resizable,
                        ImVec2(0, tableHeight))) {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("Class");
                    ImGui::TableSetupColumn("GT");
                    ImGui::TableSetupColumn("PRED");
                    ImGui::TableSetupColumn("TP");
                    ImGui::TableSetupColumn("Missed");
                    ImGui::TableSetupColumn("False Pos");
                    ImGui::TableSetupColumn("Precision");
                    ImGui::TableSetupColumn("Recall");
                    ImGui::TableSetupColumn("F1");
                    ImGui::TableHeadersRow();
                    auto drawMetricsRow = [](
                        const segmentation_analysis::ClassMetrics& metrics) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::TextUnformatted(metrics.label.c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%d", metrics.groundTruth);
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%d", metrics.predictions);
                        ImGui::TableSetColumnIndex(3);
                        ImGui::Text("%d", metrics.truePositive);
                        ImGui::TableSetColumnIndex(4);
                        ImGui::Text("%d", metrics.falseNegative);
                        ImGui::TableSetColumnIndex(5);
                        ImGui::Text("%d", metrics.falsePositive);
                        ImGui::TableSetColumnIndex(6);
                        ImGui::Text("%.3f", metrics.precision());
                        ImGui::TableSetColumnIndex(7);
                        ImGui::Text("%.3f", metrics.recall());
                        ImGui::TableSetColumnIndex(8);
                        ImGui::Text("%.3f", metrics.f1());
                    };
                    for (const auto& metrics : currentReport.classes) {
                        drawMetricsRow(metrics);
                    }
                    if (!currentReport.classes.empty() ||
                        currentReport.images > 0) {
                        drawMetricsRow(currentReport.total);
                    }
                    ImGui::EndTable();
                }

                const size_t analysisCount =
                    currentReport.reviewItems.size();
                const bool hasAnalysisImage =
                    analysisCount > 0 && displayedAnalysisIndex !=
                        std::numeric_limits<size_t>::max();
                const bool previousAnalysisDisabled =
                    !hasAnalysisImage || currentAnalysisIndex == 0;
                if (previousAnalysisDisabled) {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Button("< Previous##Analysis",
                                  ImVec2(120, 0))) {
                    if (loadAnalysisTextures(
                            currentAnalysisIndex - 1,
                            analysisOriginalTexture,
                            analysisVisualizationTexture,
                            analysisOriginalSize,
                            analysisVisualizationSize)) {
                        --currentAnalysisIndex;
                        displayedAnalysisIndex = currentAnalysisIndex;
                    }
                }
                if (previousAnalysisDisabled) {
                    ImGui::EndDisabled();
                }
                ImGui::SameLine();
                const bool nextAnalysisDisabled =
                    !hasAnalysisImage ||
                    currentAnalysisIndex + 1 >= analysisCount;
                if (nextAnalysisDisabled) {
                    ImGui::BeginDisabled();
                }
                if (ImGui::Button("Next >##Analysis", ImVec2(120, 0))) {
                    if (loadAnalysisTextures(
                            currentAnalysisIndex + 1,
                            analysisOriginalTexture,
                            analysisVisualizationTexture,
                            analysisOriginalSize,
                            analysisVisualizationSize)) {
                        ++currentAnalysisIndex;
                        displayedAnalysisIndex = currentAnalysisIndex;
                    }
                }
                if (nextAnalysisDisabled) {
                    ImGui::EndDisabled();
                }
                ImGui::SameLine();
                if (hasAnalysisImage) {
                    ImGui::Text("%d / %d    %s",
                        static_cast<int>(currentAnalysisIndex + 1),
                        static_cast<int>(analysisCount),
                        fs::path(currentReport.reviewItems[
                            currentAnalysisIndex].visualizationPath)
                            .filename().string().c_str());
                } else {
                    ImGui::TextDisabled(
                        "0 / 0    Run analysis to generate comparisons");
                }

                const ImVec2 analysisRemaining =
                    ImGui::GetContentRegionAvail();
                const float analysisPanelWidth = std::max(
                    180.0f,
                    (analysisRemaining.x - style.ItemSpacing.x) * 0.5f);
                drawReviewImagePanel(
                    "AnalysisOriginalPanel", "Ground Truth Source",
                    analysisOriginalTexture, analysisOriginalSize,
                    ImVec2(analysisPanelWidth, analysisRemaining.y));
                ImGui::SameLine();
                drawReviewImagePanel(
                    "AnalysisResultPanel", "IoU Analysis Comparison",
                    analysisVisualizationTexture,
                    analysisVisualizationSize,
                    ImVec2(analysisPanelWidth, analysisRemaining.y));
            }
            ImGui::End();
        }

        if (fileBrowser.show()) {
            if (isDetecting || isLoadingModel) {
                addLog("Path selection ignored while a background task is running.");
            } else if (fileBrowser.title == "Select ONNX Model") {
                copyPathArgument(modelPath, sizeof(modelPath),
                                 fileBrowser.selectedPath);
                isModelLoaded = false;
                detector.reset();
                addLog("Selected model: " + std::string(modelPath));
            } else if (fileBrowser.title == "Select Image Directory") {
                copyPathArgument(imageDirectoryPath,
                                 sizeof(imageDirectoryPath),
                                 fileBrowser.selectedPath);
                selectedImages =
                    findImagesInDirectory(fs::path(imageDirectoryPath));
                clearReviewItems();
                loadRoiPreview();
                addLog("Selected image directory: " +
                       std::string(imageDirectoryPath) + " (" +
                       std::to_string(selectedImages.size()) + " image(s))");
            } else if (fileBrowser.title == "Select Result Directory") {
                copyPathArgument(resultPath, sizeof(resultPath),
                                 fileBrowser.selectedPath);
                addLog("Selected result directory: " +
                       std::string(resultPath));
            } else if (fileBrowser.title ==
                       "Select Original Directory") {
                copyPathArgument(originalDirectoryPath,
                                 sizeof(originalDirectoryPath),
                                 fileBrowser.selectedPath);
                addLog("Selected original directory: " +
                       std::string(originalDirectoryPath));
            } else if (fileBrowser.title ==
                       "Select Prediction JSON Directory") {
                copyPathArgument(analysisPredictionDirectory,
                                 sizeof(analysisPredictionDirectory),
                                 fileBrowser.selectedPath);
                addLog("Selected prediction JSON directory: " +
                       std::string(analysisPredictionDirectory));
            } else if (fileBrowser.title ==
                       "Select Ground Truth JSON Directory") {
                copyPathArgument(analysisGroundTruthDirectory,
                                 sizeof(analysisGroundTruthDirectory),
                                 fileBrowser.selectedPath);
                addLog("Selected ground-truth JSON directory: " +
                       std::string(analysisGroundTruthDirectory));
            } else if (fileBrowser.title ==
                       "Select Analysis Output Directory") {
                copyPathArgument(analysisOutputDirectory,
                                 sizeof(analysisOutputDirectory),
                                 fileBrowser.selectedPath);
                addLog("Selected analysis output directory: " +
                       std::string(analysisOutputDirectory));
            }
        }

        ImGui::Render();
        int displayWidth = 0;
        int displayHeight = 0;
        glfwGetFramebufferSize(window, &displayWidth, &displayHeight);
        glViewport(0, 0, displayWidth, displayHeight);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    saveConfig();
    shouldExit = true;
    shouldStopAnalysis = true;
    if (detectionThread.joinable()) detectionThread.join();
    if (loadModelThread.joinable()) loadModelThread.join();
    if (analysisThread.joinable()) analysisThread.join();
    if (detector != nullptr) {
        addLog("Destroying ONNX Runtime session before graphics shutdown.");
        detector.reset();
    }
    releaseTexture(originalTexture);
    releaseTexture(resultTexture);
    releaseTexture(roiPreviewTexture);
    releaseTexture(analysisOriginalTexture);
    releaseTexture(analysisVisualizationTexture);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
