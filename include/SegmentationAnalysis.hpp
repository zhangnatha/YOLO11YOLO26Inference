#pragma once

#include <functional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace segmentation_analysis {

struct Prediction {
    float score = 0.0f;
    int classId = 0;
    cv::Mat mask;
};

struct ClassMetrics {
    std::string label;
    int groundTruth = 0;
    int predictions = 0;
    int truePositive = 0;
    int falseNegative = 0;
    int falsePositive = 0;

    double precision() const;
    double recall() const;
    double f1() const;
};

struct ReviewItem {
    std::string originalPath;
    std::string visualizationPath;
    long long elapsedMs = 0;
};

struct Report {
    int images = 0;
    int skippedImages = 0;
    std::vector<ClassMetrics> classes;
    ClassMetrics total;
    std::vector<ReviewItem> reviewItems;
    std::string csvPath;
    std::string jsonPath;
};

bool writePredictionJson(
    const std::string& jsonPath,
    const std::string& imageFileName,
    const cv::Size& imageSize,
    const std::vector<Prediction>& predictions,
    const std::vector<std::string>& classNames,
    std::string* errorMessage = nullptr);

Report analyzeDirectories(
    const std::string& predictionDirectory,
    const std::string& groundTruthDirectory,
    const std::string& outputDirectory,
    float iouThreshold,
    bool showOverlayText,
    const std::function<void(const std::string&)>& logCallback = {},
    const std::function<void(int current, int total, long long elapsedMs)>& progressCallback = {},
    const std::function<bool()>& cancelCallback = {});

}  // namespace segmentation_analysis
