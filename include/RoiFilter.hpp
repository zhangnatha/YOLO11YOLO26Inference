#pragma once

#include "Utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace roi_filter {

using Polygon = std::vector<cv::Point2f>;

inline float clampNormalized(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

inline bool isValid(const Polygon& polygon) {
    return polygon.size() >= 3 && std::fabs(cv::contourArea(polygon)) > 1e-6;
}

inline Polygon toImageCoordinates(const Polygon& normalizedPolygon,
                                  const cv::Size& imageSize) {
    Polygon imagePolygon;
    if (imageSize.width <= 0 || imageSize.height <= 0) return imagePolygon;
    imagePolygon.reserve(normalizedPolygon.size());
    const float maxX = static_cast<float>(std::max(0, imageSize.width - 1));
    const float maxY = static_cast<float>(std::max(0, imageSize.height - 1));
    for (const auto& point : normalizedPolygon) {
        imagePolygon.emplace_back(clampNormalized(point.x) * maxX,
                                  clampNormalized(point.y) * maxY);
    }
    return imagePolygon;
}

inline cv::Mat detectionMask(const SegmentedDetection& detection,
                             const cv::Size& imageSize) {
    if (imageSize.width <= 0 || imageSize.height <= 0) return cv::Mat();
    cv::Mat mask;
    if (!detection.mask.empty()) {
        if (detection.mask.channels() == 1) {
            mask = detection.mask;
        } else {
            cv::cvtColor(detection.mask, mask, cv::COLOR_BGR2GRAY);
        }
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        if (mask.size() != imageSize) {
            cv::resize(mask, mask, imageSize, 0.0, 0.0,
                       cv::INTER_NEAREST);
        }
        return mask;
    }

    mask = cv::Mat::zeros(imageSize, CV_8UC1);
    const cv::Rect imageRect(0, 0, imageSize.width, imageSize.height);
    const cv::Rect box(detection.box.x, detection.box.y,
                       detection.box.width, detection.box.height);
    const cv::Rect clippedBox = box & imageRect;
    if (clippedBox.width > 0 && clippedBox.height > 0) {
        mask(clippedBox).setTo(255);
    }
    return mask;
}

inline std::vector<SegmentedDetection> filter(
    const std::vector<SegmentedDetection>& detections,
    const cv::Size& imageSize,
    const Polygon& normalizedPolygon) {
    if (!isValid(normalizedPolygon)) return detections;
    const Polygon imagePolygon = toImageCoordinates(normalizedPolygon,
                                                    imageSize);
    std::vector<cv::Point> integerPolygon;
    integerPolygon.reserve(imagePolygon.size());
    for (const auto& point : imagePolygon) {
        integerPolygon.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    cv::Mat roiMask = cv::Mat::zeros(imageSize, CV_8UC1);
    cv::fillPoly(roiMask, std::vector<std::vector<cv::Point>>{
        integerPolygon}, cv::Scalar(255));

    std::vector<SegmentedDetection> filtered;
    filtered.reserve(detections.size());
    for (const auto& detection : detections) {
        const cv::Mat sourceMask = detectionMask(detection, imageSize);
        if (sourceMask.empty()) continue;
        cv::Mat clippedMask;
        cv::bitwise_and(sourceMask, roiMask, clippedMask);
        if (cv::countNonZero(clippedMask) == 0) continue;

        SegmentedDetection clipped = detection;
        const cv::Rect clippedBox = cv::boundingRect(clippedMask);
        clipped.box = SegBoundingBox(clippedBox.x, clippedBox.y,
                                     clippedBox.width, clippedBox.height);
        if (!detection.mask.empty()) {
            // Keep only mask pixels inside the ROI. This also guarantees that
            // downstream contours and prediction JSON cannot leak outside it.
            clipped.mask = clippedMask;
        }
        filtered.push_back(std::move(clipped));
    }
    return filtered;
}

}  // namespace roi_filter
