#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <string>
#include <vector>

#include "Utils.hpp"

#if __cplusplus < 201402L
namespace std {
  template<typename T, typename... Args>
  unique_ptr<T> make_unique(Args&&... args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
  }
}
#endif

// Stable BGR colors shared by contours, numeric tags, and legend entries.
// Pure green is intentionally excluded because it is reserved for GT.
inline cv::Scalar predictionColorForClass(int classId) {
  static const cv::Scalar palette[] = {
      cv::Scalar(255, 80, 0),    // blue
      cv::Scalar(0, 64, 255),    // red
      cv::Scalar(220, 0, 220),   // magenta
      cv::Scalar(0, 150, 255),   // orange
      cv::Scalar(255, 220, 0),   // cyan
      cv::Scalar(0, 220, 220),   // yellow
      cv::Scalar(180, 60, 180),  // purple
      cv::Scalar(150, 80, 255)   // pink
  };
  const int colorCount =
      static_cast<int>(sizeof(palette) / sizeof(palette[0]));
  int colorIndex = classId % colorCount;
  if (colorIndex < 0) colorIndex += colorCount;
  return palette[colorIndex];
}

/**
 * @brief Abstract Base Class for YOLO Segmentation Detectors (YOLO11, YOLO26, etc.)
 */
class IYOLOSegDetector {
public:
  virtual ~IYOLOSegDetector() = default;

  // Main API
  virtual std::vector<SegmentedDetection> segment(const cv::Mat &image,
                                                  float confThreshold = 0.40f,
                                                  float iouThreshold = 0.45f) = 0;

  // Drawing results
  virtual void drawSegmentationsAndBoxes(cv::Mat &image,
                                         const std::vector<SegmentedDetection> &results,
                                         float maskAlpha = 0.5f) const;

  virtual void drawSegmentations(cv::Mat &image,
                                 const std::vector<SegmentedDetection> &results,
                                 float maskAlpha = 0.5f) const;

  // Accessors
  virtual const std::vector<std::string> &getClassNames() const = 0;
  virtual const std::vector<cv::Scalar> &getClassColors() const = 0;

  virtual void printSegmentationResults(const std::vector<SegmentedDetection>& results) const {
    std::cout << "Segmentation Results (" << results.size() << " objects):\n";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& det = results[i];
      std::cout << "  Object " << i+1 << ":\n";
      std::cout << "    Class ID: " << det.classId << "\n";
      std::cout << "    Confidence: " << det.conf << "\n";
      std::cout << "    Bounding Box: [x:" << det.box.x << " y:" << det.box.y
                << " w:" << det.box.width << " h:" << det.box.height << "]\n";
      std::cout << "    Mask: " << det.mask.cols << "x" << det.mask.rows
                << " (area: " << cv::countNonZero(det.mask) << " pixels)\n";
    }
  }
};

inline void IYOLOSegDetector::drawSegmentationsAndBoxes(
    cv::Mat &image, const std::vector<SegmentedDetection> &results,
    float maskAlpha) const {
  (void)maskAlpha;
  for (const auto &seg : results) {
    if (seg.mask.empty()) continue;

    cv::Mat maskGray;
    if (seg.mask.channels() == 3) {
      cv::cvtColor(seg.mask, maskGray, cv::COLOR_BGR2GRAY);
    } else {
      maskGray = seg.mask;
    }
    cv::Mat maskBinary;
    cv::threshold(maskGray, maskBinary, 127, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskBinary, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) continue;
    cv::drawContours(image, contours, -1,
                     predictionColorForClass(seg.classId), 2, cv::LINE_AA);
  }
}

inline void IYOLOSegDetector::drawSegmentations(
    cv::Mat &image, const std::vector<SegmentedDetection> &results,
    float maskAlpha) const {
  for (const auto &seg : results) {
    const cv::Scalar color = predictionColorForClass(seg.classId);

    if (!seg.mask.empty()) {
      cv::Mat mask_gray;
      if (seg.mask.channels() == 3) {
        cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY);
      } else {
        mask_gray = seg.mask.clone();
      }

      cv::Mat mask_binary;
      cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

      cv::Mat colored_mask;
      cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
      colored_mask.setTo(color, mask_binary);

      cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
    }
  }
}
