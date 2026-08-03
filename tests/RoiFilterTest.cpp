#include "RoiFilter.hpp"

#include <iostream>

namespace {

SegmentedDetection makeDetection(const cv::Rect& rectangle, bool withMask) {
    SegmentedDetection detection;
    detection.box = SegBoundingBox(rectangle.x, rectangle.y,
                                   rectangle.width, rectangle.height);
    detection.conf = 0.9f;
    detection.classId = 1;
    if (withMask) {
        detection.mask = cv::Mat::zeros(100, 100, CV_8UC1);
        detection.mask(rectangle).setTo(255);
    }
    return detection;
}

}  // namespace

int main() {
    int failures = 0;
    const auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAILED: " << message << '\n';
            ++failures;
        }
    };
    const roi_filter::Polygon leftHalf = {
        {0.0f, 0.0f}, {0.49f, 0.0f}, {0.49f, 1.0f}, {0.0f, 1.0f}};
    expect(roi_filter::isValid(leftHalf), "four-point ROI is valid");

    const std::vector<SegmentedDetection> detections = {
        makeDetection(cv::Rect(10, 10, 20, 20), true),
        makeDetection(cv::Rect(70, 10, 20, 20), true),
        makeDetection(cv::Rect(40, 40, 20, 20), true),
        makeDetection(cv::Rect(20, 60, 10, 10), false),
        makeDetection(cv::Rect(80, 60, 10, 10), false)};
    const auto filtered =
        roi_filter::filter(detections, cv::Size(100, 100), leftHalf);
    expect(filtered.size() == 3,
           "outside mask and box detections removed while overlap is kept");
    if (filtered.size() == 3) {
        expect(filtered[0].box.x == 10 && filtered[0].box.width == 20,
               "fully inside mask keeps its box");
        expect(filtered[1].box.x == 40 && filtered[1].box.width == 10,
               "overlapping mask box is recomputed after clipping");
        expect(cv::countNonZero(filtered[1].mask) == 200,
               "overlapping mask is clipped to ROI");
        expect(filtered[2].box.x == 20,
               "empty-mask detection uses box center");
    }

    const roi_filter::Polygon invalid = {{0.0f, 0.0f}, {1.0f, 1.0f}};
    expect(!roi_filter::isValid(invalid), "two-point ROI is invalid");
    expect(roi_filter::filter(detections, cv::Size(100, 100), invalid).size()
               == detections.size(),
           "invalid ROI does not filter detections");

    if (failures == 0) std::cout << "ROI filter tests passed\n";
    return failures == 0 ? 0 : 1;
}
