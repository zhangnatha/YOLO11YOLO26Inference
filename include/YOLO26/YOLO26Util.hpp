#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <set>
#include <random>
#include <opencv2/opencv.hpp>

#include "../Utils.hpp"

namespace yolo26 {

using ::ClassificationResult;
using ::BoundingBox;
using ::Detection;
using ::SegBoundingBox;
using ::SegmentedDetection;

namespace utils {
    using ::utils::clamp;
    using ::utils::getClassNames;
    using ::utils::vectorProduct;
    using ::utils::SegLetterBox;
    using ::utils::SegNMSBoxes;
    using ::utils::sigmoid;
    using ::utils::scaleCoords;
    using ::utils::generateColors;
} // namespace utils

} // namespace yolo26
