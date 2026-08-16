#include "SegmentationAnalysis.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#if defined(__has_include)
  #if __has_include(<filesystem>) && __cplusplus >= 201703L && \
      !(defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8)
    #include <filesystem>
    namespace analysis_fs = std::filesystem;
  #elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace analysis_fs = std::experimental::filesystem;
  #else
    #error "Compiling requires filesystem support"
  #endif
#else
  #include <experimental/filesystem>
  namespace analysis_fs = std::experimental::filesystem;
#endif

namespace segmentation_analysis {
namespace {

struct Annotation {
    std::string label;
    float score = -1.0f;
    std::vector<cv::Point> points;
    std::vector<std::vector<cv::Point>> extraContours;
};

std::string jsonEscape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string escaped = "\"";
    for (const char character : value) {
        if (character == '"') escaped += '"';
        escaped += character;
    }
    escaped += '"';
    return escaped;
}

std::string lowerExtension(const analysis_fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension;
}

bool isImage(const analysis_fs::path& path) {
    if (!analysis_fs::is_regular_file(path)) return false;
    const std::string extension = lowerExtension(path);
    return extension == ".png" || extension == ".jpg" ||
           extension == ".jpeg" || extension == ".bmp" ||
           extension == ".tif" || extension == ".tiff";
}

std::string normalizedStem(const analysis_fs::path& path,
                           bool predictionSide) {
    std::string stem = path.stem().string();
    if (predictionSide && stem.size() > 2 &&
        stem.compare(stem.size() - 2, 2, "_r") == 0) {
        stem.resize(stem.size() - 2);
    }
    return stem;
}

std::map<std::string, analysis_fs::path> collectJsonFiles(
    const analysis_fs::path& directory, bool predictionSide) {
    std::map<std::string, analysis_fs::path> result;
    if (!analysis_fs::exists(directory) ||
        !analysis_fs::is_directory(directory)) {
        return result;
    }
    for (const auto& entry : analysis_fs::directory_iterator(directory)) {
        if (!analysis_fs::is_regular_file(entry.path()) ||
            lowerExtension(entry.path()) != ".json" ||
            (predictionSide && entry.path().filename() == "predictions.json")) {
            continue;
        }
        result[normalizedStem(entry.path(), predictionSide)] = entry.path();
    }
    return result;
}

analysis_fs::path findImage(const analysis_fs::path& directory,
                            const std::string& stem,
                            bool predictionSide) {
    if (!analysis_fs::exists(directory) ||
        !analysis_fs::is_directory(directory)) {
        return {};
    }
    const std::string resultStem = predictionSide ? stem + "_r" : stem;
    for (const auto& entry : analysis_fs::directory_iterator(directory)) {
        if (isImage(entry.path()) && entry.path().stem().string() == resultStem) {
            return entry.path();
        }
    }
    return {};
}

std::vector<Annotation> loadAnnotations(const analysis_fs::path& jsonPath,
                                        const cv::Size& imageSize,
                                        cv::Size* declaredSize = nullptr) {
    if (jsonPath.empty() || !analysis_fs::exists(jsonPath)) return {};
    const YAML::Node root = YAML::LoadFile(jsonPath.string());
    if (declaredSize) {
        const int width = root["imageWidth"]
            ? root["imageWidth"].as<int>() : imageSize.width;
        const int height = root["imageHeight"]
            ? root["imageHeight"].as<int>() : imageSize.height;
        *declaredSize = cv::Size(width, height);
    }
    const YAML::Node shapes = root["shapes"];
    std::vector<Annotation> annotations;
    if ((!shapes || !shapes.IsSequence()) && root["detections"] &&
        root["detections"].IsSequence()) {
        for (const auto& detection : root["detections"]) {
            Annotation annotation;
            annotation.label = detection["class_name"]
                ? detection["class_name"].as<std::string>() : "?";
            if (detection["score"] && !detection["score"].IsNull())
                annotation.score = detection["score"].as<float>();
            const YAML::Node contours = detection["contours_xy"];
            if (contours && contours.IsSequence()) {
                for (const auto& contour : contours) {
                    std::vector<cv::Point> points;
                    if (!contour.IsSequence()) continue;
                    for (const auto& pointNode : contour) {
                        if (!pointNode.IsSequence() || pointNode.size() < 2) continue;
                        int x = cvRound(pointNode[0].as<double>());
                        int y = cvRound(pointNode[1].as<double>());
                        if (imageSize.width > 0 && imageSize.height > 0) {
                            x = std::max(0, std::min(imageSize.width - 1, x));
                            y = std::max(0, std::min(imageSize.height - 1, y));
                        }
                        points.emplace_back(x, y);
                    }
                    if (points.size() >= 3) {
                        if (annotation.points.empty()) annotation.points = std::move(points);
                        else annotation.extraContours.push_back(std::move(points));
                    }
                }
            }
            if (annotation.points.size() >= 3) annotations.push_back(std::move(annotation));
        }
        return annotations;
    }
    if (!shapes || !shapes.IsSequence()) return annotations;
    for (const auto& shape : shapes) {
        const YAML::Node pointNodes = shape["points"];
        if (!pointNodes || !pointNodes.IsSequence()) continue;
        Annotation annotation;
        annotation.label = shape["label"]
            ? shape["label"].as<std::string>() : "unknown";
        if (shape["score"] && !shape["score"].IsNull()) {
            annotation.score = shape["score"].as<float>();
        }
        for (const auto& pointNode : pointNodes) {
            if (!pointNode.IsSequence() || pointNode.size() < 2) continue;
            int x = cvRound(pointNode[0].as<double>());
            int y = cvRound(pointNode[1].as<double>());
            if (imageSize.width > 0 && imageSize.height > 0) {
                x = std::max(0, std::min(imageSize.width - 1, x));
                y = std::max(0, std::min(imageSize.height - 1, y));
            }
            annotation.points.emplace_back(x, y);
        }
        const std::string shapeType = shape["shape_type"]
            ? shape["shape_type"].as<std::string>() : "polygon";
        if (shapeType == "rectangle" && annotation.points.size() >= 2) {
            const cv::Point first = annotation.points[0];
            const cv::Point second = annotation.points[1];
            annotation.points = {
                {first.x, first.y}, {second.x, first.y},
                {second.x, second.y}, {first.x, second.y}};
        }
        if (annotation.points.size() >= 3) {
            annotations.push_back(std::move(annotation));
        }
    }
    return annotations;
}

double polygonIou(const Annotation& first, const Annotation& second,
                  const cv::Size& imageSize) {
    if (first.points.size() < 3 || second.points.size() < 3) return 0.0;
    cv::Rect bounds = cv::boundingRect(first.points) |
                      cv::boundingRect(second.points);
    for (const auto& contour : first.extraContours) bounds |= cv::boundingRect(contour);
    for (const auto& contour : second.extraContours) bounds |= cv::boundingRect(contour);
    bounds &= cv::Rect(0, 0, imageSize.width, imageSize.height);
    if (bounds.empty()) return 0.0;

    auto shifted = [&](const std::vector<cv::Point>& input) {
        std::vector<cv::Point> output;
        output.reserve(input.size());
        for (const auto& point : input) {
            output.emplace_back(point.x - bounds.x, point.y - bounds.y);
        }
        return output;
    };
    cv::Mat firstMask(bounds.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat secondMask(bounds.size(), CV_8UC1, cv::Scalar(0));
    std::vector<std::vector<cv::Point>> firstPolygon{shifted(first.points)};
    std::vector<std::vector<cv::Point>> secondPolygon{shifted(second.points)};
    for (const auto& contour : first.extraContours) firstPolygon.push_back(shifted(contour));
    for (const auto& contour : second.extraContours) secondPolygon.push_back(shifted(contour));
    cv::fillPoly(firstMask, firstPolygon, cv::Scalar(255));
    cv::fillPoly(secondMask, secondPolygon, cv::Scalar(255));
    cv::Mat intersection;
    cv::Mat unionMask;
    cv::bitwise_and(firstMask, secondMask, intersection);
    cv::bitwise_or(firstMask, secondMask, unionMask);
    const int unionArea = cv::countNonZero(unionMask);
    return unionArea > 0
        ? static_cast<double>(cv::countNonZero(intersection)) / unionArea
        : 0.0;
}

void drawContour(cv::Mat& image, const Annotation& annotation,
                 const cv::Scalar& color, int thickness) {
    std::vector<std::vector<cv::Point>> contours{annotation.points};
    contours.insert(contours.end(), annotation.extraContours.begin(),
                    annotation.extraContours.end());
    cv::drawContours(image, contours, -1, color, thickness, cv::LINE_AA);
}

void drawTag(cv::Mat& image, const Annotation& annotation,
             const std::string& text, const cv::Scalar& color) {
    if (annotation.points.empty()) return;
    cv::Rect bounds = cv::boundingRect(annotation.points);
    for (const auto& contour : annotation.extraContours) bounds |= cv::boundingRect(contour);
    int baseline = 0;
    const cv::Size textSize = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
    const int width = textSize.width + 8;
    const int height = textSize.height + 8;
    const int x = std::max(0, std::min(bounds.x, image.cols - width));
    const int y = std::max(0, std::min(bounds.y - height, image.rows - height));
    cv::rectangle(image, cv::Rect(x, y, width, height), color, cv::FILLED);
    cv::putText(image, text, cv::Point(x + 4, y + textSize.height + 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(255, 255, 255),
                1, cv::LINE_AA);
}

void drawLegend(cv::Mat& image, const std::vector<std::string>& lines,
                const std::vector<cv::Scalar>& colors) {
    if (image.empty() || lines.empty()) return;
    const int margin = 8;
    const int padding = 8;
    const int lineHeight = 19;
    const int swatch = 11;
    const int maxRows = std::max(1,
        (image.rows - 2 * margin - 2 * padding) / lineHeight);
    const int rows = std::min(static_cast<int>(lines.size()), maxRows);
    int maxWidth = 0;
    for (int index = 0; index < rows; ++index) {
        maxWidth = std::max(maxWidth, cv::getTextSize(
            lines[static_cast<size_t>(index)], cv::FONT_HERSHEY_SIMPLEX,
            0.43, 1, nullptr).width);
    }
    const int panelWidth = std::min(
        image.cols - 2 * margin, 3 * padding + swatch + maxWidth);
    const int panelHeight = std::min(
        image.rows - 2 * margin, 2 * padding + rows * lineHeight);
    if (panelWidth <= 0 || panelHeight <= 0) return;
    const cv::Rect panel(margin, margin, panelWidth, panelHeight);
    cv::Mat roi = image(panel);
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.25, dark, 0.75, 0.0, roi);
    for (int index = 0; index < rows; ++index) {
        const int centerY = margin + padding + index * lineHeight + lineHeight / 2;
        const cv::Scalar color = colors[static_cast<size_t>(index)];
        cv::rectangle(image, cv::Point(margin + padding, centerY - swatch / 2),
                      cv::Point(margin + padding + swatch, centerY + swatch / 2),
                      color, cv::FILLED);
        cv::putText(image, lines[static_cast<size_t>(index)],
                    cv::Point(margin + 2 * padding + swatch, centerY + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.43,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

void addMetrics(std::map<std::string, ClassMetrics>& metrics,
                const std::string& label, int groundTruth, int predictions,
                int truePositive, int falseNegative, int falsePositive) {
    auto& item = metrics[label];
    item.label = label;
    item.groundTruth += groundTruth;
    item.predictions += predictions;
    item.truePositive += truePositive;
    item.falseNegative += falseNegative;
    item.falsePositive += falsePositive;
}

void writeReports(const analysis_fs::path& outputDirectory,
                  float iouThreshold, Report& report) {
    report.csvPath = (outputDirectory / "analysis_report.csv").string();
    report.jsonPath = (outputDirectory / "analysis_report.json").string();

    std::ofstream csv(report.csvPath, std::ios::trunc);
    if (!csv.is_open()) throw std::runtime_error("Cannot write " + report.csvPath);
    csv << "class,ground_truth,predictions,true_positive,missed,false_positive,precision,recall,f1\n";
    auto writeCsvRow = [&](const ClassMetrics& item) {
        csv << csvEscape(item.label) << ',' << item.groundTruth << ','
            << item.predictions << ',' << item.truePositive << ','
            << item.falseNegative << ',' << item.falsePositive << ','
            << std::fixed << std::setprecision(6) << item.precision() << ','
            << item.recall() << ',' << item.f1() << '\n';
    };
    for (const auto& item : report.classes) writeCsvRow(item);
    writeCsvRow(report.total);

    std::ofstream json(report.jsonPath, std::ios::trunc);
    if (!json.is_open()) throw std::runtime_error("Cannot write " + report.jsonPath);
    json << "{\n  \"iouThreshold\": " << std::fixed << std::setprecision(4)
         << iouThreshold << ",\n  \"images\": " << report.images
         << ",\n  \"skippedImages\": " << report.skippedImages
         << ",\n  \"classes\": [\n";
    auto writeJsonItem = [&](const ClassMetrics& item, const std::string& indent) {
        json << indent << "{\"label\": \"" << jsonEscape(item.label)
             << "\", \"groundTruth\": " << item.groundTruth
             << ", \"predictions\": " << item.predictions
             << ", \"truePositive\": " << item.truePositive
             << ", \"missed\": " << item.falseNegative
             << ", \"falsePositive\": " << item.falsePositive
             << ", \"precision\": " << std::setprecision(6) << item.precision()
             << ", \"recall\": " << item.recall()
             << ", \"f1\": " << item.f1() << '}';
    };
    for (size_t index = 0; index < report.classes.size(); ++index) {
        writeJsonItem(report.classes[index], "    ");
        json << (index + 1 < report.classes.size() ? ",\n" : "\n");
    }
    json << "  ],\n  \"total\": ";
    writeJsonItem(report.total, "");
    json << "\n}\n";
}

}  // namespace

double ClassMetrics::precision() const {
    const int denominator = truePositive + falsePositive;
    return denominator > 0 ? static_cast<double>(truePositive) / denominator : 0.0;
}

double ClassMetrics::recall() const {
    const int denominator = truePositive + falseNegative;
    return denominator > 0 ? static_cast<double>(truePositive) / denominator : 0.0;
}

double ClassMetrics::f1() const {
    const double precisionValue = precision();
    const double recallValue = recall();
    const double denominator = precisionValue + recallValue;
    return denominator > 0.0
        ? 2.0 * precisionValue * recallValue / denominator : 0.0;
}

bool writePredictionJson(
    const std::string& jsonPath,
    const std::string& imageFileName,
    const cv::Size& imageSize,
    const std::vector<Prediction>& predictions,
    const std::vector<std::string>& classNames,
    std::string* errorMessage) {
    try {
        std::ofstream output(jsonPath, std::ios::trunc);
        if (!output.is_open()) throw std::runtime_error("Cannot open output file");
        output << "{\n  \"version\": \"4.0.0-beta.7\",\n"
               << "  \"flags\": {},\n  \"checked\": false,\n"
               << "  \"shapes\": [";
        bool firstShape = true;
        for (const auto& prediction : predictions) {
            if (prediction.mask.empty()) continue;
            cv::Mat mask;
            if (prediction.mask.channels() == 1) {
                mask = prediction.mask.clone();
            } else {
                cv::cvtColor(prediction.mask, mask, cv::COLOR_BGR2GRAY);
            }
            if (mask.size() != imageSize) {
                cv::resize(mask, mask, imageSize, 0.0, 0.0, cv::INTER_NEAREST);
            }
            cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                             cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty()) continue;
            const auto largest = std::max_element(
                contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& left,
                   const std::vector<cv::Point>& right) {
                    return std::fabs(cv::contourArea(left)) <
                           std::fabs(cv::contourArea(right));
                });
            std::vector<cv::Point> polygon;
            cv::approxPolyDP(*largest, polygon, 1.5, true);
            if (polygon.size() < 3) polygon = *largest;
            if (polygon.size() < 3) continue;

            const bool knownClass = prediction.classId >= 0 &&
                static_cast<size_t>(prediction.classId) < classNames.size();
            const std::string label = knownClass
                ? classNames[static_cast<size_t>(prediction.classId)]
                : "class_" + std::to_string(prediction.classId);
            if (!firstShape) output << ',';
            firstShape = false;
            output << "\n    {\n      \"label\": \"" << jsonEscape(label)
                   << "\",\n      \"score\": " << std::fixed
                   << std::setprecision(6) << prediction.score
                   << ",\n      \"points\": [";
            for (size_t index = 0; index < polygon.size(); ++index) {
                if (index > 0) output << ',';
                output << "\n        [" << polygon[index].x << ", "
                       << polygon[index].y << ']';
            }
            output << "\n      ],\n      \"group_id\": null,"
                   << "\n      \"description\": \"prediction\","
                   << "\n      \"difficult\": false,"
                   << "\n      \"shape_type\": \"polygon\","
                   << "\n      \"flags\": {},\n      \"attributes\": {},"
                   << "\n      \"kie_linking\": []\n    }";
        }
        output << "\n  ],\n  \"imagePath\": \""
               << jsonEscape(imageFileName) << "\",\n"
               << "  \"imageData\": null,\n  \"imageHeight\": "
               << imageSize.height << ",\n  \"imageWidth\": "
               << imageSize.width << "\n}\n";
        if (!output.good()) throw std::runtime_error("Write failed");
        return true;
    } catch (const std::exception& error) {
        if (errorMessage) *errorMessage = error.what();
        return false;
    }
}

Report analyzeDirectories(
    const std::string& predictionDirectory,
    const std::string& groundTruthDirectory,
    const std::string& outputDirectory,
    float iouThreshold,
    bool showOverlayText,
    const std::function<void(const std::string&)>& logCallback) {
    const analysis_fs::path predictionRoot(predictionDirectory);
    const analysis_fs::path groundTruthRoot(groundTruthDirectory);
    const analysis_fs::path outputRoot(outputDirectory);
    if (!analysis_fs::exists(predictionRoot) ||
        !analysis_fs::is_directory(predictionRoot)) {
        throw std::runtime_error("Prediction directory does not exist: " +
                                 predictionDirectory);
    }
    if (!analysis_fs::exists(groundTruthRoot) ||
        !analysis_fs::is_directory(groundTruthRoot)) {
        throw std::runtime_error("Ground-truth directory does not exist: " +
                                 groundTruthDirectory);
    }
    analysis_fs::create_directories(outputRoot);
    iouThreshold = std::max(0.0f, std::min(1.0f, iouThreshold));

    const auto predictionJson = collectJsonFiles(predictionRoot, true);
    const auto groundTruthJson = collectJsonFiles(groundTruthRoot, false);
    std::set<std::string> stems;
    for (const auto& item : predictionJson) stems.insert(item.first);
    for (const auto& item : groundTruthJson) stems.insert(item.first);

    Report report;
    std::map<std::string, ClassMetrics> metrics;
    for (const auto& stem : stems) {
        if (logCallback) logCallback("Analyzing: " + stem);
        const analysis_fs::path originalPath =
            findImage(groundTruthRoot, stem, false);
        const analysis_fs::path predictionImagePath =
            findImage(predictionRoot, stem, true);
        cv::Mat original;
        if (!originalPath.empty()) original = cv::imread(originalPath.string());
        if (original.empty() && !predictionImagePath.empty()) {
            original = cv::imread(predictionImagePath.string());
        }

        cv::Size declaredSize;
        std::vector<Annotation> groundTruth;
        std::vector<Annotation> predictions;
        try {
            const auto gtIterator = groundTruthJson.find(stem);
            if (gtIterator != groundTruthJson.end()) {
                groundTruth = loadAnnotations(gtIterator->second,
                    original.size(), &declaredSize);
            }
            const auto predictionIterator = predictionJson.find(stem);
            if (predictionIterator != predictionJson.end()) {
                cv::Size predictionSize;
                predictions = loadAnnotations(predictionIterator->second,
                    original.size(), &predictionSize);
                if (declaredSize.width <= 0 || declaredSize.height <= 0) {
                    declaredSize = predictionSize;
                }
            }
        } catch (const std::exception& error) {
            ++report.skippedImages;
            if (logCallback) {
                logCallback("Skipped " + stem + ": " + error.what());
            }
            continue;
        }
        if (original.empty() && declaredSize.width > 0 && declaredSize.height > 0) {
            original = cv::Mat(declaredSize, CV_8UC3, cv::Scalar(245, 245, 245));
        }
        if (original.empty()) {
            ++report.skippedImages;
            if (logCallback) logCallback("Skipped " + stem + ": image missing");
            continue;
        }

        std::vector<size_t> predictionOrder(predictions.size());
        for (size_t index = 0; index < predictions.size(); ++index) {
            predictionOrder[index] = index;
        }
        std::stable_sort(predictionOrder.begin(), predictionOrder.end(),
            [&](size_t left, size_t right) {
                return predictions[left].score > predictions[right].score;
            });
        std::vector<int> predictionMatches(predictions.size(), -1);
        std::vector<int> groundTruthMatches(groundTruth.size(), -1);
        std::vector<double> matchIou(predictions.size(), 0.0);
        for (const size_t predictionIndex : predictionOrder) {
            int bestGroundTruth = -1;
            double bestIou = 0.0;
            for (size_t gtIndex = 0; gtIndex < groundTruth.size(); ++gtIndex) {
                if (groundTruthMatches[gtIndex] >= 0 ||
                    groundTruth[gtIndex].label !=
                        predictions[predictionIndex].label) {
                    continue;
                }
                const double iou = polygonIou(
                    predictions[predictionIndex], groundTruth[gtIndex],
                    original.size());
                if (iou > bestIou) {
                    bestIou = iou;
                    bestGroundTruth = static_cast<int>(gtIndex);
                }
            }
            if (bestGroundTruth >= 0 && bestIou >= iouThreshold) {
                predictionMatches[predictionIndex] = bestGroundTruth;
                groundTruthMatches[static_cast<size_t>(bestGroundTruth)] =
                    static_cast<int>(predictionIndex);
                matchIou[predictionIndex] = bestIou;
            }
        }

        for (size_t gtIndex = 0; gtIndex < groundTruth.size(); ++gtIndex) {
            const bool matched = groundTruthMatches[gtIndex] >= 0;
            addMetrics(metrics, groundTruth[gtIndex].label, 1, 0,
                       matched ? 1 : 0, matched ? 0 : 1, 0);
        }
        for (size_t predictionIndex = 0;
             predictionIndex < predictions.size(); ++predictionIndex) {
            const bool matched = predictionMatches[predictionIndex] >= 0;
            addMetrics(metrics, predictions[predictionIndex].label, 0, 1,
                       0, 0, matched ? 0 : 1);
        }

        cv::Mat visualization = original.clone();
        const cv::Scalar matchedGroundTruthColor(0, 255, 0);
        const cv::Scalar missedColor(0, 0, 255);
        const cv::Scalar matchedPredictionColor(255, 80, 0);
        const cv::Scalar falsePositiveColor(255, 0, 255);
        std::vector<std::string> legendLines = {
            "GT matched - green", "GT missed - red",
            "PRED matched - blue", "PRED false positive - magenta"};
        std::vector<cv::Scalar> legendColors = {
            matchedGroundTruthColor, missedColor,
            matchedPredictionColor, falsePositiveColor};

        for (size_t gtIndex = 0; gtIndex < groundTruth.size(); ++gtIndex) {
            const bool matched = groundTruthMatches[gtIndex] >= 0;
            const cv::Scalar color = matched
                ? matchedGroundTruthColor : missedColor;
            drawContour(visualization, groundTruth[gtIndex], color, 3);
            drawTag(visualization, groundTruth[gtIndex],
                    "G" + std::to_string(gtIndex + 1), color);
            legendLines.push_back("G" + std::to_string(gtIndex + 1) + " " +
                groundTruth[gtIndex].label + (matched ? " TP" : " MISSED"));
            legendColors.push_back(color);
        }
        for (size_t predictionIndex = 0;
             predictionIndex < predictions.size(); ++predictionIndex) {
            const bool matched = predictionMatches[predictionIndex] >= 0;
            const cv::Scalar color = matched
                ? matchedPredictionColor : falsePositiveColor;
            drawContour(visualization, predictions[predictionIndex], color, 2);
            drawTag(visualization, predictions[predictionIndex],
                    "P" + std::to_string(predictionIndex + 1), color);
            std::ostringstream entry;
            entry << "P" << predictionIndex + 1 << ' '
                  << predictions[predictionIndex].label << ' '
                  << std::fixed << std::setprecision(3)
                  << std::max(0.0f, predictions[predictionIndex].score)
                  << (matched ? " TP IoU=" : " FALSE POSITIVE");
            if (matched) entry << std::setprecision(3) << matchIou[predictionIndex];
            legendLines.push_back(entry.str());
            legendColors.push_back(color);
        }
        if (showOverlayText) drawLegend(visualization, legendLines, legendColors);

        const analysis_fs::path visualizationPath =
            outputRoot / (stem + "_analysis.png");
        if (!cv::imwrite(visualizationPath.string(), visualization)) {
            ++report.skippedImages;
            if (logCallback) {
                logCallback("Cannot save visualization: " +
                            visualizationPath.string());
            }
            continue;
        }
        analysis_fs::path reviewOriginal = originalPath;
        if (reviewOriginal.empty()) {
            reviewOriginal = outputRoot / (stem + "_source.png");
            cv::imwrite(reviewOriginal.string(), original);
        }
        report.reviewItems.push_back(
            {reviewOriginal.string(), visualizationPath.string()});
        ++report.images;
    }

    for (const auto& item : metrics) report.classes.push_back(item.second);
    report.total.label = "TOTAL";
    for (const auto& item : report.classes) {
        report.total.groundTruth += item.groundTruth;
        report.total.predictions += item.predictions;
        report.total.truePositive += item.truePositive;
        report.total.falseNegative += item.falseNegative;
        report.total.falsePositive += item.falsePositive;
    }
    writeReports(outputRoot, iouThreshold, report);
    return report;
}

}  // namespace segmentation_analysis
