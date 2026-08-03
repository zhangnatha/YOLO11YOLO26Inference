#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <string>
#include <vector>

#include "../YOLOSegBase.hpp"
#include "YOLO11Util.hpp"

namespace yolo11 {

/**
 * @brief YOLOv11SegDetector class handles loading YOLO11
 * segmentation ONNX models, preprocessing, inference, and postprocessing.
 * Output tensor format: [1, 40, 8400] (CbyN) with bounding box [cx, cy, w, h].
 */
class YOLOv11SegDetector : public IYOLOSegDetector {
public:
  YOLOv11SegDetector(const std::string &modelPath,
                     const std::string &labelsPath, bool useGPU = false);

  // Main API
  std::vector<SegmentedDetection> segment(const cv::Mat &image,
                                          float confThreshold = 0.40f,
                                          float iouThreshold = 0.45f) override;

  // Accessors
  const std::vector<std::string> &getClassNames() const override { return _classNames; }
  const std::vector<cv::Scalar> &getClassColors() const override { return _classColors; }

private:
  Ort::Env _env;
  Ort::SessionOptions _sessionOptions;
  Ort::Session _session{nullptr};

  bool _isDynamicInputShape{false};
  cv::Size _inputImageShape{640, 640};

  std::vector<Ort::AllocatedStringPtr> _inputNameAllocs;
  std::vector<const char *> _inputNames;
  std::vector<Ort::AllocatedStringPtr> _outputNameAllocs;
  std::vector<const char *> _outputNames;

  size_t _numInputNodes = 0;
  size_t _numOutputNodes = 0;

  std::vector<std::string> _classNames;
  std::vector<cv::Scalar> _classColors;

  // Helper functions
  cv::Mat preprocess(const cv::Mat &image, float *&blobPtr,
                     std::vector<int64_t> &inputTensorShape);

  std::vector<SegmentedDetection>
  postprocess(const cv::Size &origSize, const cv::Size &letterboxSize,
              const std::vector<Ort::Value> &outputs, float confThreshold,
              float iouThreshold);
};

inline YOLOv11SegDetector::YOLOv11SegDetector(const std::string &modelPath,
                                               const std::string &labelsPath,
                                               bool useGPU)
    : _env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11Seg") {
  _sessionOptions.SetIntraOpNumThreads(
      std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
  _sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::vector<std::string> providers = Ort::GetAvailableProviders();
  if (useGPU && std::find(providers.begin(), providers.end(),
                          "CUDAExecutionProvider") != providers.end()) {
    OrtCUDAProviderOptions cudaOptions;
    _sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    std::cout << "[INFO] Using GPU (CUDA) for YOLOv11 segmentation inference.\n";
  } else {
    std::cout << "[INFO] Using CPU for YOLOv11 segmentation inference.\n";
  }

#ifdef _WIN32
  std::wstring w_modelPath(modelPath.begin(), modelPath.end());
  _session = Ort::Session(_env, w_modelPath.c_str(), _sessionOptions);
#else
  _session = Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

  _numInputNodes = _session.GetInputCount();
  _numOutputNodes = _session.GetOutputCount();

  Ort::AllocatorWithDefaultOptions allocator;

  // Input
  {
    auto inNameAlloc = _session.GetInputNameAllocated(0, allocator);
    _inputNameAllocs.emplace_back(std::move(inNameAlloc));
    _inputNames.push_back(_inputNameAllocs.back().get());

    auto inTypeInfo = _session.GetInputTypeInfo(0);
    auto inShape = inTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (inShape.size() == 4) {
      if (inShape[2] == -1 || inShape[3] == -1) {
        _isDynamicInputShape = true;
        _inputImageShape = cv::Size(640, 640);
      } else {
        _inputImageShape = cv::Size(static_cast<int>(inShape[3]),
                                    static_cast<int>(inShape[2]));
      }
    } else {
      throw std::runtime_error("YOLO11 Model input is not 4D! Expected [N, C, H, W].");
    }
  }

  // Output
  if (_numOutputNodes != 2) {
    throw std::runtime_error("Expected exactly 2 output nodes: output0 and output1.");
  }

  for (size_t i = 0; i < _numOutputNodes; ++i) {
    auto outNameAlloc = _session.GetOutputNameAllocated(i, allocator);
    _outputNameAllocs.emplace_back(std::move(outNameAlloc));
    _outputNames.push_back(_outputNameAllocs.back().get());
  }

  _classNames = utils::getClassNames(labelsPath);
  _classColors = utils::generateColors(_classNames);

  std::cout << "[INFO] YOLOv11Seg loaded: " << modelPath << std::endl
            << "      Input Shape: " << _inputImageShape
            << (_isDynamicInputShape ? " (Dynamic)" : "") << std::endl
            << "      Output Count: " << _numOutputNodes << std::endl
            << "      Class Count: " << _classNames.size() << std::endl;
}

inline cv::Mat
YOLOv11SegDetector::preprocess(const cv::Mat &image, float *&blobPtr,
                               std::vector<int64_t> &inputTensorShape) {
  std::cout << "******************** [YOLO11] Preprocessing ********************" << std::endl;

  cv::Mat letterboxImage;
  utils::SegLetterBox(image, letterboxImage, _inputImageShape,
                      cv::Scalar(114, 114, 114), /*auto_=*/_isDynamicInputShape,
                      /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);

  inputTensorShape[2] = static_cast<int64_t>(letterboxImage.rows);
  inputTensorShape[3] = static_cast<int64_t>(letterboxImage.cols);

  letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f / 255.0f);

  size_t size = static_cast<size_t>(letterboxImage.rows) *
                static_cast<size_t>(letterboxImage.cols) * 3;
  blobPtr = new float[size];

  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] =
        cv::Mat(letterboxImage.rows, letterboxImage.cols, CV_32FC1,
                blobPtr + c * (letterboxImage.rows * letterboxImage.cols));
  }
  cv::split(letterboxImage, channels);

  return letterboxImage;
}

inline std::vector<SegmentedDetection>
YOLOv11SegDetector::postprocess(const cv::Size &origSize,
                                const cv::Size &letterboxSize,
                                const std::vector<Ort::Value> &outputs,
                                float confThreshold, float iouThreshold) {
  std::cout << "******************** [YOLO11] Postprocessing ********************" << std::endl;

  std::vector<SegmentedDetection> results;

  if (outputs.size() < 2) {
    throw std::runtime_error("Insufficient model outputs. Expected at least 2 outputs.");
  }

  const float *output0_ptr = outputs[0].GetTensorData<float>();
  const float *output1_ptr = outputs[1].GetTensorData<float>();

  auto shape0 = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 40, 8400]
  auto shape1 = outputs[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]

  if (shape1.size() != 4 || shape1[0] != 1 || shape1[1] != 32)
    throw std::runtime_error("Unexpected output1 shape. Expected [1, 32, maskH, maskW].");

  const size_t num_features = shape0[1]; // 40
  const size_t num_detections = shape0[2]; // 8400

  if (num_detections == 0) {
    return results;
  }

  const int numClasses = static_cast<int>(num_features - 4 - 32);
  if (numClasses <= 0) {
    throw std::runtime_error("Invalid number of classes for YOLO11.");
  }

  const int numBoxes = static_cast<int>(num_detections);
  const int maskH = static_cast<int>(shape1[2]);
  const int maskW = static_cast<int>(shape1[3]);

  constexpr int BOX_OFFSET = 0;
  constexpr int CLASS_CONF_OFFSET = 4;
  const int MASK_COEFF_OFFSET = numClasses + CLASS_CONF_OFFSET;

  std::vector<cv::Mat> prototypeMasks;
  prototypeMasks.reserve(32);
  for (int m = 0; m < 32; ++m) {
    cv::Mat proto(maskH, maskW, CV_32F,
                  const_cast<float *>(output1_ptr + m * maskH * maskW));
    prototypeMasks.emplace_back(proto.clone());
  }

  std::vector<SegBoundingBox> boxes;
  boxes.reserve(numBoxes);
  std::vector<float> confidences;
  confidences.reserve(numBoxes);
  std::vector<int> classIds;
  classIds.reserve(numBoxes);
  std::vector<std::vector<float>> maskCoefficientsList;
  maskCoefficientsList.reserve(numBoxes);

  for (int i = 0; i < numBoxes; ++i) {
    float xc = output0_ptr[BOX_OFFSET * numBoxes + i];
    float yc = output0_ptr[(BOX_OFFSET + 1) * numBoxes + i];
    float w = output0_ptr[(BOX_OFFSET + 2) * numBoxes + i];
    float h = output0_ptr[(BOX_OFFSET + 3) * numBoxes + i];

    SegBoundingBox box{static_cast<int>(std::round(xc - w / 2.0f)),
                       static_cast<int>(std::round(yc - h / 2.0f)),
                       static_cast<int>(std::round(w)),
                       static_cast<int>(std::round(h))};

    float maxConf = 0.0f;
    int classId = -1;
    for (int c = 0; c < numClasses; ++c) {
      float conf = output0_ptr[(CLASS_CONF_OFFSET + c) * numBoxes + i];
      if (conf > maxConf) {
        maxConf = conf;
        classId = c;
      }
    }

    if (maxConf < confThreshold) continue;

    boxes.push_back(box);
    confidences.push_back(maxConf);
    classIds.push_back(classId);

    std::vector<float> maskCoeffs(32);
    for (int m = 0; m < 32; ++m) {
      maskCoeffs[m] = output0_ptr[(MASK_COEFF_OFFSET + m) * numBoxes + i];
    }
    maskCoefficientsList.emplace_back(std::move(maskCoeffs));
  }

  if (boxes.empty()) {
    return results;
  }

  std::vector<int> nmsIndices;
  utils::SegNMSBoxes(boxes, confidences, confThreshold, iouThreshold, nmsIndices);

  if (nmsIndices.empty()) {
    return results;
  }

  results.reserve(nmsIndices.size());

  const float gain =
      std::min(static_cast<float>(letterboxSize.height) / origSize.height,
               static_cast<float>(letterboxSize.width) / origSize.width);
  const int scaledW = static_cast<int>(origSize.width * gain);
  const int scaledH = static_cast<int>(origSize.height * gain);
  const float padW = (letterboxSize.width - scaledW) / 2.0f;
  const float padH = (letterboxSize.height - scaledH) / 2.0f;

  const float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
  const float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

  for (const int idx : nmsIndices) {
    SegmentedDetection seg;
    seg.box = boxes[idx];
    seg.conf = confidences[idx];
    seg.classId = classIds[idx];

    seg.box = utils::scaleCoords(letterboxSize, seg.box, origSize, true);

    const auto &maskCoeffs = maskCoefficientsList[idx];

    cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
    for (int m = 0; m < 32; ++m) {
      finalMask += maskCoeffs[m] * prototypeMasks[m];
    }

    finalMask = utils::sigmoid(finalMask);

    int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
    int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
    int x2 = static_cast<int>(
        std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
    int y2 = static_cast<int>(
        std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));

    x1 = std::max(0, std::min(x1, maskW - 1));
    y1 = std::max(0, std::min(y1, maskH - 1));
    x2 = std::max(x1, std::min(x2, maskW));
    y2 = std::max(y1, std::min(y2, maskH));

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat croppedMask = finalMask(cropRect).clone();

    cv::Mat resizedMask;
    cv::resize(croppedMask, resizedMask, origSize, 0, 0, cv::INTER_LINEAR);

    cv::Mat binaryMask;
    cv::threshold(resizedMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8U);

    cv::Mat finalBinaryMask = cv::Mat::zeros(origSize, CV_8U);
    cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
    roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows);
    if (roi.area() > 0) {
      binaryMask(roi).copyTo(finalBinaryMask(roi));
    }

    seg.mask = finalBinaryMask;
    results.push_back(seg);
  }

  return results;
}

inline std::vector<SegmentedDetection>
YOLOv11SegDetector::segment(const cv::Mat &image, float confThreshold,
                            float iouThreshold) {
  std::cout << "******************** [YOLO11] Instance Segmentation Task ********************"
            << std::endl;
  cv::Mat src = image.clone();
  if (src.channels() != 3) { cv::cvtColor(src, src, cv::COLOR_GRAY2RGB); }
  else { cv::cvtColor(src, src, cv::COLOR_BGR2RGB); }

  float *blobPtr = nullptr;
  std::vector<int64_t> inputShape = {1, 3, _inputImageShape.height, _inputImageShape.width};
  cv::Mat letterboxImg = preprocess(src, blobPtr, inputShape);

  size_t inputSize = utils::vectorProduct(inputShape);
  std::vector<float> inputVals(blobPtr, blobPtr + inputSize);
  delete[] blobPtr;

  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
      Ort::Value::CreateTensor<float>(memInfo, inputVals.data(), inputSize,
                                      inputShape.data(), inputShape.size());

  std::vector<Ort::Value> outputs =
      _session.Run(Ort::RunOptions{nullptr}, _inputNames.data(), &inputTensor,
                   _numInputNodes, _outputNames.data(), _numOutputNodes);

  cv::Size letterboxSize(static_cast<int>(inputShape[3]),
                         static_cast<int>(inputShape[2]));
  return postprocess(src.size(), letterboxSize, outputs, confThreshold, iouThreshold);
}

} // namespace yolo11
