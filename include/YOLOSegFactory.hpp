#pragma once

#include "YOLOSegBase.hpp"
#include "YOLO11/YOLO11Seg.hpp"
#include "YOLO26/YOLO26Seg.hpp"

/**
 * @brief Factory class to dynamically instantiate the appropriate detector
 * (YOLOv11SegDetector or YOLOv26SegDetector) by inspecting the ONNX model tensor shape.
 */
class YOLOSegFactory {
public:
  static std::unique_ptr<IYOLOSegDetector> createDetector(
      const std::string &modelPath,
      const std::string &labelsPath,
      bool useGPU = false) {
    
    try {
      Ort::Env tempEnv(ORT_LOGGING_LEVEL_WARNING, "FactoryCheck");
      Ort::SessionOptions options;
#ifdef _WIN32
      std::wstring w_modelPath(modelPath.begin(), modelPath.end());
      Ort::Session tempSession(tempEnv, w_modelPath.c_str(), options);
#else
      Ort::Session tempSession(tempEnv, modelPath.c_str(), options);
#endif
      if (tempSession.GetOutputCount() > 0) {
        auto typeInfo = tempSession.GetOutputTypeInfo(0);
        auto shape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3) {
          if (shape[1] > shape[2]) {
            // [1, N, C] format e.g. [1, 300, 38] -> YOLO26
            std::cout << "[INFO] YOLOSegFactory: Detected YOLO26 tensor layout [1, N, C]." << std::endl;
            return std::make_unique<yolo26::YOLOv26SegDetector>(modelPath, labelsPath, useGPU);
          }
        }
      }
    } catch (const std::exception &e) {
      std::cout << "[WARNING] YOLOSegFactory shape inspection: " << e.what()
                << " - falling back to default detector." << std::endl;
    }

    std::cout << "[INFO] YOLOSegFactory: Detected YOLO11 tensor layout [1, C, N]." << std::endl;
    return std::make_unique<yolo11::YOLOv11SegDetector>(modelPath, labelsPath, useGPU);
  }
};
