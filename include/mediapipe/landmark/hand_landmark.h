#pragma once

#include <string>
#include <optional>

#include <opencv2/core.hpp>

#include "mediapipe/common/types.h"
#include "mediapipe/common/rknn_model.h"

namespace mediapipe_demo {

class HandLandmark {
 public:
  bool LoadModel(const std::string& model_path, int core_mask = -1);
  rknn_tensor_mem* InputMemory();
  const rknn_tensor_attr& InputAttr() const;
  bool SyncInputMemory();
  std::optional<HandLandmarks> InferPrepared(const PreprocessMeta& meta);
  std::optional<HandLandmarks> Infer(const cv::Mat& roi,
                                     const PreprocessMeta& meta);

 private:
  RknnModel model_;
};

}  // namespace mediapipe_demo
