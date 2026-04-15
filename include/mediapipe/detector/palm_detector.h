#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "mediapipe/common/types.h"
#include "mediapipe/common/rknn_model.h"
#include "mediapipe/detector/palm_decoder.h"

namespace mediapipe_demo {

class PalmDetector {
 public:
  PalmDetector();

  bool LoadModel(const std::string& model_path);
  rknn_tensor_mem* InputMemory();
  const rknn_tensor_attr& InputAttr() const;
  bool SyncInputMemory();
  std::optional<PalmDetection> InferPrepared(float score_threshold);
  std::vector<PalmDetection> InferPreparedMulti(float score_threshold,
                                                size_t max_detections);
  std::optional<PalmDetection> Infer(const cv::Mat& frame,
                                     const PreprocessMeta& meta,
                                     float score_threshold);
  std::vector<PalmDetection> InferMulti(const cv::Mat& frame,
                                        const PreprocessMeta& meta,
                                        float score_threshold,
                                        size_t max_detections);

 private:
  std::vector<PalmDetection> DecodeOutputs(const std::vector<std::vector<float>>& outputs,
                                           float score_threshold,
                                           size_t max_detections) const;

  PalmDecoder decoder_;
  RknnModel model_;
};

}  // namespace mediapipe_demo
