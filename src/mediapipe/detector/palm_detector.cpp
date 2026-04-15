#include "mediapipe/detector/palm_detector.h"

#include <iostream>

namespace mediapipe_demo {

PalmDetector::PalmDetector() : decoder_(192) {}

bool PalmDetector::LoadModel(const std::string& model_path) {
  return model_.Load(model_path);
}

rknn_tensor_mem* PalmDetector::InputMemory() {
  return model_.InputMemory();
}

const rknn_tensor_attr& PalmDetector::InputAttr() const {
  return model_.InputAttr();
}

bool PalmDetector::SyncInputMemory() {
  return model_.SyncInputMemory();
}

std::optional<PalmDetection> PalmDetector::InferPrepared(float score_threshold) {
  const std::vector<PalmDetection> detections = InferPreparedMulti(score_threshold, 1);
  if (detections.empty()) {
    return std::nullopt;
  }
  return detections.front();
}

std::vector<PalmDetection> PalmDetector::InferPreparedMulti(float score_threshold,
                                                            size_t max_detections) {
  std::vector<std::vector<float>> outputs;
  if (!model_.Run(&outputs)) {
    return {};
  }
  return DecodeOutputs(outputs, score_threshold, max_detections);
}

std::vector<PalmDetection> PalmDetector::DecodeOutputs(
    const std::vector<std::vector<float>>& outputs,
    float score_threshold,
    size_t max_detections) const {
  std::vector<float> regressions;
  std::vector<float> scores;
  for (const auto& output : outputs) {
    if (output.size() == 2016U * 18U) {
      regressions = output;
    } else if (output.size() == 2016U || output.size() == 2016U * 1U) {
      scores = output;
    }
  }

  if (regressions.empty() || scores.empty()) {
    std::cerr << "detector outputs do not match expected shapes\n";
    return {};
  }
  return decoder_.DecodeMulti(regressions, scores, score_threshold, max_detections);
}

std::optional<PalmDetection> PalmDetector::Infer(const cv::Mat& frame,
                                                 const PreprocessMeta& meta,
                                                 float score_threshold) {
  (void)meta;
  if (!model_.CopyInput(frame)) {
    return std::nullopt;
  }
  return InferPrepared(score_threshold);
}

std::vector<PalmDetection> PalmDetector::InferMulti(const cv::Mat& frame,
                                                    const PreprocessMeta& meta,
                                                    float score_threshold,
                                                    size_t max_detections) {
  (void)meta;
  if (!model_.CopyInput(frame)) {
    return {};
  }
  return InferPreparedMulti(score_threshold, max_detections);
}

}  // namespace mediapipe_demo
