#include "mediapipe/landmark/hand_landmark.h"

#include <algorithm>

#include "mediapipe/preprocess/image_ops.h"

namespace mediapipe_demo {

bool HandLandmark::LoadModel(const std::string& model_path, int core_mask) {
  return model_.Load(model_path, core_mask);
}

rknn_tensor_mem* HandLandmark::InputMemory() {
  return model_.InputMemory();
}

const rknn_tensor_attr& HandLandmark::InputAttr() const {
  return model_.InputAttr();
}

bool HandLandmark::SyncInputMemory() {
  return model_.SyncInputMemory();
}

std::optional<HandLandmarks> HandLandmark::InferPrepared(const PreprocessMeta& meta) {
  std::vector<std::vector<float>> outputs;
  if (!model_.Run(&outputs)) {
    return std::nullopt;
  }

  const std::vector<float>* landmark_tensor = nullptr;
  for (const auto& output : outputs) {
    if (output.size() == 63U) {
      landmark_tensor = &output;
      break;
    }
  }
  if (landmark_tensor == nullptr) {
    return std::nullopt;
  }

  HandLandmarks landmarks;
  float max_xy = 0.0f;
  for (size_t i = 0; i < 21; ++i) {
    max_xy = std::max(max_xy, (*landmark_tensor)[i * 3 + 0]);
    max_xy = std::max(max_xy, (*landmark_tensor)[i * 3 + 1]);
  }

  const bool is_normalized = max_xy <= 2.0f;
  for (size_t i = 0; i < 21; ++i) {
    float x = (*landmark_tensor)[i * 3 + 0];
    float y = (*landmark_tensor)[i * 3 + 1];
    if (is_normalized) {
      x *= meta.input_w;
      y *= meta.input_h;
    }
    const cv::Point2f mapped = MapPointFromInputToSource(cv::Point2f(x, y), meta);
    landmarks.points[i] = cv::Point3f(mapped.x, mapped.y, (*landmark_tensor)[i * 3 + 2]);
  }
  return landmarks;
}

std::optional<HandLandmarks> HandLandmark::Infer(const cv::Mat& roi,
                                                 const PreprocessMeta& meta) {
  if (!model_.CopyInput(roi)) {
    return std::nullopt;
  }
  return InferPrepared(meta);
}

}  // namespace mediapipe_demo
