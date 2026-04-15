#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "mediapipe/common/types.h"

namespace mediapipe_demo {

class PalmDecoder {
 public:
  explicit PalmDecoder(int input_size = 192);

  std::optional<PalmDetection> Decode(const std::vector<float>& regressions,
                                      const std::vector<float>& scores,
                                      float score_threshold) const;
  std::vector<PalmDetection> DecodeMulti(const std::vector<float>& regressions,
                                         const std::vector<float>& scores,
                                         float score_threshold,
                                         size_t max_detections) const;

 private:
  struct Anchor {
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 1.0f;
    float h = 1.0f;
  };

  std::vector<Anchor> GenerateAnchors() const;
  static std::vector<int> NmsXyxy(const std::vector<BBox>& boxes,
                                  const std::vector<float>& scores,
                                  float iou_threshold);

  int input_size_ = 192;
  std::vector<Anchor> anchors_;
};

}  // namespace mediapipe_demo
