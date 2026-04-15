#include "mediapipe/detector/palm_decoder.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace mediapipe_demo {

PalmDecoder::PalmDecoder(int input_size)
    : input_size_(input_size), anchors_(GenerateAnchors()) {}

std::optional<PalmDetection> PalmDecoder::Decode(const std::vector<float>& regressions,
                                                 const std::vector<float>& scores,
                                                 float score_threshold) const {
  const std::vector<PalmDetection> detections =
      DecodeMulti(regressions, scores, score_threshold, 1);
  if (detections.empty()) {
    return std::nullopt;
  }
  return detections.front();
}

std::vector<PalmDetection> PalmDecoder::DecodeMulti(const std::vector<float>& regressions,
                                                    const std::vector<float>& scores,
                                                    float score_threshold,
                                                    size_t max_detections) const {
  const size_t box_count = std::min(anchors_.size(), scores.size());
  if (max_detections == 0 || box_count == 0 || regressions.size() < box_count * 18) {
    return {};
  }

  std::vector<BBox> boxes;
  std::vector<float> valid_scores;
  boxes.reserve(box_count);
  valid_scores.reserve(box_count);

  for (size_t i = 0; i < box_count; ++i) {
    const float activated_score = 1.0f / (1.0f + std::exp(-scores[i]));
    if (!std::isfinite(activated_score) || activated_score <= score_threshold) {
      continue;
    }

    const float* reg = regressions.data() + i * 18;
    const Anchor& anchor = anchors_[i];
    const float cx = reg[0] / static_cast<float>(input_size_) + anchor.cx;
    const float cy = reg[1] / static_cast<float>(input_size_) + anchor.cy;
    const float w = reg[2] / static_cast<float>(input_size_);
    const float h = reg[3] / static_cast<float>(input_size_);
    const BBox box{
        cx - w * 0.5f,
        cy - h * 0.5f,
        cx + w * 0.5f,
        cy + h * 0.5f,
    };

    if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
        !std::isfinite(box.x2) || !std::isfinite(box.y2) ||
        box.x2 <= box.x1 || box.y2 <= box.y1) {
      continue;
    }

    boxes.push_back(box);
    valid_scores.push_back(activated_score);
  }

  if (boxes.empty()) {
    return {};
  }

  const std::vector<int> keep = NmsXyxy(boxes, valid_scores, 0.3f);
  if (keep.empty()) {
    return {};
  }

  std::vector<PalmDetection> detections;
  detections.reserve(std::min(max_detections, keep.size()));
  for (const int idx : keep) {
    detections.push_back(PalmDetection{boxes[idx], valid_scores[idx]});
    if (detections.size() >= max_detections) {
      break;
    }
  }
  return detections;
}

std::vector<PalmDecoder::Anchor> PalmDecoder::GenerateAnchors() const {
  std::vector<Anchor> anchors;
  for (const auto& spec : {std::pair<int, int>{8, 2}, {16, 6}}) {
    const int stride = spec.first;
    const int anchors_num = spec.second;
    const int grid_size = input_size_ / stride;
    for (int y = 0; y < grid_size; ++y) {
      for (int x = 0; x < grid_size; ++x) {
        const float cx = (x + 0.5f) / grid_size;
        const float cy = (y + 0.5f) / grid_size;
        for (int i = 0; i < anchors_num; ++i) {
          anchors.push_back(Anchor{cx, cy, 1.0f, 1.0f});
        }
      }
    }
  }
  return anchors;
}

std::vector<int> PalmDecoder::NmsXyxy(const std::vector<BBox>& boxes,
                                      const std::vector<float>& scores,
                                      float iou_threshold) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return scores[lhs] > scores[rhs];
  });

  std::vector<int> keep;
  while (!order.empty()) {
    const int current = order.front();
    keep.push_back(current);
    std::vector<int> remaining;
    for (size_t i = 1; i < order.size(); ++i) {
      const BBox& a = boxes[current];
      const BBox& b = boxes[order[i]];
      const float xx1 = std::max(a.x1, b.x1);
      const float yy1 = std::max(a.y1, b.y1);
      const float xx2 = std::min(a.x2, b.x2);
      const float yy2 = std::min(a.y2, b.y2);
      const float inter_w = std::max(0.0f, xx2 - xx1);
      const float inter_h = std::max(0.0f, yy2 - yy1);
      const float inter = inter_w * inter_h;
      const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
      const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
      const float denom = area_a + area_b - inter;
      const float iou = denom > 0.0f ? inter / denom : 0.0f;
      if (iou <= iou_threshold) {
        remaining.push_back(order[i]);
      }
    }
    order.swap(remaining);
  }
  return keep;
}

}  // namespace mediapipe_demo
