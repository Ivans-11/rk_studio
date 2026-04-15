#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace mediapipe_demo {

struct BBox {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

struct RoiRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;
};

struct PreprocessMeta {
  float scale = 1.0f;
  float pad_left = 0.0f;
  float pad_top = 0.0f;
  float input_w = 0.0f;
  float input_h = 0.0f;
  float src_w = 0.0f;
  float src_h = 0.0f;
};

struct CameraFrame {
  int width = 0;
  int height = 0;
  int stride = 0;
  uint32_t fourcc = 0;
  size_t bytes_used = 0;
  int dmabuf_fd = -1;
  int buffer_index = -1;
  uint8_t* data = nullptr;
};

struct PalmDetection {
  BBox bbox;
  float score = 0.0f;
};

struct HandLandmarks {
  std::array<cv::Point3f, 21> points{};
};

enum class TrackingMode {
  kDetect,
  kTrack,
  kRecover,
  kNoHand,
};

struct HandFrameResult {
  int id = 0;
  TrackingMode mode = TrackingMode::kNoHand;
  std::optional<RoiRect> roi;
  std::optional<HandLandmarks> landmarks;
  float rotation_deg = 0.0f;
  float motion_norm = 0.0f;
  int fast_motion_cooldown = 0;
};

struct FrameTimings {
  double bgr_ms = 0.0;
  double detector_ms = 0.0;
  double landmark_ms = 0.0;
  double pipeline_ms = 0.0;
};

struct FrameResult {
  std::vector<HandFrameResult> hands;
  std::optional<FrameTimings> timings;
  float fps = 0.0f;
};

}  // namespace mediapipe_demo
