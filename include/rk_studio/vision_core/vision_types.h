#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rkstudio::vision {

enum class PixelFormat {
  kNv12,
  kBgr,
  kRgb,
};

enum class TrackingMode {
  kDetect,
  kTrack,
  kRecover,
  kNoHand,
};

struct FrameRef {
  std::string camera_id;
  uint64_t pts_ns = 0;
  int width = 0;
  int height = 0;
  int stride = 0;
  uint32_t fourcc = 0;
  PixelFormat pixel_format = PixelFormat::kNv12;
  const uint8_t* mapped_ptr = nullptr;
  size_t bytes_used = 0;
  int dmabuf_fd = -1;
  std::shared_ptr<void> owned_data;  // type-erased; keeps frame data alive
};

struct VisionFrame {
  FrameRef rgb;
};

struct MediapipeProcessorConfig {
  std::string detector_model;
  std::string landmark_model;
  int queue_depth = 1;
};

struct RoiRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;
};

struct Landmark3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct HandResult {
  int hand_id = -1;
  std::optional<RoiRect> roi;
  std::vector<Landmark3f> landmarks;
  std::string gesture;
  float gesture_score = 0.0f;
  TrackingMode tracking_mode = TrackingMode::kNoHand;
  float motion_norm = 0.0f;
  float rotation_deg = 0.0f;
  int fast_motion_cooldown = 0;
};

struct MediapipeResult {
  std::string camera_id;
  uint64_t pts_ns = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::vector<HandResult> hands;
  float fps = 0.0f;
  bool ok = false;
  std::string error;
};

struct ObjectDetection {
  int class_id = -1;
  std::string class_name;
  float score = 0.0f;
  RoiRect box;
};

struct YoloProcessorConfig {
  std::string model;
  std::vector<std::string> class_names;
  int queue_depth = 1;
  float confidence_threshold = 0.25f;
  float nms_threshold = 0.45f;
  int max_detections = 50;
};

struct YoloResult {
  std::string camera_id;
  uint64_t pts_ns = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::vector<ObjectDetection> detections;
  float fps = 0.0f;
  bool ok = false;
  std::string error;
};

struct FaceLandmark2f {
  float x = 0.0f;
  float y = 0.0f;
};

struct ExpressionScore {
  std::string label;
  float score = 0.0f;
};

struct ActionUnitScore {
  std::string name;
  float score = 0.0f;
};

struct FaceExpressionResultItem {
  int face_id = -1;
  RoiRect box;
  std::vector<FaceLandmark2f> landmarks;
  std::string expression;
  float expression_score = 0.0f;
  std::vector<ExpressionScore> expression_scores;
  std::vector<ActionUnitScore> action_units;
};

struct FaceExpressionProcessorConfig {
  std::string detector_model;
  std::string expression_model;
  std::vector<std::string> expression_labels;
  int queue_depth = 1;
  float confidence_threshold = 0.5f;
  float nms_threshold = 0.4f;
  float expression_threshold = 0.35f;
  int max_faces = 1;
};

struct FaceExpressionResult {
  std::string camera_id;
  uint64_t pts_ns = 0;
  int frame_width = 0;
  int frame_height = 0;
  std::vector<FaceExpressionResultItem> faces;
  float fps = 0.0f;
  bool ok = false;
  std::string error;
};

}  // namespace rkstudio::vision
