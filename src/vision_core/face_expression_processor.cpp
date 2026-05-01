#include "rk_studio/vision_core/vision_processor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "mediapipe/common/rknn_model.h"
#include "mediapipe/preprocess/image_ops.h"

namespace rkstudio::vision {
namespace {

constexpr int kDetectorInputSize = 320;
constexpr int kExpressionInputSize = 112;
constexpr int kDetectorHeadCount = 3;
constexpr std::array<int, kDetectorHeadCount> kDetectorStrides = {8, 16, 32};
const std::array<cv::Point2f, 5> kCanonicalLandmarks = {
    cv::Point2f(38.2946f, 51.6963f),
    cv::Point2f(73.5318f, 51.5014f),
    cv::Point2f(56.0252f, 71.7366f),
    cv::Point2f(41.5493f, 92.3655f),
    cv::Point2f(70.7299f, 92.2041f),
};

std::vector<std::string> DefaultExpressionLabels() {
  return {"angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"};
}

cv::Mat ToRgbMat(const FrameRef& frame) {
  if (frame.mapped_ptr == nullptr || frame.width <= 0 || frame.height <= 0) {
    return {};
  }
  if (frame.pixel_format == PixelFormat::kRgb) {
    return cv::Mat(frame.height, frame.width, CV_8UC3,
                   const_cast<uint8_t*>(frame.mapped_ptr),
                   frame.stride > 0 ? frame.stride : frame.width * 3)
        .clone();
  }
  if (frame.pixel_format == PixelFormat::kBgr) {
    cv::Mat bgr(frame.height, frame.width, CV_8UC3,
                const_cast<uint8_t*>(frame.mapped_ptr),
                frame.stride > 0 ? frame.stride : frame.width * 3);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }
  if (frame.pixel_format == PixelFormat::kNv12) {
    const int stride = frame.stride > 0 ? frame.stride : frame.width;
    cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1,
                 const_cast<uint8_t*>(frame.mapped_ptr), stride);
    cv::Mat rgb;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    return rgb;
  }
  return {};
}

float IoU(const FaceExpressionResultItem& a, const FaceExpressionResultItem& b) {
  const int x1 = std::max(a.box.x1, b.box.x1);
  const int y1 = std::max(a.box.y1, b.box.y1);
  const int x2 = std::min(a.box.x2, b.box.x2);
  const int y2 = std::min(a.box.y2, b.box.y2);
  const int iw = std::max(0, x2 - x1);
  const int ih = std::max(0, y2 - y1);
  const float inter = static_cast<float>(iw * ih);
  const float area_a = static_cast<float>(
      std::max(0, a.box.x2 - a.box.x1) * std::max(0, a.box.y2 - a.box.y1));
  const float area_b = static_cast<float>(
      std::max(0, b.box.x2 - b.box.x1) * std::max(0, b.box.y2 - b.box.y1));
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

std::vector<FaceExpressionResultItem> Nms(std::vector<FaceExpressionResultItem> faces,
                                          float threshold,
                                          int max_faces) {
  std::sort(faces.begin(), faces.end(), [](const auto& a, const auto& b) {
    return a.expression_score > b.expression_score;
  });
  std::vector<FaceExpressionResultItem> kept;
  for (const auto& face : faces) {
    bool suppressed = false;
    for (const auto& prev : kept) {
      if (IoU(face, prev) > threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(face);
      if (max_faces > 0 && static_cast<int>(kept.size()) >= max_faces) {
        break;
      }
    }
  }
  return kept;
}

std::vector<float> Softmax(const std::vector<float>& logits) {
  std::vector<float> probs(logits.size(), 0.0f);
  if (logits.empty()) {
    return probs;
  }
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (float logit : logits) {
    sum += std::exp(logit - max_logit);
  }
  if (sum <= 0.0f) {
    return probs;
  }
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_logit) / sum;
  }
  return probs;
}

int64_t Product(const rknn_tensor_attr& attr) {
  int64_t product = 1;
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    product *= std::max(1u, attr.dims[i]);
  }
  return product;
}

bool IsNchwLayout(const rknn_tensor_attr& attr) {
  return attr.fmt == RKNN_TENSOR_NCHW;
}

int InferLastDim(const rknn_tensor_attr& attr, size_t buffer_size) {
  if (attr.n_dims == 0 || buffer_size == 0) {
    return 0;
  }
  if (IsNchwLayout(attr)) {
    if (attr.n_dims >= 4) {
      return static_cast<int>(attr.dims[1]);
    }
    return static_cast<int>(attr.dims[0]);
  }
  return static_cast<int>(attr.dims[attr.n_dims - 1]);
}

int InferLocationCount(const rknn_tensor_attr& attr, size_t buffer_size) {
  const int channels = InferLastDim(attr, buffer_size);
  if (channels <= 0) {
    return 0;
  }
  const int64_t elems = Product(attr);
  if (elems <= 0) {
    return 0;
  }
  const int64_t locations = elems / channels;
  return locations > 0 ? static_cast<int>(locations) : 0;
}

float TensorValueAt(const std::vector<float>& buffer,
                    const rknn_tensor_attr& attr,
                    int location,
                    int channel) {
  const int channels = InferLastDim(attr, buffer.size());
  const int locations = InferLocationCount(attr, buffer.size());
  if (channel < 0 || location < 0 || channel >= channels || location >= locations) {
    return 0.0f;
  }
  if (IsNchwLayout(attr)) {
    return buffer[static_cast<size_t>(channel) * static_cast<size_t>(locations) +
                  static_cast<size_t>(location)];
  }
  return buffer[static_cast<size_t>(location) * static_cast<size_t>(channels) +
                static_cast<size_t>(channel)];
}

struct DetectorOutputGroup {
  int stride = 0;
  int locations = 0;
  const std::vector<float>* cls = nullptr;
  const std::vector<float>* obj = nullptr;
  const std::vector<float>* bbox = nullptr;
  const std::vector<float>* kps = nullptr;
  const rknn_tensor_attr* cls_attr = nullptr;
  const rknn_tensor_attr* obj_attr = nullptr;
  const rknn_tensor_attr* bbox_attr = nullptr;
  const rknn_tensor_attr* kps_attr = nullptr;
};

std::vector<DetectorOutputGroup> GroupDetectorOutputs(
    const std::vector<std::vector<float>>& outputs,
    const std::vector<rknn_tensor_attr>& attrs) {
  std::vector<DetectorOutputGroup> groups;
  if (outputs.size() != attrs.size()) {
    return groups;
  }

  std::array<DetectorOutputGroup, kDetectorHeadCount> grouped{};
  std::array<bool, kDetectorHeadCount> valid{};
  for (size_t i = 0; i < outputs.size(); ++i) {
    const int locations = InferLocationCount(attrs[i], outputs[i].size());
    if (locations <= 0) {
      continue;
    }
    const int grid = static_cast<int>(std::round(std::sqrt(static_cast<float>(locations))));
    if (grid <= 0 || grid * grid != locations) {
      continue;
    }
    const int stride = kDetectorInputSize / grid;
    int group_index = -1;
    for (int head = 0; head < kDetectorHeadCount; ++head) {
      if (kDetectorStrides[head] == stride) {
        group_index = head;
        break;
      }
    }
    if (group_index < 0) {
      continue;
    }

    DetectorOutputGroup& group = grouped[static_cast<size_t>(group_index)];
    group.stride = stride;
    group.locations = locations;

    const int channels = InferLastDim(attrs[i], outputs[i].size());
    if (channels == 1) {
      if (group.obj == nullptr) {
        group.obj = &outputs[i];
        group.obj_attr = &attrs[i];
      } else {
        group.cls = &outputs[i];
        group.cls_attr = &attrs[i];
      }
    } else if (channels == 4) {
      group.bbox = &outputs[i];
      group.bbox_attr = &attrs[i];
    } else if (channels == 10) {
      group.kps = &outputs[i];
      group.kps_attr = &attrs[i];
    } else if (channels == 2) {
      if (group.cls == nullptr) {
        group.cls = &outputs[i];
        group.cls_attr = &attrs[i];
      }
    }
    valid[static_cast<size_t>(group_index)] = true;
  }

  for (int head = 0; head < kDetectorHeadCount; ++head) {
    const DetectorOutputGroup& group = grouped[static_cast<size_t>(head)];
    if (valid[static_cast<size_t>(head)] && group.cls != nullptr && group.obj != nullptr &&
        group.bbox != nullptr && group.kps != nullptr) {
      groups.push_back(group);
    }
  }
  return groups;
}

std::vector<FaceExpressionResultItem> DecodeDetectorOutputs(
    const std::vector<std::vector<float>>& outputs,
    const std::vector<rknn_tensor_attr>& attrs,
    const mediapipe_demo::PreprocessMeta& meta,
    int frame_w,
    int frame_h,
    float conf_threshold) {
  std::vector<FaceExpressionResultItem> faces;
  const auto groups = GroupDetectorOutputs(outputs, attrs);
  for (const auto& group : groups) {
    const int grid = group.locations > 0
                         ? static_cast<int>(std::round(std::sqrt(static_cast<float>(group.locations))))
                         : 0;
    if (grid <= 0 || group.stride <= 0) {
      continue;
    }

    for (int y = 0; y < grid; ++y) {
      for (int x = 0; x < grid; ++x) {
        const int location = y * grid + x;
        const float cls_score = std::clamp(TensorValueAt(*group.cls, *group.cls_attr, location, 0),
                                           0.0f,
                                           1.0f);
        const float obj_score = std::clamp(TensorValueAt(*group.obj, *group.obj_attr, location, 0),
                                           0.0f,
                                           1.0f);
        const float score = std::sqrt(std::max(0.0f, cls_score * obj_score));
        if (score < conf_threshold) {
          continue;
        }

        const float stride = static_cast<float>(group.stride);
        const float dx = TensorValueAt(*group.bbox, *group.bbox_attr, location, 0);
        const float dy = TensorValueAt(*group.bbox, *group.bbox_attr, location, 1);
        const float dw = TensorValueAt(*group.bbox, *group.bbox_attr, location, 2);
        const float dh = TensorValueAt(*group.bbox, *group.bbox_attr, location, 3);

        const float cx = (static_cast<float>(x) + dx) * stride;
        const float cy = (static_cast<float>(y) + dy) * stride;
        const float w = std::exp(dw) * stride;
        const float h = std::exp(dh) * stride;

        const cv::Point2f src_p1 = mediapipe_demo::MapPointFromInputToSource(
            cv::Point2f(cx - 0.5f * w, cy - 0.5f * h), meta);
        const cv::Point2f src_p2 = mediapipe_demo::MapPointFromInputToSource(
            cv::Point2f(cx + 0.5f * w, cy + 0.5f * h), meta);

        FaceExpressionResultItem item;
        item.face_id = static_cast<int>(faces.size());
        item.expression_score = score;
        item.box.x1 = std::clamp(static_cast<int>(std::round(std::min(src_p1.x, src_p2.x))), 0, frame_w - 1);
        item.box.y1 = std::clamp(static_cast<int>(std::round(std::min(src_p1.y, src_p2.y))), 0, frame_h - 1);
        item.box.x2 = std::clamp(static_cast<int>(std::round(std::max(src_p1.x, src_p2.x))), 0, frame_w);
        item.box.y2 = std::clamp(static_cast<int>(std::round(std::max(src_p1.y, src_p2.y))), 0, frame_h);
        if (item.box.x2 <= item.box.x1 || item.box.y2 <= item.box.y1) {
          continue;
        }

        item.landmarks.reserve(5);
        for (int k = 0; k < 5; ++k) {
          const float kx = TensorValueAt(*group.kps, *group.kps_attr, location, k * 2 + 0);
          const float ky = TensorValueAt(*group.kps, *group.kps_attr, location, k * 2 + 1);
          const cv::Point2f src_point = mediapipe_demo::MapPointFromInputToSource(
              cv::Point2f((static_cast<float>(x) + kx) * stride,
                          (static_cast<float>(y) + ky) * stride),
              meta);
          item.landmarks.push_back(
              FaceLandmark2f{std::clamp(src_point.x, 0.0f, static_cast<float>(frame_w - 1)),
                             std::clamp(src_point.y, 0.0f, static_cast<float>(frame_h - 1))});
        }
        faces.push_back(std::move(item));
      }
    }
  }
  return faces;
}

std::optional<cv::Mat> AlignFace(const cv::Mat& rgb,
                                 const FaceExpressionResultItem& face) {
  if (rgb.empty() || face.landmarks.size() < 5) {
    return std::nullopt;
  }

  std::vector<cv::Point2f> src_points;
  src_points.reserve(5);
  std::vector<cv::Point2f> dst_points;
  dst_points.reserve(5);
  for (size_t i = 0; i < 5; ++i) {
    src_points.emplace_back(face.landmarks[i].x, face.landmarks[i].y);
    dst_points.push_back(kCanonicalLandmarks[i]);
  }

  cv::Mat affine = cv::estimateAffinePartial2D(src_points, dst_points);
  if (affine.empty()) {
    return std::nullopt;
  }

  cv::Mat aligned;
  cv::warpAffine(rgb,
                 aligned,
                 affine,
                 cv::Size(kExpressionInputSize, kExpressionInputSize),
                 cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT,
                 cv::Scalar(0, 0, 0));
  if (aligned.empty() || !aligned.isContinuous()) {
    aligned = aligned.clone();
  }
  return aligned;
}

std::vector<float> FlattenExpressionOutput(
    const std::vector<std::vector<float>>& outputs,
    const std::vector<rknn_tensor_attr>& attrs) {
  if (outputs.empty() || outputs.size() != attrs.size()) {
    return {};
  }

  size_t best_index = 0;
  size_t best_size = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const int channels = InferLastDim(attrs[i], outputs[i].size());
    if (channels > 1 && outputs[i].size() > best_size) {
      best_index = i;
      best_size = outputs[i].size();
    }
  }
  if (best_size == 0) {
    best_size = outputs[0].size();
    best_index = 0;
  }

  const int channels = InferLastDim(attrs[best_index], outputs[best_index].size());
  if (channels <= 0) {
    return outputs[best_index];
  }

  std::vector<float> logits;
  logits.reserve(static_cast<size_t>(channels));
  for (int c = 0; c < channels; ++c) {
    logits.push_back(TensorValueAt(outputs[best_index], attrs[best_index], 0, c));
  }
  return logits;
}

}  // namespace

class FaceExpressionProcessor final : public IFaceExpressionProcessor {
 public:
  ~FaceExpressionProcessor() override { Stop(); }

  bool Start(const FaceExpressionProcessorConfig& config, std::string* err) override {
    Stop();
    config_ = config;
    if (config_.expression_labels.empty()) {
      config_.expression_labels = DefaultExpressionLabels();
    }

    detector_model_ = std::make_unique<mediapipe_demo::RknnModel>();
    if (!detector_model_->Load(config_.detector_model)) {
      if (err) {
        *err = "failed to load face detector model: " + config_.detector_model;
      }
      detector_model_.reset();
      return false;
    }

    expression_model_ = std::make_unique<mediapipe_demo::RknnModel>();
    if (!expression_model_->Load(config_.expression_model)) {
      if (err) {
        *err = "failed to load face expression model: " + config_.expression_model;
      }
      detector_model_.reset();
      expression_model_.reset();
      return false;
    }

    running_ = true;
    worker_ = std::thread([this] { RunLoop(); });
    return true;
  }

  void Submit(const FrameRef& frame) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (!running_) {
      return;
    }
    pending_frames_.push_back(frame);
    while (pending_frames_.size() > std::max<size_t>(1, config_.queue_depth)) {
      pending_frames_.pop_front();
    }
    cv_.notify_one();
  }

  std::optional<FaceExpressionResult> PollResult() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (results_.empty()) {
      return std::nullopt;
    }
    FaceExpressionResult result = std::move(results_.front());
    results_.pop_front();
    return result;
  }

  void Stop() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      running_ = false;
      pending_frames_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      results_.clear();
    }
    detector_model_.reset();
    expression_model_.reset();
  }

 private:
  void RunLoop() {
    while (true) {
      FrameRef frame;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !running_ || !pending_frames_.empty(); });
        if (!running_ && pending_frames_.empty()) {
          break;
        }
        frame = std::move(pending_frames_.front());
        pending_frames_.pop_front();
      }

      FaceExpressionResult result = ProcessFrame(frame);
      {
        std::lock_guard<std::mutex> lock(mu_);
        results_.push_back(std::move(result));
        while (results_.size() > std::max<size_t>(1, config_.queue_depth * 2)) {
          results_.pop_front();
        }
      }
    }
  }

  FaceExpressionResult ProcessFrame(const FrameRef& frame) {
    const auto started = std::chrono::steady_clock::now();

    FaceExpressionResult result;
    result.camera_id = frame.camera_id;
    result.pts_ns = frame.pts_ns;
    result.frame_width = frame.width;
    result.frame_height = frame.height;

    cv::Mat rgb = ToRgbMat(frame);
    if (rgb.empty()) {
      result.error = "empty face expression input frame";
      return result;
    }

    mediapipe_demo::PreprocessMeta meta;
    cv::Mat detector_input = mediapipe_demo::LetterboxPadding(
        rgb, cv::Size(kDetectorInputSize, kDetectorInputSize), &meta);
    if (!detector_input.isContinuous()) {
      detector_input = detector_input.clone();
    }

    std::vector<std::vector<float>> detector_outputs;
    if (!detector_model_ || !detector_model_->Infer(detector_input, &detector_outputs)) {
      result.error = "face detector RKNN inference failed";
      return result;
    }

    std::vector<FaceExpressionResultItem> faces = DecodeDetectorOutputs(
        detector_outputs,
        detector_model_->OutputAttrs(),
        meta,
        frame.width,
        frame.height,
        config_.confidence_threshold);
    faces = Nms(std::move(faces), config_.nms_threshold, config_.max_faces);

    int face_id = 0;
    for (auto& face : faces) {
      face.face_id = face_id++;
      auto aligned = AlignFace(rgb, face);
      if (!aligned.has_value()) {
        continue;
      }

      std::vector<std::vector<float>> expression_outputs;
      if (!expression_model_ || !expression_model_->Infer(*aligned, &expression_outputs)) {
        continue;
      }
      const std::vector<float> logits = FlattenExpressionOutput(
          expression_outputs, expression_model_->OutputAttrs());
      const std::vector<float> probs = Softmax(logits);
      if (probs.empty()) {
        continue;
      }

      const auto best_it = std::max_element(probs.begin(), probs.end());
      const int best_index = static_cast<int>(std::distance(probs.begin(), best_it));
      face.expression_score = *best_it;
      if (face.expression_score >= config_.expression_threshold &&
          best_index >= 0 &&
          best_index < static_cast<int>(config_.expression_labels.size())) {
        face.expression = config_.expression_labels[static_cast<size_t>(best_index)];
      } else {
        face.expression = "unknown";
      }

      face.expression_scores.clear();
      const size_t label_count = std::min(probs.size(), config_.expression_labels.size());
      face.expression_scores.reserve(label_count);
      for (size_t i = 0; i < label_count; ++i) {
        face.expression_scores.push_back(
            ExpressionScore{config_.expression_labels[i], probs[i]});
      }
    }

    result.faces = std::move(faces);
    const auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    result.fps = elapsed.count() > 0.0 ? static_cast<float>(1.0 / elapsed.count()) : 0.0f;
    result.ok = true;
    return result;
  }

  FaceExpressionProcessorConfig config_;
  std::unique_ptr<mediapipe_demo::RknnModel> detector_model_;
  std::unique_ptr<mediapipe_demo::RknnModel> expression_model_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<FrameRef> pending_frames_;
  std::deque<FaceExpressionResult> results_;
  bool running_ = false;
  std::thread worker_;
};

std::unique_ptr<IFaceExpressionProcessor> CreateFaceExpressionProcessor() {
  return std::make_unique<FaceExpressionProcessor>();
}

}  // namespace rkstudio::vision
