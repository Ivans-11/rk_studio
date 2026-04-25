#include "rk_studio/vision_core/vision_processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "mediapipe/common/rknn_model.h"
#include "mediapipe/preprocess/hw_preprocess.h"
#include "mediapipe/preprocess/image_ops.h"

namespace rkstudio::vision {
namespace {

constexpr int kYoloInputSize = 640;
constexpr int kRegMax = 16;
constexpr int kHeadCount = 3;

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

float Iou(const ObjectDetection& a, const ObjectDetection& b) {
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

std::vector<ObjectDetection> Nms(std::vector<ObjectDetection> detections,
                                 float threshold,
                                 int max_detections) {
  std::sort(detections.begin(), detections.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });
  std::vector<ObjectDetection> kept;
  for (const auto& det : detections) {
    bool suppressed = false;
    for (const auto& prev : kept) {
      if (det.class_id == prev.class_id && Iou(det, prev) > threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(det);
      if (max_detections > 0 && static_cast<int>(kept.size()) >= max_detections) {
        break;
      }
    }
  }
  return kept;
}

float DflDistance(const std::vector<float>& box_output,
                  int hw,
                  int offset,
                  int side) {
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < kRegMax; ++i) {
    const float v = box_output[(side * kRegMax + i) * hw + offset];
    max_logit = std::max(max_logit, v);
  }

  float denom = 0.0f;
  float numer = 0.0f;
  for (int i = 0; i < kRegMax; ++i) {
    const float weight = std::exp(box_output[(side * kRegMax + i) * hw + offset] - max_logit);
    denom += weight;
    numer += weight * static_cast<float>(i);
  }
  return denom > 0.0f ? numer / denom : 0.0f;
}

void DecodeYoloHead(const std::vector<float>& box_output,
                    const std::vector<float>& cls_output,
                    const std::vector<float>& score_sum_output,
                    int grid,
                    int frame_w,
                    int frame_h,
                    const mediapipe_demo::PreprocessMeta& meta,
                    float conf_threshold,
                    std::vector<ObjectDetection>* detections) {
  if (detections == nullptr || grid <= 0) {
    return;
  }
  const int hw = grid * grid;
  if (hw <= 0 || cls_output.size() % static_cast<size_t>(hw) != 0) {
    return;
  }
  const int class_count = static_cast<int>(cls_output.size() / static_cast<size_t>(hw));
  if (box_output.size() < static_cast<size_t>(64 * hw) ||
      class_count <= 0 ||
      score_sum_output.size() < static_cast<size_t>(hw)) {
    return;
  }

  const float stride = static_cast<float>(kYoloInputSize) / static_cast<float>(grid);
  for (int y = 0; y < grid; ++y) {
    for (int x = 0; x < grid; ++x) {
      const int offset = y * grid + x;
      if (score_sum_output[offset] < conf_threshold) {
        continue;
      }

      int class_id = -1;
      float best_score = 0.0f;
      for (int c = 0; c < class_count; ++c) {
        const float score = cls_output[c * hw + offset];
        if (score > best_score) {
          best_score = score;
          class_id = c;
        }
      }
      if (best_score < conf_threshold) {
        continue;
      }

      const float left = DflDistance(box_output, hw, offset, 0);
      const float top = DflDistance(box_output, hw, offset, 1);
      const float right = DflDistance(box_output, hw, offset, 2);
      const float bottom = DflDistance(box_output, hw, offset, 3);

      const float anchor_x = static_cast<float>(x) + 0.5f;
      const float anchor_y = static_cast<float>(y) + 0.5f;
      const cv::Point2f p1((anchor_x - left) * stride, (anchor_y - top) * stride);
      const cv::Point2f p2((anchor_x + right) * stride, (anchor_y + bottom) * stride);
      const cv::Point2f src_p1 = mediapipe_demo::MapPointFromInputToSource(p1, meta);
      const cv::Point2f src_p2 = mediapipe_demo::MapPointFromInputToSource(p2, meta);

      ObjectDetection det;
      det.class_id = class_id;
      det.score = best_score;
      det.box.x1 = std::clamp(static_cast<int>(std::round(std::min(src_p1.x, src_p2.x))), 0, frame_w - 1);
      det.box.y1 = std::clamp(static_cast<int>(std::round(std::min(src_p1.y, src_p2.y))), 0, frame_h - 1);
      det.box.x2 = std::clamp(static_cast<int>(std::round(std::max(src_p1.x, src_p2.x))), 0, frame_w);
      det.box.y2 = std::clamp(static_cast<int>(std::round(std::max(src_p1.y, src_p2.y))), 0, frame_h);
      if (det.box.x2 > det.box.x1 && det.box.y2 > det.box.y1) {
        detections->push_back(det);
      }
    }
  }
}

int InferGridSize(const std::vector<float>& score_sum_output) {
  const int hw = static_cast<int>(score_sum_output.size());
  const int grid = static_cast<int>(std::round(std::sqrt(static_cast<float>(hw))));
  return grid > 0 && grid * grid == hw ? grid : 0;
}

}  // namespace

class YoloProcessor final : public IYoloProcessor {
 public:
  ~YoloProcessor() override { Stop(); }

  bool Start(const YoloProcessorConfig& config, std::string* err) override {
    Stop();
    config_ = config;
    model_ = std::make_unique<mediapipe_demo::RknnModel>();
    if (!model_->Load(config.model)) {
      if (err) {
        *err = "failed to load YOLO model: " + config.model;
      }
      model_.reset();
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

  std::optional<YoloResult> PollResult() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (results_.empty()) {
      return std::nullopt;
    }
    YoloResult result = std::move(results_.front());
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
    model_.reset();
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

      YoloResult result = ProcessFrame(frame);
      {
        std::lock_guard<std::mutex> lock(mu_);
        results_.push_back(std::move(result));
        while (results_.size() > std::max<size_t>(1, config_.queue_depth * 2)) {
          results_.pop_front();
        }
      }
    }
  }

  YoloResult ProcessFrame(const FrameRef& frame) {
    const auto started = std::chrono::steady_clock::now();

    YoloResult result;
    result.camera_id = frame.camera_id;
    result.pts_ns = frame.pts_ns;
    result.frame_width = frame.width;
    result.frame_height = frame.height;

    mediapipe_demo::PreprocessMeta meta;
    std::vector<std::vector<float>> outputs;
    if (!model_ || !InferWithHardwarePreprocess(frame, &meta, &outputs)) {
      cv::Mat rgb = ToRgbMat(frame);
      if (rgb.empty()) {
        result.error = "empty YOLO input frame";
        return result;
      }

      cv::Mat input = mediapipe_demo::LetterboxPadding(
          rgb, cv::Size(kYoloInputSize, kYoloInputSize), &meta);
      if (!input.isContinuous()) {
        input = input.clone();
      }

      if (!model_ || !model_->Infer(input, &outputs)) {
        result.error = "YOLO RKNN inference failed";
        return result;
      }
    }
    if (outputs.size() != kHeadCount * 3) {
      result.error = "YOLO RKNN inference failed";
      return result;
    }

    std::vector<ObjectDetection> detections;
    for (int head = 0; head < kHeadCount; ++head) {
      const int base = head * 3;
      const int grid = InferGridSize(outputs[base + 2]);
      DecodeYoloHead(outputs[base], outputs[base + 1], outputs[base + 2],
                     grid, frame.width, frame.height, meta,
                     config_.confidence_threshold, &detections);
    }

    result.detections = Nms(std::move(detections),
                            config_.nms_threshold,
                            config_.max_detections);
    const auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    result.fps = elapsed.count() > 0.0 ? static_cast<float>(1.0 / elapsed.count()) : 0.0f;
    result.ok = true;
    return result;
  }

  bool InferWithHardwarePreprocess(const FrameRef& frame,
                                   mediapipe_demo::PreprocessMeta* meta,
                                   std::vector<std::vector<float>>* outputs) {
    if (!model_ || meta == nullptr || outputs == nullptr ||
        frame.pixel_format != PixelFormat::kNv12 ||
        frame.dmabuf_fd < 0 || frame.fourcc == 0 ||
        frame.width <= 0 || frame.height <= 0 || frame.stride <= 0) {
      return false;
    }

    mediapipe_demo::CameraFrame camera_frame;
    camera_frame.width = frame.width;
    camera_frame.height = frame.height;
    camera_frame.stride = frame.stride;
    camera_frame.fourcc = frame.fourcc;
    camera_frame.bytes_used = frame.bytes_used;
    camera_frame.dmabuf_fd = frame.dmabuf_fd;

    rknn_tensor_mem* input_mem = model_->InputMemory();
    if (input_mem == nullptr ||
        !mediapipe_demo::PreprocessFrameToRknn(
            camera_frame,
            cv::Rect(0, 0, frame.width, frame.height),
            true,
            input_mem,
            model_->InputAttr(),
            meta)) {
      return false;
    }

    return model_->SyncInputMemory() && model_->Run(outputs);
  }

  YoloProcessorConfig config_;
  std::unique_ptr<mediapipe_demo::RknnModel> model_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<FrameRef> pending_frames_;
  std::deque<YoloResult> results_;
  bool running_ = false;
  std::thread worker_;
};

std::unique_ptr<IYoloProcessor> CreateYoloProcessor() {
  return std::make_unique<YoloProcessor>();
}

}  // namespace rkstudio::vision
