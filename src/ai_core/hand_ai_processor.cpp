#include "rk_studio/ai_core/ai_processor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "mediapipe/common/config.h"
#include "mediapipe/common/types.h"
#include "mediapipe/detector/palm_detector.h"
#include "mediapipe/landmark/hand_landmark.h"
#include "mediapipe/preprocess/hw_preprocess.h"
#include "mediapipe/preprocess/image_ops.h"
#include "mediapipe/tracking/hand_tracker.h"

namespace rkstudio::ai {
namespace {

constexpr int kMaxHands = 2;

std::vector<cv::Point2f> ExtractLandmarkPoints(const mediapipe_demo::HandLandmarks& landmarks) {
  std::vector<cv::Point2f> points;
  points.reserve(landmarks.points.size());
  for (const auto& point : landmarks.points) {
    points.emplace_back(point.x, point.y);
  }
  return points;
}

float EstimateHandAngleDeg(const std::vector<cv::Point2f>& landmarks_xy) {
  if (landmarks_xy.size() < 10) {
    return 0.0f;
  }
  const cv::Point2f v = landmarks_xy[9] - landmarks_xy[0];
  return static_cast<float>(std::atan2(v.y, v.x) * 180.0 / CV_PI);
}

float RotationToVerticalDeg(float angle_deg) {
  float rotation = angle_deg + 90.0f;
  while (rotation > 180.0f) {
    rotation -= 360.0f;
  }
  while (rotation < -180.0f) {
    rotation += 360.0f;
  }
  return rotation;
}

bool PointInImage(const cv::Point2f& point, const cv::Mat& image) {
  return point.x >= 0.0f && point.x < static_cast<float>(image.cols) &&
         point.y >= 0.0f && point.y < static_cast<float>(image.rows);
}

TrackingMode ConvertMode(mediapipe_demo::TrackingMode mode) {
  switch (mode) {
    case mediapipe_demo::TrackingMode::kDetect:
      return TrackingMode::kDetect;
    case mediapipe_demo::TrackingMode::kTrack:
      return TrackingMode::kTrack;
    case mediapipe_demo::TrackingMode::kRecover:
      return TrackingMode::kRecover;
    case mediapipe_demo::TrackingMode::kNoHand:
    default:
      return TrackingMode::kNoHand;
  }
}

mediapipe_demo::CameraFrame ToCameraFrame(const FrameRef& frame) {
  mediapipe_demo::CameraFrame cf;
  cf.width = frame.width;
  cf.height = frame.height;
  cf.stride = frame.stride;
  cf.fourcc = frame.fourcc;
  cf.bytes_used = frame.bytes_used;
  cf.dmabuf_fd = frame.dmabuf_fd;
  cf.data = const_cast<uint8_t*>(frame.mapped_ptr);
  return cf;
}

cv::Mat ToBgrMat(const FrameRef& frame) {
  if (frame.mapped_ptr == nullptr || frame.width <= 0 || frame.height <= 0 || frame.stride <= 0) {
    return {};
  }

  if (frame.pixel_format == PixelFormat::kNv12) {
    cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1,
                 const_cast<uint8_t*>(frame.mapped_ptr), frame.stride);
    cv::Mat bgr;
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    return bgr;
  }
  if (frame.pixel_format == PixelFormat::kBgr) {
    return cv::Mat(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.mapped_ptr), frame.stride);
  }
  if (frame.pixel_format == PixelFormat::kRgb) {
    cv::Mat rgb(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.mapped_ptr), frame.stride);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  return {};
}

cv::Point2f RoiCenter(const mediapipe_demo::RoiRect& roi) {
  return {static_cast<float>(roi.x1 + roi.x2) * 0.5f, static_cast<float>(roi.y1 + roi.y2) * 0.5f};
}

float PointDistSq(const cv::Point2f& a, const cv::Point2f& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}

// Greedy match: assign detections to trackers by nearest ROI center.
// assignments[tracker_id] = index into detections, or -1 if unmatched.
std::array<int, kMaxHands> MatchDetectionsToTrackers(
    const std::vector<mediapipe_demo::PalmDetection>& detections,
    const std::vector<mediapipe_demo::RoiRect>& det_rois,
    const std::array<std::unique_ptr<mediapipe_demo::HandTracker>, kMaxHands>& trackers) {
  std::array<int, kMaxHands> assignments;
  assignments.fill(-1);
  std::vector<bool> det_used(detections.size(), false);

  // First pass: match trackers that have a current ROI to nearest detection
  for (int t = 0; t < kMaxHands; ++t) {
    const auto tracker_roi = trackers[t]->CurrentRoi();
    if (!tracker_roi.has_value()) {
      continue;
    }
    const cv::Point2f tc = RoiCenter(*tracker_roi);
    float best_dist = std::numeric_limits<float>::max();
    int best_d = -1;
    for (size_t d = 0; d < det_rois.size(); ++d) {
      if (det_used[d]) continue;
      const float dist = PointDistSq(tc, RoiCenter(det_rois[d]));
      if (dist < best_dist) {
        best_dist = dist;
        best_d = static_cast<int>(d);
      }
    }
    if (best_d >= 0) {
      assignments[t] = best_d;
      det_used[static_cast<size_t>(best_d)] = true;
    }
  }

  // Second pass: assign remaining detections to idle trackers
  for (size_t d = 0; d < det_rois.size(); ++d) {
    if (det_used[d]) continue;
    for (int t = 0; t < kMaxHands; ++t) {
      if (assignments[t] >= 0) continue;
      if (trackers[t]->CurrentRoi().has_value()) continue;
      assignments[t] = static_cast<int>(d);
      det_used[d] = true;
      break;
    }
  }

  return assignments;
}

}  // namespace

class HandAiProcessor final : public IAiProcessor {
 public:
  ~HandAiProcessor() override { Stop(); }

  bool Start(const AiProcessorConfig& config, std::string* err) override {
    Stop();

    config_ = config;
    pipeline_config_ = mediapipe_demo::PipelineConfig{};
    for (int i = 0; i < kMaxHands; ++i) {
      trackers_[i] = std::make_unique<mediapipe_demo::HandTracker>(pipeline_config_);
    }
    if (!detector_.LoadModel(config.detector_model)) {
      if (err) {
        *err = "failed to load detector model: " + config.detector_model;
      }
      for (auto& t : trackers_) t.reset();
      return false;
    }
    if (!landmark_.LoadModel(config.landmark_model)) {
      if (err) {
        *err = "failed to load landmark model: " + config.landmark_model;
      }
      for (auto& t : trackers_) t.reset();
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

  std::optional<AiResult> PollResult() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (results_.empty()) {
      return std::nullopt;
    }
    AiResult result = std::move(results_.front());
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
    for (auto& t : trackers_) t.reset();
    frame_index_ = 1;
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

      AiResult result = ProcessFrame(frame);
      {
        std::lock_guard<std::mutex> lock(mu_);
        results_.push_back(std::move(result));
        while (results_.size() > std::max<size_t>(1, config_.queue_depth * 2)) {
          results_.pop_front();
        }
      }
    }
  }

  HandResult ProcessOneHand(int hand_id,
                            mediapipe_demo::HandTracker& tracker,
                            mediapipe_demo::TrackingMode frame_mode,
                            const FrameRef& frame,
                            const mediapipe_demo::CameraFrame& camera_frame,
                            bool rga_available,
                            std::function<cv::Mat&()> ensure_bgr) {
    HandResult hand;
    hand.hand_id = hand_id;

    const std::optional<mediapipe_demo::RoiRect> current_roi = tracker.CurrentRoi();
    hand.tracking_mode = ConvertMode(frame_mode);

    if (!current_roi.has_value()) {
      tracker.MarkLost();
      hand.tracking_mode = TrackingMode::kNoHand;
      return hand;
    }

    mediapipe_demo::RoiRect roi_rect = *current_roi;
    hand.roi = RoiRect{roi_rect.x1, roi_rect.y1, roi_rect.x2, roi_rect.y2};

    const cv::Rect roi_cv(roi_rect.x1, roi_rect.y1, roi_rect.x2 - roi_rect.x1,
                           roi_rect.y2 - roi_rect.y1);
    const cv::Size landmark_size(224, 224);

    bool use_cpu_landmark_path = false;
    cv::Mat roi_for_landmark;
    cv::Mat inverse_affine;
    float align_rotation_deg = 0.0f;

    const auto& prev_landmarks = tracker.LastGoodLandmarks();
    if (pipeline_config_.enable_affine_align && prev_landmarks.size() == 21 &&
        !(pipeline_config_.affine_disable_on_fast_motion && tracker.FastMotionCooldown() > 0)) {
      std::vector<cv::Point2f> prev_local = prev_landmarks;
      for (auto& point : prev_local) {
        point.x -= static_cast<float>(roi_rect.x1);
        point.y -= static_cast<float>(roi_rect.y1);
      }

      cv::Mat roi_check = ensure_bgr();
      if (!roi_check.empty() &&
          roi_cv.x >= 0 && roi_cv.y >= 0 &&
          roi_cv.x + roi_cv.width <= roi_check.cols &&
          roi_cv.y + roi_cv.height <= roi_check.rows &&
          PointInImage(prev_local[0], roi_check(roi_cv)) &&
          PointInImage(prev_local[9], roi_check(roi_cv))) {
        const float angle_deg = EstimateHandAngleDeg(prev_local);
        align_rotation_deg = std::clamp(RotationToVerticalDeg(angle_deg),
                                        -pipeline_config_.affine_max_abs_deg,
                                        pipeline_config_.affine_max_abs_deg);
        if (std::abs(align_rotation_deg) > 1.0f) {
          cv::Mat roi = ensure_bgr()(roi_cv).clone();
          roi_for_landmark = mediapipe_demo::RotateRoi(roi, align_rotation_deg, &inverse_affine);
          use_cpu_landmark_path = true;
        }
      }
    }

    mediapipe_demo::PreprocessMeta lm_meta;
    std::optional<mediapipe_demo::HandLandmarks> landmarks;

    if (!use_cpu_landmark_path && rga_available &&
        mediapipe_demo::PreprocessFrameToRknn(camera_frame, roi_cv, false,
                                              landmark_.InputMemory(), landmark_.InputAttr(),
                                              &lm_meta) &&
        landmark_.SyncInputMemory()) {
      landmarks = landmark_.InferPrepared(lm_meta);
    } else {
      if (roi_for_landmark.empty()) {
        cv::Mat& bgr_frame = ensure_bgr();
        if (bgr_frame.empty() ||
            roi_cv.x < 0 || roi_cv.y < 0 ||
            roi_cv.x + roi_cv.width > bgr_frame.cols ||
            roi_cv.y + roi_cv.height > bgr_frame.rows) {
          tracker.MarkLost();
          hand.tracking_mode = TrackingMode::kNoHand;
          return hand;
        }
        roi_for_landmark = bgr_frame(roi_cv).clone();
      }
      cv::Mat lm_input = mediapipe_demo::PreprocessBgr(roi_for_landmark, landmark_size, &lm_meta);
      landmarks = landmark_.Infer(lm_input, lm_meta);
    }

    if (!landmarks.has_value()) {
      tracker.MarkLost();
    } else {
      std::vector<cv::Point2f> roi_points = ExtractLandmarkPoints(*landmarks);
      if (!inverse_affine.empty()) {
        roi_points = mediapipe_demo::AffinePoints(roi_points, inverse_affine);
        for (auto& point : roi_points) {
          point.x = std::clamp(point.x, 0.0f, static_cast<float>(roi_cv.width - 1));
          point.y = std::clamp(point.y, 0.0f, static_cast<float>(roi_cv.height - 1));
        }
      }

      std::vector<cv::Point2f> global_points = roi_points;
      for (auto& point : global_points) {
        point.x += static_cast<float>(roi_rect.x1);
        point.y += static_cast<float>(roi_rect.y1);
      }

      float motion_norm = 0.0f;
      if (tracker.AcceptLandmarks(&global_points, roi_rect, frame.width, frame.height, &motion_norm)) {
        for (size_t i = 0; i < landmarks->points.size(); ++i) {
          landmarks->points[i].x = global_points[i].x;
          landmarks->points[i].y = global_points[i].y;
        }
        hand.landmarks.reserve(landmarks->points.size());
        for (const auto& point : landmarks->points) {
          hand.landmarks.push_back(Landmark3f{point.x, point.y, point.z});
        }
        hand.motion_norm = motion_norm;
        hand.rotation_deg = align_rotation_deg;
        hand.fast_motion_cooldown = tracker.FastMotionCooldown();
        hand.tracking_mode = ConvertMode(frame_mode);
      } else {
        hand.tracking_mode = tracker.CurrentRoi().has_value() ? TrackingMode::kTrack : TrackingMode::kNoHand;
      }
    }

    return hand;
  }

  AiResult ProcessFrame(const FrameRef& frame) {
    auto started = std::chrono::steady_clock::now();

    const bool rga_available = config_.allow_rga && frame.dmabuf_fd >= 0;
    mediapipe_demo::CameraFrame camera_frame;
    if (rga_available) {
      camera_frame = ToCameraFrame(frame);
    }

    cv::Mat bgr;
    auto ensure_bgr = [&]() -> cv::Mat& {
      if (bgr.empty()) {
        bgr = ToBgrMat(frame);
      }
      return bgr;
    };

    const cv::Size detector_size(192, 192);
    AiResult result;
    result.camera_id = frame.camera_id;
    result.pts_ns = frame.pts_ns;
    result.frame_width = frame.width;
    result.frame_height = frame.height;

    // --- Detection phase: get up to 2 detections ---
    bool any_should_detect = false;
    for (int i = 0; i < kMaxHands; ++i) {
      if (trackers_[i]->ShouldRunDetector(frame_index_)) {
        any_should_detect = true;
        break;
      }
    }

    std::vector<mediapipe_demo::PalmDetection> detections;
    std::vector<mediapipe_demo::RoiRect> det_rois;

    if (any_should_detect) {
      mediapipe_demo::PreprocessMeta det_meta;
      const cv::Rect full_frame_rect(0, 0, frame.width, frame.height);

      if (rga_available &&
          mediapipe_demo::PreprocessFrameToRknn(camera_frame, full_frame_rect, true,
                                                detector_.InputMemory(), detector_.InputAttr(),
                                                &det_meta) &&
          detector_.SyncInputMemory()) {
        detections = detector_.InferPreparedMulti(pipeline_config_.detector_score_threshold, kMaxHands);
      } else {
        cv::Mat& bgr_frame = ensure_bgr();
        if (!bgr_frame.empty()) {
          cv::Mat det_input = mediapipe_demo::PreprocessBgr(bgr_frame, detector_size, &det_meta);
          detections = detector_.InferMulti(det_input, det_meta, pipeline_config_.detector_score_threshold, kMaxHands);
        }
      }

      for (const auto& det : detections) {
        auto roi = mediapipe_demo::MakeRoiFromDetection(det.bbox, det_meta, frame.width, frame.height,
                                                         pipeline_config_.det_scale);
        if (roi.has_value()) {
          det_rois.push_back(*roi);
        } else {
          det_rois.push_back({});  // placeholder
        }
      }

      // Match detections to trackers
      auto assignments = MatchDetectionsToTrackers(detections, det_rois, trackers_);
      for (int t = 0; t < kMaxHands; ++t) {
        if (assignments[t] >= 0) {
          const size_t d = static_cast<size_t>(assignments[t]);
          trackers_[t]->UpdateFromDetection(det_rois[d], detections[d].score);
        }
      }
    }

    // --- Landmark phase: process each tracked hand ---
    for (int i = 0; i < kMaxHands; ++i) {
      const auto current_roi = trackers_[i]->CurrentRoi();
      if (!current_roi.has_value()) {
        continue;
      }
      mediapipe_demo::TrackingMode frame_mode = mediapipe_demo::TrackingMode::kTrack;
      HandResult hand = ProcessOneHand(i, *trackers_[i], frame_mode, frame, camera_frame, rga_available, ensure_bgr);
      if (hand.tracking_mode != TrackingMode::kNoHand || !hand.landmarks.empty()) {
        result.hands.push_back(std::move(hand));
      }
    }

    auto finished = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finished - started;
    result.fps = elapsed.count() > 0.0 ? static_cast<float>(1.0 / elapsed.count()) : 0.0f;
    result.ok = true;
    ++frame_index_;
    return result;
  }

  AiProcessorConfig config_;
  mediapipe_demo::PipelineConfig pipeline_config_;
  mediapipe_demo::PalmDetector detector_;
  mediapipe_demo::HandLandmark landmark_;
  std::array<std::unique_ptr<mediapipe_demo::HandTracker>, kMaxHands> trackers_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<FrameRef> pending_frames_;
  std::deque<AiResult> results_;
  bool running_ = false;
  std::thread worker_;
  int frame_index_ = 1;
};

std::unique_ptr<IAiProcessor> CreateHandAiProcessor() {
  return std::make_unique<HandAiProcessor>();
}

}  // namespace rkstudio::ai
