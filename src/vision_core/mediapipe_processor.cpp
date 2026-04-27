#include "rk_studio/vision_core/vision_processor.h"

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
#include "mediapipe/preprocess/image_ops.h"
#include "mediapipe/tracking/hand_tracker.h"

namespace rkstudio::vision {
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

cv::Mat ToRgbMat(const FrameRef& frame) {
  if (frame.mapped_ptr == nullptr || frame.width <= 0 || frame.height <= 0 || frame.stride <= 0) {
    return {};
  }

  if (frame.pixel_format == PixelFormat::kRgb) {
    return cv::Mat(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.mapped_ptr), frame.stride);
  }
  if (frame.pixel_format == PixelFormat::kNv12) {
    cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1,
                 const_cast<uint8_t*>(frame.mapped_ptr), frame.stride);
    cv::Mat rgb;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    return rgb;
  }
  if (frame.pixel_format == PixelFormat::kBgr) {
    cv::Mat bgr(frame.height, frame.width, CV_8UC3, const_cast<uint8_t*>(frame.mapped_ptr), frame.stride);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
  }
  return {};
}

float PointDistSq(const cv::Point2f& a, const cv::Point2f& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return dx * dx + dy * dy;
}

float PointDist(const cv::Point2f& a, const cv::Point2f& b) {
  return std::sqrt(PointDistSq(a, b));
}

float Dot(const cv::Point2f& a, const cv::Point2f& b) {
  return a.x * b.x + a.y * b.y;
}

float ProjectFromWrist(const std::vector<cv::Point2f>& points, int index, const cv::Point2f& axis) {
  return Dot(points[index] - points[0], axis);
}

float RecognizeThumbsUpScore(const std::vector<cv::Point2f>& points) {
  if (points.size() < 21) {
    return 0.0f;
  }

  const cv::Point2f wrist = points[0];
  const cv::Point2f middle_mcp = points[9];
  const float palm_len = PointDist(wrist, middle_mcp);
  const float palm_width = PointDist(points[5], points[17]);
  const float palm_size = std::max({palm_len, palm_width, 1.0f});
  cv::Point2f palm_axis = middle_mcp - wrist;
  const float axis_len = std::max(PointDist(wrist, middle_mcp), 1.0f);
  palm_axis.x /= axis_len;
  palm_axis.y /= axis_len;

  const float thumb_projection = Dot(points[4] - points[2], palm_axis);
  const bool thumb_extended =
      thumb_projection > 0.32f * palm_size &&
      ProjectFromWrist(points, 4, palm_axis) > ProjectFromWrist(points, 3, palm_axis) + 0.08f * palm_size &&
      PointDist(points[4], points[2]) > 0.35f * palm_size;
  const bool thumb_points_up = points[4].y < points[2].y - 0.12f * palm_size &&
                               points[4].y < wrist.y;

  constexpr std::array<std::array<int, 3>, 4> kFingers = {{
      {{5, 6, 8}},
      {{9, 10, 12}},
      {{13, 14, 16}},
      {{17, 18, 20}},
  }};

  int folded_count = 0;
  for (const auto& finger : kFingers) {
    const int mcp = finger[0];
    const int pip = finger[1];
    const int tip = finger[2];
    const float tip_proj = ProjectFromWrist(points, tip, palm_axis);
    const float pip_proj = ProjectFromWrist(points, pip, palm_axis);
    const bool folded_by_axis = tip_proj < pip_proj + 0.08f * palm_size;
    const bool folded_near_palm = PointDist(points[tip], points[mcp]) < 0.72f * palm_size;
    if (folded_by_axis || folded_near_palm) {
      ++folded_count;
    }
  }

  if (!thumb_extended || !thumb_points_up || folded_count < 3) {
    return 0.0f;
  }

  float score = 0.45f;
  score += 0.12f * static_cast<float>(folded_count);
  score += std::min(0.20f, std::max(0.0f, (thumb_projection - 0.32f * palm_size) / palm_size));
  if (folded_count == 4) {
    score += 0.08f;
  }
  return std::clamp(score, 0.0f, 1.0f);
}

}  // namespace

class MediapipeProcessor final : public IMediapipeProcessor {
 public:
  ~MediapipeProcessor() override { Stop(); }

  bool Start(const MediapipeProcessorConfig& config, std::string* err) override {
    Stop();

    config_ = config;
    pipeline_config_ = mediapipe_demo::PipelineConfig{};
    if (!detector_.LoadModel(config.detector_model)) {
      if (err) {
        *err = "failed to load detector model: " + config.detector_model;
      }
      return false;
    }
    if (!landmark_.LoadModel(config.landmark_model)) {
      if (err) {
        *err = "failed to load landmark model: " + config.landmark_model;
      }
      return false;
    }

    running_ = true;
    worker_ = std::thread([this] { RunLoop(); });
    return true;
  }

  void Submit(const VisionFrame& frame) override {
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

  std::optional<MediapipeResult> PollResult() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (results_.empty()) {
      return std::nullopt;
    }
    MediapipeResult result = std::move(results_.front());
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
    frame_index_ = 1;
  }

 private:
  void RunLoop() {
    while (true) {
      VisionFrame frame;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !running_ || !pending_frames_.empty(); });
        if (!running_ && pending_frames_.empty()) {
          break;
        }
        frame = std::move(pending_frames_.front());
        pending_frames_.pop_front();
      }

      MediapipeResult result = ProcessFrame(frame);
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
                            std::function<cv::Mat&()> ensure_rgb) {
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

      cv::Mat roi_check = ensure_rgb();
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
          cv::Mat roi = ensure_rgb()(roi_cv).clone();
          roi_for_landmark = mediapipe_demo::RotateRoi(roi, align_rotation_deg, &inverse_affine);
        }
      }
    }

    mediapipe_demo::PreprocessMeta lm_meta;
    std::optional<mediapipe_demo::HandLandmarks> landmarks;

    if (roi_for_landmark.empty()) {
      cv::Mat& rgb_frame = ensure_rgb();
      if (rgb_frame.empty() ||
          roi_cv.x < 0 || roi_cv.y < 0 ||
          roi_cv.x + roi_cv.width > rgb_frame.cols ||
          roi_cv.y + roi_cv.height > rgb_frame.rows) {
        tracker.MarkLost();
        hand.tracking_mode = TrackingMode::kNoHand;
        return hand;
      }
      roi_for_landmark = rgb_frame(roi_cv).clone();
    }
    cv::Mat lm_input = mediapipe_demo::LetterboxPadding(roi_for_landmark, landmark_size, &lm_meta);
    landmarks = landmark_.Infer(lm_input, lm_meta);

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
        hand.gesture_score = RecognizeThumbsUpScore(global_points);
        if (hand.gesture_score > 0.0f) {
          hand.gesture = "thumbs_up";
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

  MediapipeResult ProcessFrame(const VisionFrame& input) {
    auto started = std::chrono::steady_clock::now();
    const FrameRef& frame = input.rgb;

    cv::Mat rgb;
    auto ensure_rgb = [&]() -> cv::Mat& {
      if (rgb.empty()) {
        rgb = ToRgbMat(frame);
      }
      return rgb;
    };

    const cv::Size detector_size(192, 192);
    MediapipeResult result;
    result.camera_id = frame.camera_id;
    result.pts_ns = frame.pts_ns;
    result.frame_width = frame.width;
    result.frame_height = frame.height;

    std::vector<mediapipe_demo::PalmDetection> detections;
    std::vector<mediapipe_demo::RoiRect> det_rois;
    std::vector<float> det_scores;

    mediapipe_demo::PreprocessMeta det_meta;

    cv::Mat& rgb_frame = ensure_rgb();
    if (!rgb_frame.empty()) {
      cv::Mat det_input = mediapipe_demo::LetterboxPadding(rgb_frame, detector_size, &det_meta);
      detections = detector_.InferMulti(det_input, det_meta, pipeline_config_.detector_score_threshold, kMaxHands);
    }

    for (const auto& det : detections) {
      auto roi = mediapipe_demo::MakeRoiFromDetection(det.bbox, det_meta, frame.width, frame.height,
                                                       pipeline_config_.det_scale);
      if (roi.has_value()) {
        det_rois.push_back(*roi);
        det_scores.push_back(det.score);
      }
    }

    for (size_t i = 0; i < det_rois.size(); ++i) {
      mediapipe_demo::HandTracker tracker(pipeline_config_);
      tracker.UpdateFromDetection(det_rois[i], det_scores[i]);
      HandResult hand = ProcessOneHand(
          static_cast<int>(i), tracker, mediapipe_demo::TrackingMode::kDetect, frame, ensure_rgb);
      if (!hand.landmarks.empty()) {
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

  MediapipeProcessorConfig config_;
  mediapipe_demo::PipelineConfig pipeline_config_;
  mediapipe_demo::PalmDetector detector_;
  mediapipe_demo::HandLandmark landmark_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<VisionFrame> pending_frames_;
  std::deque<MediapipeResult> results_;
  bool running_ = false;
  std::thread worker_;
  int frame_index_ = 1;
};

std::unique_ptr<IMediapipeProcessor> CreateMediapipeProcessor() {
  return std::make_unique<MediapipeProcessor>();
}

}  // namespace rkstudio::vision
