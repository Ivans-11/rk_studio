#include "rk_studio/media_core/rtsp_server.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>

#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/video/video.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "mediapipe/preprocess/hw_preprocess.h"
#include "rk_studio/media_core/v4l2_pipeline.h"

namespace rkstudio::media {
namespace {

std::string Uppercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

int AlignUp(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

int ExtractDmabufFd(GstBuffer* buffer) {
  if (buffer == nullptr) {
    return -1;
  }
  GstMemory* mem = gst_buffer_peek_memory(buffer, 0);
  if (mem != nullptr && gst_is_dmabuf_memory(mem)) {
    return gst_dmabuf_memory_get_fd(mem);
  }
  return -1;
}

std::shared_ptr<GstSample> HoldSampleRef(GstSample* sample) {
  if (sample == nullptr) {
    return {};
  }
  return std::shared_ptr<GstSample>(gst_sample_ref(sample), [](GstSample* ptr) {
    if (ptr != nullptr) {
      gst_sample_unref(ptr);
    }
  });
}

void AppendEncoderTail(std::ostringstream& ss, const std::string& codec, int gop, int bitrate) {
  const bool h265 = Uppercase(codec) == "H265";
  const std::string encoder = h265 ? "mpph265enc" : "mpph264enc";
  const std::string parser = h265 ? "h265parse" : "h264parse";
  const std::string payloader = h265 ? "rtph265pay" : "rtph264pay";

  ss << "! " << encoder << " bps=" << bitrate << " gop=" << gop << " header-mode=1 "
     << "! " << parser << " "
     << "! " << payloader << " config-interval=-1 name=pay0 pt=96 ";
}

std::string BuildCameraAppSrcLaunchString(int width,
                                          int height,
                                          int fps,
                                          const std::string& codec,
                                          int gop,
                                          int bitrate) {
  std::ostringstream ss;
  ss << "( appsrc name=src is-live=true format=time do-timestamp=false block=false "
     << "! video/x-raw,format=NV12,width=" << width << ",height=" << height
     << ",framerate=" << fps << "/1 "
     << "! queue leaky=downstream max-size-buffers=2 ";
  AppendEncoderTail(ss, codec, gop, bitrate);
  ss << ")";
  return ss.str();
}

std::string NormalizeMount(std::string mount) {
  while (!mount.empty() && mount.front() == '/') {
    mount.erase(mount.begin());
  }
  return mount;
}

int ClampInt(int value, int min_value, int max_value) {
  return std::clamp(value, min_value, max_value);
}

cv::Point MapOverlayPoint(float x,
                          float y,
                          int source_w,
                          int source_h,
                          int target_w,
                          int target_h) {
  const float scale_x = target_w > 0 && source_w > 0
                            ? static_cast<float>(target_w) / source_w
                            : 1.0f;
  const float scale_y = target_h > 0 && source_h > 0
                            ? static_cast<float>(target_h) / source_h
                            : 1.0f;
  return cv::Point(
      ClampInt(static_cast<int>(std::round(x * scale_x)), 0, std::max(0, target_w - 1)),
      ClampInt(static_cast<int>(std::round(y * scale_y)), 0, std::max(0, target_h - 1)));
}

cv::Rect MapOverlayRect(const vision::RoiRect& rect,
                        int source_w,
                        int source_h,
                        int target_w,
                        int target_h) {
  const cv::Point p1 = MapOverlayPoint(static_cast<float>(rect.x1),
                                       static_cast<float>(rect.y1),
                                       source_w,
                                       source_h,
                                       target_w,
                                       target_h);
  const cv::Point p2 = MapOverlayPoint(static_cast<float>(rect.x2),
                                       static_cast<float>(rect.y2),
                                       source_w,
                                       source_h,
                                       target_w,
                                       target_h);
  return cv::Rect(p1, p2) & cv::Rect(0, 0, target_w, target_h);
}

void DrawLabel(cv::Mat& rgb,
               const std::string& text,
               const cv::Point& anchor,
               const cv::Scalar& color) {
  if (text.empty()) {
    return;
  }

  int baseline = 0;
  const double font_scale = 0.45;
  const int thickness = 1;
  const cv::Size text_size =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
  const int x = ClampInt(anchor.x, 0, std::max(0, rgb.cols - text_size.width - 4));
  int y = anchor.y - text_size.height - 6;
  if (y < 0) {
    y = ClampInt(anchor.y + text_size.height + 6, text_size.height + 4, rgb.rows - 1);
  }

  cv::Rect box(x, y - text_size.height - 3, text_size.width + 6, text_size.height + 6);
  box &= cv::Rect(0, 0, rgb.cols, rgb.rows);
  if (box.width > 0 && box.height > 0) {
    cv::rectangle(rgb, box, color, cv::FILLED);
  }
  cv::putText(rgb,
              text,
              cv::Point(x + 3, y),
              cv::FONT_HERSHEY_SIMPLEX,
              font_scale,
              cv::Scalar(0, 0, 0),
              thickness,
              cv::LINE_AA);
}

void DrawMediapipeOverlay(cv::Mat& rgb, const vision::MediapipeResult& result) {
  if (!result.ok || result.hands.empty()) {
    return;
  }

  static constexpr std::array<std::array<int, 2>, 24> kHandEdges{{
      {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 4}},
      {{0, 5}}, {{5, 6}}, {{6, 7}}, {{7, 8}},
      {{0, 9}}, {{9, 10}}, {{10, 11}}, {{11, 12}},
      {{0, 13}}, {{13, 14}}, {{14, 15}}, {{15, 16}},
      {{0, 17}}, {{17, 18}}, {{18, 19}}, {{19, 20}},
      {{5, 9}}, {{9, 13}}, {{13, 17}}, {{5, 17}},
  }};
  static const cv::Scalar kRoiColors[] = {
      cv::Scalar(255, 160, 0), cv::Scalar(0, 160, 255)};
  static const cv::Scalar kLandmarkColors[] = {
      cv::Scalar(0, 255, 100), cv::Scalar(255, 100, 200)};

  const int source_w = result.frame_width > 0 ? result.frame_width : rgb.cols;
  const int source_h = result.frame_height > 0 ? result.frame_height : rgb.rows;
  for (size_t i = 0; i < result.hands.size(); ++i) {
    const auto& hand = result.hands[i];
    const cv::Scalar roi_color = kRoiColors[i % 2];
    const cv::Scalar point_color = kLandmarkColors[i % 2];

    if (hand.roi.has_value()) {
      const cv::Rect roi =
          MapOverlayRect(*hand.roi, source_w, source_h, rgb.cols, rgb.rows);
      if (roi.width > 0 && roi.height > 0) {
        cv::rectangle(rgb, roi, roi_color, 2, cv::LINE_AA);
        if (!hand.gesture.empty()) {
          std::ostringstream label;
          label.setf(std::ios::fixed);
          label.precision(2);
          label << hand.gesture << " " << hand.gesture_score;
          DrawLabel(rgb, label.str(), roi.tl(), roi_color);
        }
      }
    }

    std::vector<cv::Point> points;
    points.reserve(hand.landmarks.size());
    for (const auto& landmark : hand.landmarks) {
      points.push_back(MapOverlayPoint(landmark.x,
                                       landmark.y,
                                       source_w,
                                       source_h,
                                       rgb.cols,
                                       rgb.rows));
    }

    for (const auto& edge : kHandEdges) {
      if (edge[0] >= static_cast<int>(points.size()) ||
          edge[1] >= static_cast<int>(points.size())) {
        continue;
      }
      cv::line(rgb, points[edge[0]], points[edge[1]], roi_color, 2, cv::LINE_AA);
    }
    for (const auto& point : points) {
      cv::circle(rgb, point, 3, point_color, cv::FILLED, cv::LINE_AA);
    }
  }
}

void DrawYoloOverlay(cv::Mat& rgb, const vision::YoloResult& result) {
  if (!result.ok || result.detections.empty()) {
    return;
  }

  static const cv::Scalar kBoxColors[] = {
      cv::Scalar(0, 220, 140),
      cv::Scalar(255, 180, 0),
      cv::Scalar(0, 180, 255),
      cv::Scalar(255, 90, 160),
  };

  const int source_w = result.frame_width > 0 ? result.frame_width : rgb.cols;
  const int source_h = result.frame_height > 0 ? result.frame_height : rgb.rows;
  for (size_t i = 0; i < result.detections.size(); ++i) {
    const auto& det = result.detections[i];
    const cv::Scalar color = kBoxColors[i % 4];
    const cv::Rect box =
        MapOverlayRect(det.box, source_w, source_h, rgb.cols, rgb.rows);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }
    cv::rectangle(rgb, box, color, 2, cv::LINE_AA);

    std::ostringstream label;
    label.setf(std::ios::fixed);
    label.precision(2);
    label << (det.class_name.empty() ? ("#" + std::to_string(det.class_id))
                                     : det.class_name)
          << " " << det.score;
    DrawLabel(rgb, label.str(), box.tl(), color);
  }
}

}  // namespace

struct MosaicFrame {
  bool valid = false;
  int width = 0;
  int height = 0;
  int stride = 0;
  int dmabuf_fd = -1;
  cv::Mat nv12;
  std::shared_ptr<GstSample> sample_ref;
};

struct RtspServer::CameraStream {
  RtspServer* owner = nullptr;
  RtspRoute* route = nullptr;
  std::string mount;
  std::string camera_id;
  std::string appsrc_name = "src";
  int route_index = -1;
  int width = 0;
  int height = 0;
  int fps = 30;
  V4l2Pipeline::BuildOptions capture_options;
  std::unique_ptr<V4l2Pipeline> capture;

  std::mutex mu;
  GstElement* appsrc = nullptr;
  GstRTSPMedia* media = nullptr;
  gulong unprepared_handler = 0;
  GstClockTime first_pts = GST_CLOCK_TIME_NONE;
  GstClockTime next_pts = 0;
};

struct RtspServer::RtspRoute {
  RtspServer* owner = nullptr;
  std::string mount;
  bool mosaic = false;
  std::string appsrc_name = "src";
  int cols = 0;
  int rows = 0;
  int tile_width = 0;
  int tile_height = 0;
  int output_width = 0;
  int output_height = 0;
  int fps = 30;
  std::vector<CameraStream*> inputs;

  std::mutex mu;
  GstElement* appsrc = nullptr;
  GstRTSPMedia* media = nullptr;
  gulong unprepared_handler = 0;
  GstClockTime first_pts = GST_CLOCK_TIME_NONE;
  GstClockTime next_pts = 0;
  std::vector<MosaicFrame> latest_frames;
};

RtspServer::~RtspServer() { Stop(); }

void RtspServer::UpdateMediapipeResult(const vision::MediapipeResult& result) {
  if (result.camera_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(overlay_mu_);
  mediapipe_results_[result.camera_id] = result;
}

void RtspServer::UpdateYoloResult(const vision::YoloResult& result) {
  if (result.camera_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(overlay_mu_);
  yolo_results_[result.camera_id] = result;
}

void RtspServer::ClearMediapipeResult(const std::string& camera_id) {
  std::lock_guard<std::mutex> lock(overlay_mu_);
  mediapipe_results_.erase(camera_id);
}

void RtspServer::ClearYoloResult(const std::string& camera_id) {
  std::lock_guard<std::mutex> lock(overlay_mu_);
  yolo_results_.erase(camera_id);
}

std::optional<vision::MediapipeResult> RtspServer::LatestMediapipeResult(
    const std::string& camera_id) const {
  std::lock_guard<std::mutex> lock(overlay_mu_);
  auto it = mediapipe_results_.find(camera_id);
  if (it == mediapipe_results_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<vision::YoloResult> RtspServer::LatestYoloResult(
    const std::string& camera_id) const {
  std::lock_guard<std::mutex> lock(overlay_mu_);
  auto it = yolo_results_.find(camera_id);
  if (it == yolo_results_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<cv::Mat> RtspServer::BuildOverlayNv12(CameraStream* stream, GstSample* sample) const {
  if (stream == nullptr || sample == nullptr) {
    return std::nullopt;
  }

  GstBuffer* input = gst_sample_get_buffer(sample);
  if (input == nullptr) {
    return std::nullopt;
  }

  const auto mediapipe_result = LatestMediapipeResult(stream->camera_id);
  const auto yolo_result = LatestYoloResult(stream->camera_id);
  const bool has_mediapipe_overlay =
      mediapipe_result.has_value() && mediapipe_result->ok &&
      !mediapipe_result->hands.empty();
  const bool has_yolo_overlay =
      yolo_result.has_value() && yolo_result->ok &&
      !yolo_result->detections.empty();
  if (!has_mediapipe_overlay && !has_yolo_overlay) {
    return std::nullopt;
  }

  auto rgb_frame = frame_converter_.ConvertToRgbFrame(sample, stream->camera_id);
  if (!rgb_frame.has_value() || !rgb_frame->owned_data) {
    return std::nullopt;
  }

  auto rgb_holder = std::static_pointer_cast<cv::Mat>(rgb_frame->owned_data);
  if (!rgb_holder || rgb_holder->empty()) {
    return std::nullopt;
  }

  if (has_mediapipe_overlay) {
    DrawMediapipeOverlay(*rgb_holder, *mediapipe_result);
  }
  if (has_yolo_overlay) {
    DrawYoloOverlay(*rgb_holder, *yolo_result);
  }

  cv::Mat nv12;
  if (!mediapipe_demo::ConvertRgbToNv12(*rgb_holder, &nv12) || nv12.empty()) {
    return std::nullopt;
  }
  return nv12;
}

GstBuffer* RtspServer::BuildOverlayBuffer(CameraStream* stream, GstSample* sample) const {
  if (sample == nullptr) {
    return nullptr;
  }

  GstBuffer* input = gst_sample_get_buffer(sample);
  if (input == nullptr) {
    return nullptr;
  }

  auto overlay_nv12 = BuildOverlayNv12(stream, sample);
  if (!overlay_nv12.has_value() || overlay_nv12->empty()) {
    return gst_buffer_copy(input);
  }

  const size_t bytes = overlay_nv12->total() * overlay_nv12->elemSize();
  GstBuffer* output = gst_buffer_new_allocate(nullptr, bytes, nullptr);
  if (output == nullptr) {
    return nullptr;
  }
  gst_buffer_fill(output, 0, overlay_nv12->data, bytes);
  return output;
}

void RtspServer::StopRouteMedia() {
  for (auto& [_, route] : routes_) {
    if (!route) {
      continue;
    }
    GstElement* appsrc = nullptr;
    GstRTSPMedia* media = nullptr;
    gulong unprepared_handler = 0;
    {
      std::lock_guard<std::mutex> lock(route->mu);
      appsrc = route->appsrc;
      route->appsrc = nullptr;
      media = route->media;
      route->media = nullptr;
      unprepared_handler = route->unprepared_handler;
      route->unprepared_handler = 0;
      route->first_pts = GST_CLOCK_TIME_NONE;
      route->next_pts = 0;
      route->latest_frames.clear();
    }
    if (media != nullptr && unprepared_handler != 0) {
      g_signal_handler_disconnect(media, unprepared_handler);
    }
    if (appsrc != nullptr) {
      gst_object_unref(appsrc);
    }
    if (media != nullptr) {
      gst_object_unref(media);
    }
  }
}

void RtspServer::StopCameraStreams() {
  for (auto& [_, stream] : camera_streams_) {
    if (!stream) {
      continue;
    }

    GstElement* appsrc = nullptr;
    GstRTSPMedia* media = nullptr;
    gulong unprepared_handler = 0;
    std::unique_ptr<V4l2Pipeline> capture;
    {
      std::lock_guard<std::mutex> lock(stream->mu);
      appsrc = stream->appsrc;
      stream->appsrc = nullptr;
      media = stream->media;
      stream->media = nullptr;
      unprepared_handler = stream->unprepared_handler;
      stream->unprepared_handler = 0;
      capture = std::move(stream->capture);
      stream->first_pts = GST_CLOCK_TIME_NONE;
      stream->next_pts = 0;
    }
    if (media != nullptr && unprepared_handler != 0) {
      g_signal_handler_disconnect(media, unprepared_handler);
    }
    if (appsrc != nullptr) {
      gst_object_unref(appsrc);
    }
    if (media != nullptr) {
      gst_object_unref(media);
    }

    if (capture) {
      capture->Stop();
    }
  }
  camera_streams_.clear();
}

void RtspServer::OnRouteMediaConfigure(GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer user_data) {
  auto* route = static_cast<RtspRoute*>(user_data);
  if (route == nullptr || media == nullptr) {
    return;
  }

  GstElement* element = gst_rtsp_media_get_element(media);
  if (element == nullptr) {
    return;
  }

  if (route->mosaic) {
    GstElement* appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), route->appsrc_name.c_str());
    if (appsrc == nullptr) {
      std::cerr << "[rtsp] appsrc " << route->appsrc_name
                << " not found for " << route->mount << "\n";
    } else {
      AttachRouteAppSrc(route, media, appsrc);
    }
    gst_object_unref(element);
    return;
  }

  for (CameraStream* stream : route->inputs) {
    if (stream == nullptr) {
      continue;
    }
    GstElement* appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), stream->appsrc_name.c_str());
    if (appsrc == nullptr) {
      std::cerr << "[rtsp] appsrc " << stream->appsrc_name
                << " not found for " << route->mount << "\n";
      continue;
    }
    AttachAppSrc(stream, media, appsrc);
  }

  gst_object_unref(element);
}

void RtspServer::AttachAppSrc(CameraStream* stream, GstRTSPMedia* media, GstElement* appsrc) {
  if (stream == nullptr || media == nullptr) {
    return;
  }
  if (appsrc == nullptr) {
    std::cerr << "[rtsp] appsrc not found for " << stream->mount << "\n";
    return;
  }

  g_object_set(G_OBJECT(appsrc),
               "is-live", TRUE,
               "format", GST_FORMAT_TIME,
               "do-timestamp", FALSE,
               "block", FALSE,
               nullptr);

  GstRTSPMedia* media_ref = GST_RTSP_MEDIA(gst_object_ref(media));
  const gulong unprepared_handler =
      g_signal_connect(media, "unprepared", G_CALLBACK(&RtspServer::OnMediaUnprepared), stream);

  bool start_capture = false;
  {
    std::lock_guard<std::mutex> lock(stream->mu);
    if (stream->appsrc != nullptr) {
      gst_object_unref(stream->appsrc);
    }
    if (stream->media != nullptr && stream->unprepared_handler != 0) {
      g_signal_handler_disconnect(stream->media, stream->unprepared_handler);
    }
    if (stream->media != nullptr) {
      gst_object_unref(stream->media);
    }
    stream->appsrc = appsrc;
    stream->media = media_ref;
    stream->unprepared_handler = unprepared_handler;
    stream->first_pts = GST_CLOCK_TIME_NONE;
    stream->next_pts = 0;
    start_capture = stream->capture == nullptr;
  }
  if (!start_capture) {
    return;
  }

  auto capture = std::make_unique<V4l2Pipeline>();
  std::string capture_err;
  if (!capture->Build(stream->capture_options,
                      [](const TelemetryEvent&) {},
                      [stream](const std::string& reason, bool fatal) {
                        std::cerr << "[rtsp] capture error on " << stream->mount
                                  << " fatal=" << fatal << ": " << reason << "\n";
                      },
                      &capture_err) ||
      !capture->Start(&capture_err)) {
    std::cerr << "[rtsp] failed to start capture for " << stream->mount
              << ": " << capture_err << "\n";
    return;
  }

  {
    std::lock_guard<std::mutex> lock(stream->mu);
    if (stream->appsrc != nullptr && stream->capture == nullptr) {
      stream->capture = std::move(capture);
    }
  }
  if (capture) {
    capture->Stop();
  }
}

void RtspServer::AttachRouteAppSrc(RtspRoute* route, GstRTSPMedia* media, GstElement* appsrc) {
  if (route == nullptr || media == nullptr) {
    return;
  }
  if (appsrc == nullptr) {
    std::cerr << "[rtsp] appsrc not found for " << route->mount << "\n";
    return;
  }

  g_object_set(G_OBJECT(appsrc),
               "is-live", TRUE,
               "format", GST_FORMAT_TIME,
               "do-timestamp", FALSE,
               "block", FALSE,
               nullptr);

  GstRTSPMedia* media_ref = GST_RTSP_MEDIA(gst_object_ref(media));
  const gulong unprepared_handler =
      g_signal_connect(media, "unprepared", G_CALLBACK(&RtspServer::OnRouteMediaUnprepared), route);

  std::vector<CameraStream*> streams_to_start;
  {
    std::lock_guard<std::mutex> lock(route->mu);
    if (route->appsrc != nullptr) {
      gst_object_unref(route->appsrc);
    }
    if (route->media != nullptr && route->unprepared_handler != 0) {
      g_signal_handler_disconnect(route->media, route->unprepared_handler);
    }
    if (route->media != nullptr) {
      gst_object_unref(route->media);
    }
    route->appsrc = appsrc;
    route->media = media_ref;
    route->unprepared_handler = unprepared_handler;
    route->first_pts = GST_CLOCK_TIME_NONE;
    route->next_pts = 0;
    route->latest_frames.assign(route->inputs.size(), MosaicFrame{});

    for (CameraStream* stream : route->inputs) {
      if (stream != nullptr && stream->capture == nullptr) {
        streams_to_start.push_back(stream);
      }
    }
  }

  for (CameraStream* stream : streams_to_start) {
    auto capture = std::make_unique<V4l2Pipeline>();
    std::string capture_err;
    if (!capture->Build(stream->capture_options,
                        [](const TelemetryEvent&) {},
                        [stream](const std::string& reason, bool fatal) {
                          std::cerr << "[rtsp] capture error on " << stream->mount
                                    << " fatal=" << fatal << ": " << reason << "\n";
                        },
                        &capture_err) ||
        !capture->Start(&capture_err)) {
      std::cerr << "[rtsp] failed to start capture for " << stream->mount
                << ": " << capture_err << "\n";
      continue;
    }

    std::lock_guard<std::mutex> lock(stream->mu);
    if (stream->capture == nullptr) {
      stream->capture = std::move(capture);
    }
    if (capture) {
      capture->Stop();
    }
  }
}

void RtspServer::OnMediaUnprepared(GstRTSPMedia* media, gpointer user_data) {
  auto* stream = static_cast<CameraStream*>(user_data);
  if (stream == nullptr) {
    return;
  }
  GstElement* appsrc = nullptr;
  GstRTSPMedia* media_ref = nullptr;
  {
    std::lock_guard<std::mutex> lock(stream->mu);
    appsrc = stream->appsrc;
    stream->appsrc = nullptr;
    if (stream->media == media) {
      if (stream->unprepared_handler != 0) {
        g_signal_handler_disconnect(media, stream->unprepared_handler);
      }
      media_ref = stream->media;
      stream->media = nullptr;
      stream->unprepared_handler = 0;
    }
    stream->first_pts = GST_CLOCK_TIME_NONE;
    stream->next_pts = 0;
  }
  if (appsrc != nullptr) {
    gst_object_unref(appsrc);
  }
  if (media_ref != nullptr) {
    gst_object_unref(media_ref);
  }
}

void RtspServer::OnRouteMediaUnprepared(GstRTSPMedia* media, gpointer user_data) {
  auto* route = static_cast<RtspRoute*>(user_data);
  if (route == nullptr) {
    return;
  }
  GstElement* appsrc = nullptr;
  GstRTSPMedia* media_ref = nullptr;
  {
    std::lock_guard<std::mutex> lock(route->mu);
    appsrc = route->appsrc;
    route->appsrc = nullptr;
    if (route->media == media) {
      if (route->unprepared_handler != 0) {
        g_signal_handler_disconnect(media, route->unprepared_handler);
      }
      media_ref = route->media;
      route->media = nullptr;
      route->unprepared_handler = 0;
    }
    route->first_pts = GST_CLOCK_TIME_NONE;
    route->next_pts = 0;
    route->latest_frames.clear();
  }
  if (appsrc != nullptr) {
    gst_object_unref(appsrc);
  }
  if (media_ref != nullptr) {
    gst_object_unref(media_ref);
  }
}

void RtspServer::PushMosaicSample(RtspRoute* route, CameraStream* stream, GstSample* sample) {
  if (route == nullptr || stream == nullptr || sample == nullptr) {
    return;
  }

  GstBuffer* input = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);
  if (input == nullptr || caps == nullptr) {
    return;
  }

  GstVideoInfo info;
  if (!gst_video_info_from_caps(&info, caps) ||
      GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_FORMAT_NV12) {
    return;
  }

  MosaicFrame frame;
  frame.valid = true;
  frame.width = GST_VIDEO_INFO_WIDTH(&info);
  frame.height = GST_VIDEO_INFO_HEIGHT(&info);
  frame.stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);

  auto overlay_nv12 = BuildOverlayNv12(stream, sample);
  if (overlay_nv12.has_value() && !overlay_nv12->empty()) {
    frame.nv12 = std::move(*overlay_nv12);
    frame.dmabuf_fd = -1;
    frame.stride = static_cast<int>(frame.nv12.step[0]);
  } else {
    frame.dmabuf_fd = ExtractDmabufFd(input);
    if (frame.dmabuf_fd >= 0) {
      frame.sample_ref = HoldSampleRef(sample);
    } else {
      GstMapInfo map;
      if (!gst_buffer_map(input, &map, GST_MAP_READ)) {
        return;
      }
      frame.nv12 = cv::Mat(frame.height * 3 / 2, frame.width, CV_8UC1);
      const size_t bytes = std::min(map.size, frame.nv12.total() * frame.nv12.elemSize());
      std::memcpy(frame.nv12.data, map.data, bytes);
      gst_buffer_unmap(input, &map);
      frame.dmabuf_fd = -1;
      frame.stride = static_cast<int>(frame.nv12.step[0]);
    }
  }

  GstElement* appsrc = nullptr;
  std::vector<MosaicFrame> snapshot;
  GstClockTime pts = GST_BUFFER_PTS(input);
  const GstClockTime duration = route->fps > 0
      ? static_cast<GstClockTime>(GST_SECOND / route->fps)
      : GST_CLOCK_TIME_NONE;
  {
    std::lock_guard<std::mutex> lock(route->mu);
    if (route->appsrc == nullptr ||
        stream->route_index < 0 ||
        stream->route_index >= static_cast<int>(route->latest_frames.size())) {
      return;
    }

    route->latest_frames[stream->route_index] = std::move(frame);
    if (stream->route_index != 0) {
      return;
    }

    for (const auto& latest : route->latest_frames) {
      if (!latest.valid) {
        return;
      }
    }

    appsrc = route->appsrc;
    gst_object_ref(appsrc);
    snapshot = route->latest_frames;

    if (!GST_CLOCK_TIME_IS_VALID(pts)) {
      pts = route->next_pts;
    } else {
      if (!GST_CLOCK_TIME_IS_VALID(route->first_pts)) {
        route->first_pts = pts;
      }
      pts -= route->first_pts;
    }
    if (GST_CLOCK_TIME_IS_VALID(duration)) {
      route->next_pts = pts + duration;
    } else {
      route->next_pts = pts;
    }
  }

  std::vector<mediapipe_demo::Nv12RgaInput> rga_inputs;
  rga_inputs.reserve(snapshot.size());
  for (const auto& latest : snapshot) {
    mediapipe_demo::Nv12RgaInput rga_input;
    rga_input.dmabuf_fd = latest.dmabuf_fd;
    rga_input.data = latest.nv12.empty() ? nullptr : latest.nv12.data;
    rga_input.width = latest.width;
    rga_input.height = latest.height;
    rga_input.stride = latest.stride;
    rga_inputs.push_back(rga_input);
  }

  cv::Mat mosaic;
  if (!mediapipe_demo::MosaicNv12ToNv12(rga_inputs,
                                        route->cols,
                                        route->rows,
                                        route->tile_width,
                                        route->tile_height,
                                        route->output_width,
                                        route->output_height,
                                        &mosaic) ||
      mosaic.empty()) {
    gst_object_unref(appsrc);
    return;
  }

  const size_t bytes = mosaic.total() * mosaic.elemSize();
  GstBuffer* output = gst_buffer_new_allocate(nullptr, bytes, nullptr);
  if (output == nullptr) {
    gst_object_unref(appsrc);
    return;
  }
  gst_buffer_fill(output, 0, mosaic.data, bytes);
  GST_BUFFER_PTS(output) = pts;
  GST_BUFFER_DTS(output) = pts;
  GST_BUFFER_DURATION(output) = duration;

  const GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc), output);
  if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING && flow != GST_FLOW_EOS) {
    std::cerr << "[rtsp] mosaic appsrc push failed for " << route->mount
              << ": " << flow << "\n";
  }
  gst_object_unref(appsrc);
}

void RtspServer::PushRtspSample(CameraStream* stream, GstSample* sample) {
  if (stream == nullptr || sample == nullptr) {
    return;
  }

  if (stream->route != nullptr && stream->route->mosaic) {
    if (stream->owner != nullptr) {
      stream->owner->PushMosaicSample(stream->route, stream, sample);
    }
    return;
  }

  GstBuffer* input = gst_sample_get_buffer(sample);
  if (input == nullptr) {
    return;
  }

  GstElement* appsrc = nullptr;
  GstClockTime pts = GST_BUFFER_PTS(input);
  GstClockTime duration = stream->fps > 0
      ? static_cast<GstClockTime>(GST_SECOND / stream->fps)
      : GST_CLOCK_TIME_NONE;
  {
    std::lock_guard<std::mutex> lock(stream->mu);
    if (stream->appsrc == nullptr) {
      return;
    }
    appsrc = stream->appsrc;
    gst_object_ref(appsrc);

    if (!GST_CLOCK_TIME_IS_VALID(pts)) {
      pts = stream->next_pts;
    } else {
      if (!GST_CLOCK_TIME_IS_VALID(stream->first_pts)) {
        stream->first_pts = pts;
      }
      pts -= stream->first_pts;
    }
    if (GST_CLOCK_TIME_IS_VALID(duration)) {
      stream->next_pts = pts + duration;
    } else {
      stream->next_pts = pts;
    }
  }

  GstBuffer* output = stream->owner != nullptr
      ? stream->owner->BuildOverlayBuffer(stream, sample)
      : gst_buffer_copy(input);
  if (output == nullptr) {
    gst_object_unref(appsrc);
    return;
  }
  GST_BUFFER_PTS(output) = pts;
  GST_BUFFER_DTS(output) = pts;
  GST_BUFFER_DURATION(output) = duration;

  const GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(appsrc), output);
  if (flow != GST_FLOW_OK && flow != GST_FLOW_FLUSHING && flow != GST_FLOW_EOS) {
    std::cerr << "[rtsp] appsrc push failed for " << stream->mount
              << ": " << flow << "\n";
  }
  gst_object_unref(appsrc);
}

bool RtspServer::Start(const BoardConfig& board, const SessionProfile& profile, std::string* err) {
  Stop();

  if (!board.rtsp.has_value()) {
    if (err) *err = "no [rtsp] section in board config";
    return false;
  }

  const auto& rtsp = *board.rtsp;
  const std::string codec = rtsp.codec;
  const int gop = profile.gop;

  server_ = gst_rtsp_server_new();
  gst_rtsp_server_set_service(server_, std::to_string(rtsp.port).c_str());

  GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server_);
  auto fail_start = [&]() {
    g_object_unref(mounts);
    StopCameraStreams();
    StopRouteMedia();
    if (server_ != nullptr) {
      g_object_unref(server_);
      server_ = nullptr;
    }
    routes_.clear();
    port_ = 0;
    return false;
  };

  const std::vector<std::string>& rtsp_camera_ids =
      profile.record_cameras.empty() ? profile.preview_cameras : profile.record_cameras;

  // Collect cameras for mosaic and bitrate fallback.
  std::vector<const CameraNodeSet*> cams;
  for (const auto& cam_id : rtsp_camera_ids) {
    const CameraNodeSet* cam = FindCamera(board, cam_id);
    if (cam) cams.push_back(cam);
  }

  if (cams.empty()) {
    if (err) *err = "no cameras configured for RTSP";
    return fail_start();
  }

  // Total bitrate: sum of all cameras
  int total_bitrate = 0;
  for (const auto* cam : cams) total_bitrate += cam->bitrate;

  // Use RTSP-specific bitrate if configured, otherwise fall back to sum.
  const int stream_bitrate = rtsp.bitrate > 0 ? rtsp.bitrate : total_bitrate;

  auto add_capture_stream = [&](const CameraNodeSet& cam,
                                const std::string& route_mount,
                                const std::string& stream_key,
                                const std::string& appsrc_name,
                                int width,
                                int height,
                                int bitrate) -> CameraStream* {
    auto stream = std::make_shared<CameraStream>();
    stream->owner = this;
    stream->mount = route_mount;
    stream->camera_id = cam.id;
    stream->appsrc_name = appsrc_name;
    stream->width = width;
    stream->height = height;
    stream->fps = cam.fps;

    V4l2Pipeline::BuildOptions options;
    options.source.id = cam.id + "_rtsp_" + stream_key;
    options.source.device = cam.record_device;
    options.source.input_format = cam.input_format;
    options.source.io_mode = cam.io_mode;
    options.source.width = width;
    options.source.height = height;
    options.source.fps = cam.fps;
    options.source.bitrate = bitrate;
    options.app_sink.enabled = true;
    CameraStream* stream_ptr = stream.get();
    options.app_sink.sample_callback = [stream_ptr](GstSample* sample) {
      RtspServer::PushRtspSample(stream_ptr, sample);
    };
    stream->capture_options = std::move(options);

    CameraStream* raw = stream.get();
    camera_streams_.emplace(stream_key, std::move(stream));
    return raw;
  };

  std::unordered_set<std::string> added_mounts;
  for (const auto& configured_mount : rtsp.mounts) {
    const std::string mount = NormalizeMount(configured_mount);
    if (mount.empty()) {
      if (err) *err = "empty RTSP mount in rtsp.mounts";
      return fail_start();
    }
    if (!added_mounts.insert(mount).second) {
      continue;
    }

    if (mount == "cam") {
      auto route = std::make_shared<RtspRoute>();
      route->owner = this;
      route->mount = "/cam";
      route->mosaic = true;
      const int tile_w = cams[0]->preview_width;
      const int tile_h = cams[0]->preview_height;
      const int n = static_cast<int>(cams.size());
      route->cols = static_cast<int>(std::ceil(std::sqrt(n)));
      route->rows = (n + route->cols - 1) / route->cols;
      route->tile_width = tile_w;
      route->tile_height = tile_h;
      const bool h265 = Uppercase(codec) == "H265";
      route->output_width = h265 ? AlignUp(tile_w * route->cols, 16) : tile_w * route->cols;
      route->output_height = h265 ? AlignUp(tile_h * route->rows, 16) : tile_h * route->rows;
      route->fps = cams[0]->fps;
      route->latest_frames.assign(cams.size(), MosaicFrame{});
      for (int i = 0; i < static_cast<int>(cams.size()); ++i) {
        const CameraNodeSet& cam = *cams[i];
        CameraStream* stream = add_capture_stream(
            cam, route->mount, "cam_" + cam.id, "src",
            tile_w, tile_h, cam.bitrate);
        stream->route = route.get();
        stream->route_index = i;
        route->inputs.push_back(stream);
      }

      const std::string launch = BuildCameraAppSrcLaunchString(route->output_width,
                                                               route->output_height,
                                                               route->fps,
                                                               codec,
                                                               gop,
                                                               stream_bitrate);
      GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
      gst_rtsp_media_factory_set_launch(factory, launch.c_str());
      gst_rtsp_media_factory_set_shared(factory, TRUE);
      gst_rtsp_media_factory_set_latency(factory, 0);
      g_signal_connect(factory, "media-configure", G_CALLBACK(&RtspServer::OnRouteMediaConfigure), route.get());
      gst_rtsp_mount_points_add_factory(mounts, route->mount.c_str(), factory);

      std::cerr << "rtsp mount: rtsp://0.0.0.0:" << rtsp.port << route->mount << "\n";
      std::cerr << "  capture: mosaic " << tile_w << "x" << tile_h
                << "@" << route->fps << "fps x" << cams.size()
                << " via RGA -> " << route->output_width << "x"
                << route->output_height << "\n";
      std::cerr << "  pipeline: " << launch << "\n";
      routes_.emplace(mount, std::move(route));
      continue;
    }

    const CameraNodeSet* cam = FindCamera(board, mount);
    if (cam == nullptr) {
      if (err) *err = "unknown RTSP mount '" + configured_mount + "': expected 'cam' or a camera id";
      return fail_start();
    }

    const int camera_bitrate = rtsp.bitrate > 0 ? rtsp.bitrate : cam->bitrate;
    const bool h265 = Uppercase(codec) == "H265";
    const int out_w = h265 ? AlignUp(cam->preview_width, 16) : cam->preview_width;
    const int out_h = h265 ? AlignUp(cam->preview_height, 16) : cam->preview_height;

    auto route = std::make_shared<RtspRoute>();
    route->mount = "/" + mount;
    route->inputs.push_back(add_capture_stream(
        *cam, route->mount, mount, "src", out_w, out_h, camera_bitrate));

    const std::string camera_launch =
        BuildCameraAppSrcLaunchString(out_w, out_h, cam->fps, codec, gop, camera_bitrate);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, camera_launch.c_str());
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_latency(factory, 0);
    g_signal_connect(factory, "media-configure", G_CALLBACK(&RtspServer::OnRouteMediaConfigure), route.get());
    gst_rtsp_mount_points_add_factory(mounts, route->mount.c_str(), factory);

    std::cerr << "rtsp mount: rtsp://0.0.0.0:" << rtsp.port << route->mount << "\n";
    std::cerr << "  capture: " << cam->record_device << " "
              << out_w << "x" << out_h << "@" << cam->fps << "fps\n";
    std::cerr << "  pipeline: " << camera_launch << "\n";
    routes_.emplace(mount, std::move(route));
  }

  g_object_unref(mounts);

  attach_id_ = gst_rtsp_server_attach(server_, nullptr);
  if (attach_id_ == 0) {
    if (err) *err = "failed to attach RTSP server on port " + std::to_string(rtsp.port);
    StopCameraStreams();
    StopRouteMedia();
    g_object_unref(server_);
    server_ = nullptr;
    routes_.clear();
    port_ = 0;
    return false;
  }

  port_ = rtsp.port;
  std::cerr << "RTSP server listening on port " << port_ << "\n";
  return true;
}

void RtspServer::Stop() {
  const bool was_running = server_ != nullptr || !routes_.empty() || !camera_streams_.empty();

  if (attach_id_ != 0) {
    g_source_remove(attach_id_);
    attach_id_ = 0;
  }
  StopCameraStreams();
  StopRouteMedia();
  if (server_ != nullptr) {
    g_object_unref(server_);
    server_ = nullptr;
  }
  routes_.clear();
  port_ = 0;
  if (was_running) {
    std::cerr << "RTSP server stopped\n";
  }
}

}  // namespace rkstudio::media
