#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <glib.h>
#include <opencv2/core.hpp>

#include "rk_studio/domain/types.h"
#include "rk_studio/vision_core/vision_types.h"

struct _GstRTSPServer;
typedef struct _GstElement GstElement;
typedef struct _GstBuffer GstBuffer;
typedef struct _GstRTSPMedia GstRTSPMedia;
typedef struct _GstRTSPMediaFactory GstRTSPMediaFactory;
typedef struct _GstSample GstSample;

namespace rkstudio::media {

class RtspServer {
 public:
  RtspServer() = default;
  ~RtspServer();

  RtspServer(const RtspServer&) = delete;
  RtspServer& operator=(const RtspServer&) = delete;

  bool Start(const BoardConfig& board, const SessionProfile& profile, std::string* err);
  void Stop();
  bool is_running() const { return server_ != nullptr; }
  int port() const { return port_; }
  void UpdateMediapipeResult(const vision::MediapipeResult& result);
  void UpdateYoloResult(const vision::YoloResult& result);
  void UpdateFaceExpressionResult(const vision::FaceExpressionResult& result);
  void ClearMediapipeResult(const std::string& camera_id);
  void ClearYoloResult(const std::string& camera_id);
  void ClearFaceExpressionResult(const std::string& camera_id);

 private:
  struct CameraStream;
  struct RtspRoute;
  static void OnRouteMediaConfigure(GstRTSPMediaFactory* factory, GstRTSPMedia* media, gpointer user_data);
  static void OnMediaUnprepared(GstRTSPMedia* media, gpointer user_data);
  static void OnRouteMediaUnprepared(GstRTSPMedia* media, gpointer user_data);
  static void AttachAppSrc(CameraStream* stream, GstRTSPMedia* media, GstElement* appsrc);
  static void AttachRouteAppSrc(RtspRoute* route, GstRTSPMedia* media, GstElement* appsrc);
  static void PushRtspSample(CameraStream* stream, GstSample* sample);
  GstBuffer* BuildOverlayBuffer(CameraStream* stream, GstSample* sample) const;
  std::optional<cv::Mat> BuildOverlayNv12(CameraStream* stream, GstSample* sample) const;
  void PushMosaicSample(RtspRoute* route, CameraStream* stream, GstSample* sample);
  std::optional<vision::MediapipeResult> LatestMediapipeResult(const std::string& camera_id) const;
  std::optional<vision::YoloResult> LatestYoloResult(const std::string& camera_id) const;
  std::optional<vision::FaceExpressionResult> LatestFaceExpressionResult(const std::string& camera_id) const;
  void StopRouteMedia();
  void StopCameraStreams();

  _GstRTSPServer* server_ = nullptr;
  unsigned int attach_id_ = 0;
  int port_ = 0;
  mutable std::mutex overlay_mu_;
  std::map<std::string, vision::MediapipeResult> mediapipe_results_;
  std::map<std::string, vision::YoloResult> yolo_results_;
  std::map<std::string, vision::FaceExpressionResult> face_expression_results_;
  std::map<std::string, std::shared_ptr<RtspRoute>> routes_;
  std::map<std::string, std::shared_ptr<CameraStream>> camera_streams_;
};

}  // namespace rkstudio::media
