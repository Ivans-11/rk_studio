#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <glib.h>

#include "rk_studio/domain/types.h"

struct _GstRTSPServer;
typedef struct _GstElement GstElement;
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

 private:
  struct CameraStream;
  struct RtspRoute;
  static void OnRouteMediaConfigure(GstRTSPMediaFactory* factory, GstRTSPMedia* media, gpointer user_data);
  static void OnMediaUnprepared(GstRTSPMedia* media, gpointer user_data);
  static void AttachAppSrc(CameraStream* stream, GstRTSPMedia* media, GstElement* appsrc);
  static void PushRtspSample(CameraStream* stream, GstSample* sample);
  void StopCameraStreams();

  _GstRTSPServer* server_ = nullptr;
  unsigned int attach_id_ = 0;
  int port_ = 0;
  std::map<std::string, std::shared_ptr<RtspRoute>> routes_;
  std::map<std::string, std::shared_ptr<CameraStream>> camera_streams_;
};

}  // namespace rkstudio::media
