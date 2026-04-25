#include "rk_studio/media_core/rtsp_server.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

namespace rkstudio::media {
namespace {

std::string Uppercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
  return value;
}

int IoModeInt(const std::string& io_mode) {
  static const std::unordered_map<std::string, int> kModes{
      {"AUTO", 0}, {"RW", 1}, {"MMAP", 2}, {"USERPTR", 3}, {"DMABUF", 4}, {"DMABUF-IMPORT", 5}};
  const auto it = kModes.find(Uppercase(io_mode));
  return it != kModes.end() ? it->second : 0;
}

bool IsJpegFormat(const std::string& fmt) {
  const std::string upper = Uppercase(fmt);
  return upper == "MJPG" || upper == "JPEG";
}

int AlignUp(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
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

std::string BuildMosaicLaunchString(const std::vector<const CameraNodeSet*>& cams,
                                    const std::string& codec, int gop, int bitrate) {
  const int n = static_cast<int>(cams.size());
  const int cols = static_cast<int>(std::ceil(std::sqrt(n)));
  const int rows = (n + cols - 1) / cols;

  // Use first camera's preview dimensions as tile size
  const int tile_w = cams[0]->preview_width;
  const int tile_h = cams[0]->preview_height;
  const int out_w = tile_w * cols;
  const int out_h = tile_h * rows;
  const int fps = cams[0]->fps;

  std::ostringstream ss;
  ss << "( compositor name=mix background=black ";

  for (int i = 0; i < n; ++i) {
    const int col = i % cols;
    const int row = i / cols;
    ss << "sink_" << i << "::xpos=" << col * tile_w << " "
       << "sink_" << i << "::ypos=" << row * tile_h << " ";
  }

  ss << "! video/x-raw,width=" << out_w << ",height=" << out_h << " "
     << "! videoconvert ! video/x-raw,format=NV12 ";
  AppendEncoderTail(ss, codec, gop, bitrate);

  for (int i = 0; i < n; ++i) {
    const auto& cam = *cams[i];
    const int io = IoModeInt(cam.io_mode);

    ss << "v4l2src device=" << cam.record_device << " io-mode=" << io << " do-timestamp=true ";

    if (IsJpegFormat(cam.input_format)) {
      ss << "! image/jpeg,width=" << tile_w << ",height=" << tile_h
         << ",framerate=" << fps << "/1 "
         << "! jpegdec ! videoconvert ! video/x-raw,format=NV12,width="
         << tile_w << ",height=" << tile_h << " ";
    } else {
      ss << "! video/x-raw,format=NV12,width=" << tile_w << ",height=" << tile_h
         << ",framerate=" << fps << "/1 ";
    }

    ss << "! queue leaky=downstream max-size-buffers=1 "
       << "! mix.sink_" << i << " ";
  }

  ss << ")";
  return ss.str();
}

std::string BuildCameraLaunchString(const CameraNodeSet& cam,
                                    const std::string& codec, int gop, int bitrate) {
  const int io = IoModeInt(cam.io_mode);
  const bool h265 = Uppercase(codec) == "H265";
  const int out_w = h265 ? AlignUp(cam.preview_width, 16) : cam.preview_width;
  const int out_h = h265 ? AlignUp(cam.preview_height, 16) : cam.preview_height;

  std::ostringstream ss;
  ss << "( v4l2src device=" << cam.record_device << " io-mode=" << io << " do-timestamp=true ";

  if (IsJpegFormat(cam.input_format)) {
    ss << "! image/jpeg,width=" << out_w << ",height=" << out_h
       << ",framerate=" << cam.fps << "/1 "
       << "! jpegdec ! videoconvert ! video/x-raw,format=NV12,width="
       << out_w << ",height=" << out_h << " ";
  } else {
    ss << "! video/x-raw,format=NV12,width=" << out_w << ",height=" << out_h
       << ",framerate=" << cam.fps << "/1 ";
  }

  ss << "! videoconvert ! video/x-raw,format=NV12 ";
  AppendEncoderTail(ss, codec, gop, bitrate);
  ss << ")";
  return ss.str();
}

void AddRtspFactory(GstRTSPMountPoints* mounts,
                    const std::string& path,
                    const std::string& launch,
                    int port) {
  GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_launch(factory, launch.c_str());
  gst_rtsp_media_factory_set_shared(factory, TRUE);
  gst_rtsp_media_factory_set_latency(factory, 0);
  gst_rtsp_mount_points_add_factory(mounts, path.c_str(), factory);

  std::cerr << "rtsp mount: rtsp://0.0.0.0:" << port << path << "\n";
  std::cerr << "  pipeline: " << launch << "\n";
}

std::string NormalizeMount(std::string mount) {
  while (!mount.empty() && mount.front() == '/') {
    mount.erase(mount.begin());
  }
  return mount;
}

}  // namespace

RtspServer::~RtspServer() { Stop(); }

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

  // Collect cameras for mosaic
  std::vector<const CameraNodeSet*> cams;
  for (const auto& cam_id : profile.record_cameras) {
    const CameraNodeSet* cam = FindCamera(board, cam_id);
    if (cam) cams.push_back(cam);
  }

  if (cams.empty()) {
    if (err) *err = "no cameras configured for RTSP";
    g_object_unref(mounts);
    g_object_unref(server_);
    server_ = nullptr;
    return false;
  }

  // Total bitrate: sum of all cameras
  int total_bitrate = 0;
  for (const auto* cam : cams) total_bitrate += cam->bitrate;

  // Use RTSP-specific bitrate if configured, otherwise fall back to sum.
  const int stream_bitrate = rtsp.bitrate > 0 ? rtsp.bitrate : total_bitrate;

  std::unordered_set<std::string> added_mounts;
  for (const auto& configured_mount : rtsp.mounts) {
    const std::string mount = NormalizeMount(configured_mount);
    if (mount.empty()) {
      if (err) *err = "empty RTSP mount in rtsp.mounts";
      g_object_unref(mounts);
      g_object_unref(server_);
      server_ = nullptr;
      return false;
    }
    if (!added_mounts.insert(mount).second) {
      continue;
    }

    if (mount == "cam") {
      const std::string launch = BuildMosaicLaunchString(cams, codec, gop, stream_bitrate);
      AddRtspFactory(mounts, "/cam", launch, rtsp.port);
      continue;
    }

    const CameraNodeSet* cam = FindCamera(board, mount);
    if (cam == nullptr) {
      if (err) *err = "unknown RTSP mount '" + configured_mount + "': expected 'cam' or a camera id";
      g_object_unref(mounts);
      g_object_unref(server_);
      server_ = nullptr;
      return false;
    }

    const int camera_bitrate = rtsp.bitrate > 0 ? rtsp.bitrate : cam->bitrate;
    const std::string camera_launch = BuildCameraLaunchString(*cam, codec, gop, camera_bitrate);
    AddRtspFactory(mounts, "/" + mount, camera_launch, rtsp.port);
  }

  g_object_unref(mounts);

  attach_id_ = gst_rtsp_server_attach(server_, nullptr);
  if (attach_id_ == 0) {
    if (err) *err = "failed to attach RTSP server on port " + std::to_string(rtsp.port);
    g_object_unref(server_);
    server_ = nullptr;
    return false;
  }

  port_ = rtsp.port;
  std::cerr << "RTSP server listening on port " << port_ << "\n";
  return true;
}

void RtspServer::Stop() {
  if (!server_) return;

  if (attach_id_ != 0) {
    g_source_remove(attach_id_);
    attach_id_ = 0;
  }
  g_object_unref(server_);
  server_ = nullptr;
  port_ = 0;
  std::cerr << "RTSP server stopped\n";
}

}  // namespace rkstudio::media
