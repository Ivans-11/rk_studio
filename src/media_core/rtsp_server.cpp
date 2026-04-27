#include "rk_studio/media_core/rtsp_server.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

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

std::string BuildMosaicAppSrcLaunchString(const std::vector<const CameraNodeSet*>& cams,
                                          const std::string& codec,
                                          int gop,
                                          int bitrate) {
  const int n = static_cast<int>(cams.size());
  const int cols = static_cast<int>(std::ceil(std::sqrt(n)));
  const int rows = (n + cols - 1) / cols;

  // Use first camera's preview dimensions as tile size
  const int tile_w = cams[0]->preview_width;
  const int tile_h = cams[0]->preview_height;
  const bool h265 = Uppercase(codec) == "H265";
  const int out_w = h265 ? AlignUp(tile_w * cols, 16) : tile_w * cols;
  const int out_h = h265 ? AlignUp(tile_h * rows, 16) : tile_h * rows;
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
    ss << "appsrc name=src_" << i << " is-live=true format=time do-timestamp=false block=false "
       << "! video/x-raw,format=NV12,width=" << tile_w << ",height=" << tile_h
       << ",framerate=" << fps << "/1 "
       << "! queue leaky=downstream max-size-buffers=2 "
       << "! mix.sink_" << i << " ";
  }

  ss << ")";
  return ss.str();
}

std::string NormalizeMount(std::string mount) {
  while (!mount.empty() && mount.front() == '/') {
    mount.erase(mount.begin());
  }
  return mount;
}

}  // namespace

struct RtspServer::CameraStream {
  std::string mount;
  std::string camera_id;
  std::string appsrc_name = "src";
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
  std::string mount;
  std::vector<CameraStream*> inputs;
};

RtspServer::~RtspServer() { Stop(); }

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

void RtspServer::PushRtspSample(CameraStream* stream, GstSample* sample) {
  if (stream == nullptr || sample == nullptr) {
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

  GstBuffer* output = gst_buffer_copy(input);
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
      route->mount = "/cam";
      const int tile_w = cams[0]->preview_width;
      const int tile_h = cams[0]->preview_height;
      for (int i = 0; i < static_cast<int>(cams.size()); ++i) {
        const CameraNodeSet& cam = *cams[i];
        route->inputs.push_back(add_capture_stream(
            cam, route->mount, "cam_" + cam.id, "src_" + std::to_string(i),
            tile_w, tile_h, cam.bitrate));
      }

      const std::string launch = BuildMosaicAppSrcLaunchString(cams, codec, gop, stream_bitrate);
      GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
      gst_rtsp_media_factory_set_launch(factory, launch.c_str());
      gst_rtsp_media_factory_set_shared(factory, TRUE);
      gst_rtsp_media_factory_set_latency(factory, 0);
      g_signal_connect(factory, "media-configure", G_CALLBACK(&RtspServer::OnRouteMediaConfigure), route.get());
      gst_rtsp_mount_points_add_factory(mounts, route->mount.c_str(), factory);

      std::cerr << "rtsp mount: rtsp://0.0.0.0:" << rtsp.port << route->mount << "\n";
      std::cerr << "  capture: mosaic " << tile_w << "x" << tile_h
                << "@" << cams[0]->fps << "fps x" << cams.size() << "\n";
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
