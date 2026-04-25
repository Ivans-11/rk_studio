#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <QtGui/qwindowdefs.h>

#include "rk_studio/domain/types.h"

namespace rkstudio::media {

namespace fs = std::filesystem;

class V4l2Pipeline {
 public:
  struct SourceOptions {
    std::string id;
    std::string device;
    std::string input_format = "NV12";
    std::string io_mode = "dmabuf";
    int width = 640;
    int height = 360;
    int fps = 30;
    int bitrate = 8'000'000;
  };

  struct PreviewBranch {
    bool enabled = false;
    std::vector<std::string> sink_priority;
    WId window_id = 0;
  };

  struct RecordBranch {
    bool enabled = false;
    fs::path session_dir;
    int gop = 30;
  };

  struct AppSinkBranch {
    bool enabled = false;
    std::function<void(GstSample*)> sample_callback;
  };

  struct BuildOptions {
    SourceOptions source;
    PreviewBranch preview;
    RecordBranch record;
    AppSinkBranch app_sink;
  };

  using TelemetryCallback = std::function<void(const TelemetryEvent&)>;
  using ErrorCallback = std::function<void(const std::string&, bool)>;

  V4l2Pipeline();
  ~V4l2Pipeline();

  V4l2Pipeline(const V4l2Pipeline&) = delete;
  V4l2Pipeline& operator=(const V4l2Pipeline&) = delete;

  bool Build(const BuildOptions& options,
             TelemetryCallback telemetry_callback,
             ErrorCallback error_callback,
             std::string* err);
  bool Start(std::string* err);
  void Stop();
  void SetPreviewWindow(WId window_id);

  const std::string& camera_id() const;
  const std::string& resolved_device() const;
  std::string record_output_path() const;

 private:
  struct ProbeContext {
    V4l2Pipeline* self = nullptr;
    std::string stream_id;
    std::atomic<uint64_t>* seq = nullptr;
  };

  bool BuildPipeline(std::string* err);
  bool CreatePreviewSink(std::string* err);
  void InstallProbes(GstElement* source_tail);
  void RunBusLoop();
  void EmitTelemetry(TelemetryEvent event);
  void Fail(const std::string& reason, bool fatal);
  void BindPreviewOverlay();

  static GstPadProbeReturn OnCaptureProbe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
  static GstFlowReturn OnAppSinkSample(GstAppSink* sink, gpointer user_data);

  BuildOptions options_;
  TelemetryCallback telemetry_callback_;
  ErrorCallback error_callback_;
  std::string resolved_device_;
  std::string record_output_path_;

  GstElement* pipeline_ = nullptr;
  GstElement* source_ = nullptr;
  GstElement* source_caps_ = nullptr;
  GstElement* jpeg_decoder_ = nullptr;
  GstElement* normalize_convert_ = nullptr;
  GstElement* normalize_caps_ = nullptr;
  GstElement* rate_filter_ = nullptr;
  GstElement* rate_caps_ = nullptr;

  GstElement* preview_convert_ = nullptr;
  GstElement* preview_sink_ = nullptr;

  GstElement* encoder_ = nullptr;
  GstElement* parser_ = nullptr;
  GstElement* mux_ = nullptr;
  GstElement* record_sink_ = nullptr;

  GstElement* tee_ = nullptr;
  GstElement* appsink_queue_ = nullptr;
  GstElement* app_sink_ = nullptr;
  GstElement* record_queue_ = nullptr;
  GstElement* preview_queue_ = nullptr;

  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> bus_done_{false};
  std::atomic<uint64_t> seq_{0};
  std::atomic<GstClockTime> base_time_ns_{GST_CLOCK_TIME_NONE};
  std::thread bus_thread_;
  std::vector<std::unique_ptr<ProbeContext>> probe_contexts_;
};

}  // namespace rkstudio::media
