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

class CameraPipeline {
 public:
  struct BuildOptions {
    CameraNodeSet camera;
    std::vector<std::string> sink_priority;
    fs::path session_dir;
    WId preview_window_id = 0;
    bool enable_preview = true;
    bool enable_record = false;
    bool enable_ai = false;
    int gop = 30;
    std::function<void(GstSample*)> ai_sample_callback;
  };

  using TelemetryCallback = std::function<void(const TelemetryEvent&)>;
  using ErrorCallback = std::function<void(const std::string&, bool)>;

  CameraPipeline();
  ~CameraPipeline();

  CameraPipeline(const CameraPipeline&) = delete;
  CameraPipeline& operator=(const CameraPipeline&) = delete;

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
    CameraPipeline* self = nullptr;
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
  static GstFlowReturn OnAiSample(GstAppSink* sink, gpointer user_data);

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

  GstElement* preview_convert_ = nullptr;
  GstElement* preview_sink_ = nullptr;

  GstElement* encoder_ = nullptr;
  GstElement* parser_ = nullptr;
  GstElement* mux_ = nullptr;
  GstElement* record_sink_ = nullptr;

  GstElement* tee_ = nullptr;
  GstElement* ai_queue_ = nullptr;
  GstElement* ai_sink_ = nullptr;
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
