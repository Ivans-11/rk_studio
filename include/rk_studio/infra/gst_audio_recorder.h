#pragma once

#include <gst/gst.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rk_studio/infra/config_types.h"
#include "rk_studio/infra/telemetry.h"

namespace rkinfra {

namespace fs = std::filesystem;

class GstAudioRecorder {
 public:
  using EventCallback = std::function<void(const StreamEvent&)>;
  using PcmCallback = std::function<void(const std::string& source_id,
                                         GstClockTime pts,
                                         int sample_rate,
                                         int channels,
                                         const int16_t* samples,
                                         size_t sample_count)>;

  GstAudioRecorder(AudioConfig config,
                   uint64_t queue_mux_max_time_ns,
                   EventCallback on_event,
                   const fs::path& session_dir,
                   bool record_to_file = true);
  ~GstAudioRecorder();

  bool Build(std::string* err);
  bool Start(std::string* err);
  void RequestStop();
  void Stop();

  bool failed() const;
  const std::atomic<bool>& failure_flag() const;
  std::string failure_reason() const;
  const OutputStreamInfo& stream_output() const;
  void SetPcmCallback(PcmCallback callback);

 private:
  struct ProbeContext {
    GstAudioRecorder* self = nullptr;
    std::string stream_id;
    std::atomic<uint64_t>* seq = nullptr;
  };

  struct QueueSignalContext {
    GstAudioRecorder* self = nullptr;
    std::string stream_id;
    std::string queue_name;
  };

  bool BuildPipeline(std::string* err);
  bool StartPipeline(std::string* err);
  void CleanupPipeline();
  void InstallProbes();
  void InstallQueueSignals();
  void BusLoop();
  uint64_t ComputeMonotonicNsFromPts(GstClockTime pts) const;
  void Fail(const std::string& reason);
  void PushEvent(StreamEvent event);

  static GstPadProbeReturn OnBufferProbe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data);
  static GstFlowReturn OnNewSample(GstElement* sink, gpointer user_data);
  static void OnQueueOverrun(GstElement* element, gpointer user_data);

  AudioConfig config_;
  uint64_t queue_mux_max_time_ns_ = 0;
  EventCallback on_event_;
  fs::path session_dir_;
  fs::path output_path_;
  OutputStreamInfo output_;
  bool record_to_file_ = true;

  GstElement* pipeline_ = nullptr;
  GstElement* source_ = nullptr;
  GstElement* caps_filter_ = nullptr;
  GstElement* tee_ = nullptr;
  GstElement* queue_mux_ = nullptr;
  GstElement* queue_app_ = nullptr;
  GstElement* app_sink_ = nullptr;
  GstElement* wavenc_ = nullptr;
  GstElement* sink_ = nullptr;

  std::atomic<GstClockTime> base_time_ns_{GST_CLOCK_TIME_NONE};
  std::atomic<uint64_t> seq_{0};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> failed_{false};
  std::atomic<bool> force_stop_bus_watch_{false};
  std::atomic<bool> bus_thread_done_{false};
  std::thread bus_thread_;

  mutable std::mutex state_mu_;
  std::string failure_reason_;
  PcmCallback pcm_callback_;

  std::vector<std::unique_ptr<ProbeContext>> probe_contexts_;
  std::vector<std::unique_ptr<QueueSignalContext>> queue_contexts_;
};

}  // namespace rkinfra
