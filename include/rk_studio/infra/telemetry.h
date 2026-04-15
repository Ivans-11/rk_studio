#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rk_studio/infra/config_types.h"

namespace rkinfra {

struct StreamEvent {
  uint64_t monotonic_ns = 0;
  std::string stream_id;
  uint64_t seq = 0;
  int64_t pts_ns = -1;
  std::string category;
  std::string stage;
  std::string status;
  std::string reason;
};

class EventQueue {
 public:
  explicit EventQueue(size_t max_size);

  bool Push(const StreamEvent& event);
  bool Pop(StreamEvent* out);
  void Close();

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<StreamEvent> events_;
  size_t max_size_ = 0;
  bool closed_ = false;
};

class SidecarWriter {
 public:
  SidecarWriter(EventQueue* queue, std::string path);

  bool Start();
  void Stop();

 private:
  void Run();

  EventQueue* queue_ = nullptr;
  std::string path_;
  std::ofstream out_;
  std::thread worker_;
};

struct StreamSyncSummary {
  std::string stream_id;
  bool is_reference = false;
  uint64_t valid_event_count = 0;
  uint64_t matched_window_count = 0;
  bool has_first_sample_offset_ns = false;
  int64_t first_sample_offset_ns = 0;
  bool has_delta_stats = false;
  int64_t min_delta_ns = 0;
  int64_t max_delta_ns = 0;
  double mean_abs_delta_ns = 0.0;
  uint64_t warning_count = 0;
};

struct SyncReport {
  std::string reference_stream_id;
  uint64_t window_size_ns = 0;
  uint64_t warning_threshold_ns = 0;
  std::vector<StreamSyncSummary> streams;
};

class SyncAnalyzer {
 public:
  SyncAnalyzer(SyncConfig config, std::vector<std::string> stream_ids, std::string reference_stream_id);

  void Observe(const StreamEvent& event);
  SyncReport BuildReport() const;

 private:
  struct StreamState {
    uint64_t valid_event_count = 0;
    std::optional<uint64_t> first_valid_monotonic_ns;
    std::map<uint64_t, uint64_t> first_sample_by_window;
  };

  SyncConfig config_;
  std::vector<std::string> stream_ids_;
  std::string reference_stream_id_;

  mutable std::mutex mu_;
  std::map<std::string, StreamState> states_;
};

class TelemetrySink {
 public:
  TelemetrySink(size_t queue_size,
                std::string sidecar_path,
                SyncConfig sync_config,
                std::vector<std::string> stream_ids,
                std::string reference_stream_id);

  bool Start(std::string* err);
  void Record(const StreamEvent& event);
  void Stop();
  SyncReport BuildSyncReport() const;

 private:
  EventQueue queue_;
  SidecarWriter sidecar_writer_;
  SyncAnalyzer sync_analyzer_;
};

}  // namespace rkinfra
