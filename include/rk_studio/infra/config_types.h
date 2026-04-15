#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rkinfra {

constexpr uint64_t kDefaultMuxQueueTimeNs = 2'000'000'000ULL;
constexpr uint64_t kDefaultSyncWindowNs = 1'000'000'000ULL;

struct RecordConfig {
  std::string output_dir = "./records";
  std::string prefix = "session";
};

struct VideoStreamConfig {
  std::string id;
  std::string device;
  int width = 1920;
  int height = 1080;
  int fps = 30;
  int bitrate = 8'000'000;
  int gop = 0;
  int queue_capture_buffers = 0;
  uint64_t queue_mux_max_time_ns = 0;
  std::string io_mode = "auto";
  std::string input_format = "NV12";
};

struct AudioConfig {
  std::string id = "mic0";
  std::string device = "hw:0,0";
  int rate = 16'000;
  int channels = 2;
};

struct QueueConfig {
  int video_capture_buffers = 6;
  uint64_t video_mux_max_time_ns = kDefaultMuxQueueTimeNs;
  uint64_t audio_mux_max_time_ns = kDefaultMuxQueueTimeNs;
};

struct SyncConfig {
  int max_delta_ms = 50;
  uint64_t window_ns = kDefaultSyncWindowNs;
};

struct EncoderConfig {
  int gop = 30;
  int force_key_unit_interval_ms = 1000;
};

struct OutputStreamInfo {
  std::string id;
  std::string type;
  std::string device;
  std::string codec;
  std::string output_path;
};

struct AppConfig {
  RecordConfig record;
  std::vector<VideoStreamConfig> video_streams;
  std::optional<AudioConfig> audio;
  QueueConfig queue;
  SyncConfig sync;
  EncoderConfig encoder;
};

}  // namespace rkinfra
