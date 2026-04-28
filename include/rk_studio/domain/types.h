#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rkstudio {

enum class AppState {
  kIdle,
  kPreviewing,
  kRecording,
  kStreaming,
  kError,
};

struct CameraNodeSet {
  std::string id;
  std::string record_device;
  std::string input_format = "NV12";
  std::string io_mode = "dmabuf";
  int record_width = 1920;
  int record_height = 1080;
  int preview_width = 640;
  int preview_height = 360;
  int fps = 30;
  int bitrate = 8'000'000;
};

struct AudioSource {
  std::string id = "mic0";
  std::string device = "hw:0,0";
  int rate = 16'000;
  int channels = 2;
};

struct MediapipeHardwareConfig {
  std::string detector_model;   // resolved at runtime if empty
  std::string landmark_model;   // resolved at runtime if empty
};

struct YoloHardwareConfig {
  std::string model;             // resolved at runtime if empty
  std::vector<std::string> class_names;
  int fps = 5;
  double confidence_threshold = 0.25;
  double nms_threshold = 0.45;
  int max_detections = 50;
};

struct RtspConfig {
  int port = 8554;
  std::string codec = "h265";
  int bitrate = 1'800'000;
  int width = 480;
  int height = 272;
  std::vector<std::string> mounts{"cam0", "cam1", "cam2", "cam3"};
};

struct ZenohConfig {
  std::string mode = "peer";
  std::vector<std::string> connect;
  std::vector<std::string> listen;
  std::string key_prefix = "rk_studio";
};

struct EntityRegistrationConfig {
  std::string entity_id = "helmet_001";
  std::string display_name = "张三的头盔";
  std::string owner = "human_zhangsan_001";
  std::string device_type = "helmet";
  std::string provides_channels = "video_out,heartrate,eeg";
  std::string video_stream_url = "ws://192.168.1.20:9000/stream/zhangsan_helmet";
};

struct BoardConfig {
  std::vector<CameraNodeSet> cameras;
  std::vector<AudioSource> audio_sources;
  std::vector<std::string> sink_priority{"ximagesink", "glimagesink"};
  std::optional<MediapipeHardwareConfig> mediapipe;
  std::optional<YoloHardwareConfig> yolo;
  std::optional<RtspConfig> rtsp;
  std::optional<ZenohConfig> zenoh;
  EntityRegistrationConfig entity_registration;
};

struct SessionProfile {
  std::vector<std::string> preview_cameras;
  std::vector<std::string> record_cameras;
  std::string output_dir = "./records";
  std::string prefix = "session";
  std::string audio_source = "mic0";
  std::string selected_mediapipe_camera;
  std::string selected_yolo_camera;
  int preview_rows = 2;
  int preview_cols = 2;
  int gop = 30;
};

struct TelemetryEvent {
  uint64_t monotonic_ns = 0;
  std::string stream_id;
  uint64_t seq = 0;
  int64_t pts_ns = -1;
  std::string category;
  std::string stage;
  std::string status;
  std::string reason;
};

const CameraNodeSet* FindCamera(const BoardConfig& config, const std::string& id);
const AudioSource* FindAudioSource(const BoardConfig& config, const std::string& id);
std::vector<std::string> UnionCameraIds(const SessionProfile& profile);

}  // namespace rkstudio
