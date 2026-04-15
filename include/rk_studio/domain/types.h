#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rk_studio/infra/config_types.h"
#include "rk_studio/infra/telemetry.h"

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

using AudioSource = rkinfra::AudioConfig;

struct AiHardwareConfig {
  std::string detector_model;
  std::string landmark_model;
  bool allow_rga = true;
};

struct RtspConfig {
  int port = 8554;
  std::string codec = "h265";
};

struct BoardConfig {
  std::vector<CameraNodeSet> cameras;
  std::vector<AudioSource> audio_sources;
  std::vector<std::string> sink_priority{"ximagesink", "glimagesink"};
  std::optional<AiHardwareConfig> ai;
  std::optional<RtspConfig> rtsp;
};

struct SessionProfile {
  std::vector<std::string> preview_cameras;
  std::vector<std::string> record_cameras;
  std::string output_dir = "./records";
  std::string prefix = "session";
  std::string audio_source = "mic0";
  std::string selected_ai_camera;
  int preview_rows = 2;
  int preview_cols = 2;
  int gop = 30;
};

using TelemetryEvent = rkinfra::StreamEvent;

const CameraNodeSet* FindCamera(const BoardConfig& config, const std::string& id);
const AudioSource* FindAudioSource(const BoardConfig& config, const std::string& id);
std::vector<std::string> UnionCameraIds(const SessionProfile& profile);

}  // namespace rkstudio
