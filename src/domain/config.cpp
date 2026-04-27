#include "rk_studio/domain/config.h"

#include <algorithm>
#include <filesystem>
#include <string_view>
#include <unordered_set>

#include <toml.hpp>

namespace rkstudio {
namespace {

namespace fs = std::filesystem;

// Resolve a model file relative to the board config directory.
// Search order: models/<name> next to config, then ../models/<name> from config.
std::string ResolveModelPath(const std::string& config_path, const std::string& filename) {
  const fs::path config_dir = fs::path(config_path).parent_path();
  const fs::path candidates[] = {
      config_dir / "models" / filename,
      config_dir / ".." / "models" / filename,
  };
  for (const auto& candidate : candidates) {
    if (fs::exists(candidate)) {
      return fs::canonical(candidate).string();
    }
  }
  return {};
}

std::string ResolveConfigRelativePath(const std::string& config_path, const std::string& value) {
  if (value.empty()) {
    return {};
  }
  const fs::path input(value);
  if (input.is_absolute()) {
    return input.string();
  }
  const fs::path resolved = fs::path(config_path).parent_path() / input;
  if (fs::exists(resolved)) {
    return fs::canonical(resolved).string();
  }
  return resolved.lexically_normal().string();
}

bool RejectUnknownKeys(const toml::table& table,
                      const std::unordered_set<std::string>& allowed,
                      const std::string& section_path,
                      std::string* err) {
  for (const auto& [key, node] : table) {
    (void)node;
    const std::string name = std::string(key.str());
    if (allowed.find(name) != allowed.end()) {
      continue;
    }
    if (err) {
      *err = "unknown key '" + (section_path.empty() ? name : section_path + "." + name) + "'";
    }
    return false;
  }
  return true;
}

template <typename T>
bool AssignValue(const toml::table& table,
                 const char* key,
                 T* out,
                 const std::string& section_path,
                 std::string* err) {
  if (const auto* node = table.get(key)) {
    if (const auto value = node->value<T>()) {
      *out = *value;
      return true;
    }
    if (err) {
      *err = "invalid value for '" + section_path + "." + key + "'";
    }
    return false;
  }
  return true;
}

bool AssignStringArray(const toml::table& table,
                       const char* key,
                       std::vector<std::string>* out,
                       const std::string& section_path,
                       std::string* err) {
  if (const auto* node = table.get(key)) {
    const auto* array = node->as_array();
    if (!array) {
      if (err) {
        *err = "invalid value for '" + section_path + "." + key + "'";
      }
      return false;
    }
    out->clear();
    for (const auto& entry : *array) {
      if (const auto value = entry.value<std::string>()) {
        out->push_back(*value);
      } else {
        if (err) {
          *err = "invalid string item in '" + section_path + "." + key + "'";
        }
        return false;
      }
    }
  }
  return true;
}

bool ParseCamera(const std::string& camera_id,
                 const toml::table& table,
                 CameraNodeSet* camera,
                 std::string* err) {
  static const std::unordered_set<std::string> kAllowed{
      "record_device", "input_format", "io_mode", "record_width",
      "record_height", "preview_width", "preview_height", "fps", "bitrate"};
  if (!RejectUnknownKeys(table, kAllowed, "camera." + camera_id, err)) {
    return false;
  }

  camera->id = camera_id;
  if (!AssignValue(table, "record_device", &camera->record_device, "camera." + camera_id, err) ||
      !AssignValue(table, "input_format", &camera->input_format, "camera." + camera_id, err) ||
      !AssignValue(table, "io_mode", &camera->io_mode, "camera." + camera_id, err) ||
      !AssignValue(table, "record_width", &camera->record_width, "camera." + camera_id, err) ||
      !AssignValue(table, "record_height", &camera->record_height, "camera." + camera_id, err) ||
      !AssignValue(table, "preview_width", &camera->preview_width, "camera." + camera_id, err) ||
      !AssignValue(table, "preview_height", &camera->preview_height, "camera." + camera_id, err) ||
      !AssignValue(table, "fps", &camera->fps, "camera." + camera_id, err) ||
      !AssignValue(table, "bitrate", &camera->bitrate, "camera." + camera_id, err)) {
    return false;
  }
  return true;
}

bool ParseAudio(const std::string& audio_id,
                const toml::table& table,
                AudioSource* source,
                std::string* err) {
  static const std::unordered_set<std::string> kAllowed{"device", "rate", "channels"};
  if (!RejectUnknownKeys(table, kAllowed, "audio." + audio_id, err)) {
    return false;
  }

  source->id = audio_id;
  return AssignValue(table, "device", &source->device, "audio." + audio_id, err) &&
         AssignValue(table, "rate", &source->rate, "audio." + audio_id, err) &&
         AssignValue(table, "channels", &source->channels, "audio." + audio_id, err);
}

bool ValidateBoardConfig(const BoardConfig& config, std::string* err) {
  if (config.cameras.empty()) {
    if (err) {
      *err = "at least one [camera.<id>] section is required";
    }
    return false;
  }
  for (const auto& camera : config.cameras) {
    if (camera.id.empty() || camera.record_device.empty()) {
      if (err) {
        *err = "camera id and record_device must not be empty";
      }
      return false;
    }
    if (camera.record_width <= 0 || camera.record_height <= 0 || camera.preview_width <= 0 ||
        camera.preview_height <= 0 || camera.fps <= 0 || camera.bitrate <= 0) {
      if (err) {
        *err = "camera '" + camera.id + "' dimensions/fps/bitrate must be > 0";
      }
      return false;
    }
  }
  if (config.rtsp.has_value() && config.rtsp->mounts.empty()) {
    if (err) {
      *err = "rtsp.mounts must not be empty";
    }
    return false;
  }
  if (config.yolo.has_value()) {
    const auto& yolo = *config.yolo;
    if (yolo.fps <= 0 || yolo.confidence_threshold <= 0.0 ||
        yolo.nms_threshold <= 0.0 || yolo.max_detections <= 0) {
      if (err) {
        *err = "yolo fps/thresholds/max_detections must be > 0";
      }
      return false;
    }
  }
  if (config.zenoh.has_value()) {
    const auto& zenoh = *config.zenoh;
    if (zenoh.mode != "peer" && zenoh.mode != "client" && zenoh.mode != "router") {
      if (err) {
        *err = "zenoh.mode must be one of: peer, client, router";
      }
      return false;
    }
    if (zenoh.key_prefix.empty()) {
      if (err) {
        *err = "zenoh.key_prefix must not be empty";
      }
      return false;
    }
  }
  const auto& entity = config.entity_registration;
  if (entity.entity_id.empty() || entity.display_name.empty() ||
      entity.owner.empty() || entity.device_type.empty()) {
    if (err) {
      *err = "entity registration fields must not be empty";
    }
    return false;
  }
  return true;
}

bool ValidateSessionProfile(const SessionProfile& profile, std::string* err) {
  if (profile.preview_cameras.empty()) {
    if (err) {
      *err = "session.preview_cameras must not be empty";
    }
    return false;
  }
  if (profile.output_dir.empty() || profile.prefix.empty()) {
    if (err) {
      *err = "session.output_dir and session.prefix must not be empty";
    }
    return false;
  }
  if (profile.preview_rows <= 0 || profile.preview_cols <= 0) {
    if (err) {
      *err = "ui.preview_rows and ui.preview_cols must be > 0";
    }
    return false;
  }
  if (profile.gop <= 0) {
    if (err) {
      *err = "encoder.gop must be > 0";
    }
    return false;
  }
  if (!profile.selected_mediapipe_camera.empty()) {
    const auto& mediapipe_cam = profile.selected_mediapipe_camera;
    const bool in_preview = std::find(profile.preview_cameras.begin(), profile.preview_cameras.end(), mediapipe_cam) !=
                            profile.preview_cameras.end();
    const bool in_record = profile.record_cameras.empty() ||
                           std::find(profile.record_cameras.begin(), profile.record_cameras.end(), mediapipe_cam) !=
                               profile.record_cameras.end();
    if (!in_preview || !in_record) {
      if (err) {
        *err = "selected_mediapipe_camera '" + mediapipe_cam + "' must be in both preview_cameras and record_cameras";
      }
      return false;
    }
  }
  if (!profile.selected_yolo_camera.empty()) {
    const auto& yolo_cam = profile.selected_yolo_camera;
    const bool in_preview = std::find(profile.preview_cameras.begin(), profile.preview_cameras.end(), yolo_cam) !=
                            profile.preview_cameras.end();
    const bool in_record = profile.record_cameras.empty() ||
                           std::find(profile.record_cameras.begin(), profile.record_cameras.end(), yolo_cam) !=
                               profile.record_cameras.end();
    if (!in_preview || !in_record) {
      if (err) {
        *err = "selected_yolo_camera '" + yolo_cam + "' must be in both preview_cameras and record_cameras";
      }
      return false;
    }
  }
  return true;
}

}  // namespace

bool LoadBoardConfig(const std::string& path, BoardConfig* config, std::string* err) {
  if (config == nullptr) {
    if (err) {
      *err = "board config output is null";
    }
    return false;
  }

  BoardConfig parsed;
  toml::table root;
  try {
    root = toml::parse_file(path);
  } catch (const toml::parse_error& ex) {
    if (err) {
      *err = ex.description();
    }
    return false;
  }

  if (const auto* sink = root["sink"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{"priority"};
    if (!RejectUnknownKeys(*sink, kAllowed, "sink", err) ||
        !AssignStringArray(*sink, "priority", &parsed.sink_priority, "sink", err)) {
      return false;
    }
  }

  if (const auto* camera_root = root["camera"].as_table()) {
    for (const auto& [key, node] : *camera_root) {
      const auto* table = node.as_table();
      if (!table) {
        if (err) {
          *err = "camera." + std::string(key.str()) + " must be a table";
        }
        return false;
      }
      CameraNodeSet camera;
      if (!ParseCamera(std::string(key.str()), *table, &camera, err)) {
        return false;
      }
      parsed.cameras.push_back(std::move(camera));
    }
  }

  if (const auto* audio_root = root["audio"].as_table()) {
    for (const auto& [key, node] : *audio_root) {
      const auto* table = node.as_table();
      if (!table) {
        if (err) {
          *err = "audio." + std::string(key.str()) + " must be a table";
        }
        return false;
      }
      AudioSource source;
      if (!ParseAudio(std::string(key.str()), *table, &source, err)) {
        return false;
      }
      parsed.audio_sources.push_back(std::move(source));
    }
  }

  if (const auto* mediapipe_table = root["mediapipe"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{"detector_model", "landmark_model"};
    if (!RejectUnknownKeys(*mediapipe_table, kAllowed, "mediapipe", err)) {
      return false;
    }
    MediapipeHardwareConfig mediapipe;
    if (!AssignValue(*mediapipe_table, "detector_model", &mediapipe.detector_model, "mediapipe", err) ||
        !AssignValue(*mediapipe_table, "landmark_model", &mediapipe.landmark_model, "mediapipe", err)) {
      return false;
    }
    parsed.mediapipe = mediapipe;
  }

  // Auto-resolve Mediapipe model paths from models/ directory if not explicitly set.
  if (!parsed.mediapipe.has_value()) {
    parsed.mediapipe = MediapipeHardwareConfig{};
  }
  auto& mediapipe = *parsed.mediapipe;
  mediapipe.detector_model = ResolveConfigRelativePath(path, mediapipe.detector_model);
  mediapipe.landmark_model = ResolveConfigRelativePath(path, mediapipe.landmark_model);
  if (mediapipe.detector_model.empty()) {
    mediapipe.detector_model = ResolveModelPath(path, "hand_detector.rknn");
  }
  if (mediapipe.landmark_model.empty()) {
    mediapipe.landmark_model = ResolveModelPath(path, "hand_landmarks.rknn");
  }
  // If neither model was found, disable Mediapipe.
  if (mediapipe.detector_model.empty() && mediapipe.landmark_model.empty()) {
    parsed.mediapipe.reset();
  }

  if (const auto* yolo_table = root["yolo"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{
        "model", "class_names", "fps", "confidence_threshold", "nms_threshold", "max_detections"};
    if (!RejectUnknownKeys(*yolo_table, kAllowed, "yolo", err)) {
      return false;
    }
    YoloHardwareConfig yolo;
    if (!AssignValue(*yolo_table, "model", &yolo.model, "yolo", err) ||
        !AssignStringArray(*yolo_table, "class_names", &yolo.class_names, "yolo", err) ||
        !AssignValue(*yolo_table, "fps", &yolo.fps, "yolo", err) ||
        !AssignValue(*yolo_table, "confidence_threshold", &yolo.confidence_threshold, "yolo", err) ||
        !AssignValue(*yolo_table, "nms_threshold", &yolo.nms_threshold, "yolo", err) ||
        !AssignValue(*yolo_table, "max_detections", &yolo.max_detections, "yolo", err)) {
      return false;
    }
    parsed.yolo = yolo;
  }

  if (!parsed.yolo.has_value()) {
    parsed.yolo = YoloHardwareConfig{};
  }
  auto& yolo = *parsed.yolo;
  yolo.model = ResolveConfigRelativePath(path, yolo.model);
  if (yolo.model.empty()) {
    yolo.model = ResolveModelPath(path, "yolo11n_rk3588_int8.rknn");
  }
  if (yolo.model.empty()) {
    yolo.model = ResolveModelPath(path, "yolo11n_rk3588_fp.rknn");
  }
  if (yolo.model.empty()) {
    parsed.yolo.reset();
  }

  if (const auto* rtsp_table = root["rtsp"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{"port", "codec", "bitrate", "mounts"};
    if (!RejectUnknownKeys(*rtsp_table, kAllowed, "rtsp", err)) {
      return false;
    }
    RtspConfig rtsp;
    if (!AssignValue(*rtsp_table, "port", &rtsp.port, "rtsp", err) ||
        !AssignValue(*rtsp_table, "codec", &rtsp.codec, "rtsp", err) ||
        !AssignValue(*rtsp_table, "bitrate", &rtsp.bitrate, "rtsp", err) ||
        !AssignStringArray(*rtsp_table, "mounts", &rtsp.mounts, "rtsp", err)) {
      return false;
    }
    parsed.rtsp = rtsp;
  }

  if (const auto* zenoh_table = root["zenoh"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{"mode", "connect", "listen", "key_prefix"};
    if (!RejectUnknownKeys(*zenoh_table, kAllowed, "zenoh", err)) {
      return false;
    }
    ZenohConfig zenoh;
    if (!AssignValue(*zenoh_table, "mode", &zenoh.mode, "zenoh", err) ||
        !AssignStringArray(*zenoh_table, "connect", &zenoh.connect, "zenoh", err) ||
        !AssignStringArray(*zenoh_table, "listen", &zenoh.listen, "zenoh", err) ||
        !AssignValue(*zenoh_table, "key_prefix", &zenoh.key_prefix, "zenoh", err)) {
      return false;
    }
    parsed.zenoh = zenoh;
  }

  if (const auto* entity_table = root["entity"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{
        "entity_id", "display_name", "owner", "device_type",
        "provides_channels", "video_stream_url"};
    if (!RejectUnknownKeys(*entity_table, kAllowed, "entity", err)) {
      return false;
    }
    EntityRegistrationConfig entity;
    if (!AssignValue(*entity_table, "entity_id", &entity.entity_id, "entity", err) ||
        !AssignValue(*entity_table, "display_name", &entity.display_name, "entity", err) ||
        !AssignValue(*entity_table, "owner", &entity.owner, "entity", err) ||
        !AssignValue(*entity_table, "device_type", &entity.device_type, "entity", err) ||
        !AssignValue(*entity_table, "provides_channels", &entity.provides_channels, "entity", err) ||
        !AssignValue(*entity_table, "video_stream_url", &entity.video_stream_url, "entity", err)) {
      return false;
    }
    parsed.entity_registration = entity;
  }

  if (!ValidateBoardConfig(parsed, err)) {
    return false;
  }

  *config = std::move(parsed);
  return true;
}

bool LoadSessionProfile(const std::string& path, SessionProfile* profile, std::string* err) {
  if (profile == nullptr) {
    if (err) {
      *err = "session profile output is null";
    }
    return false;
  }

  SessionProfile parsed;
  toml::table root;
  try {
    root = toml::parse_file(path);
  } catch (const toml::parse_error& ex) {
    if (err) {
      *err = ex.description();
    }
    return false;
  }

  if (const auto* session = root["session"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{
        "preview_cameras", "record_cameras", "output_dir", "prefix", "audio_source",
        "selected_mediapipe_camera", "selected_yolo_camera"};
    if (!RejectUnknownKeys(*session, kAllowed, "session", err) ||
        !AssignStringArray(*session, "preview_cameras", &parsed.preview_cameras, "session", err) ||
        !AssignStringArray(*session, "record_cameras", &parsed.record_cameras, "session", err) ||
        !AssignValue(*session, "output_dir", &parsed.output_dir, "session", err) ||
        !AssignValue(*session, "prefix", &parsed.prefix, "session", err) ||
        !AssignValue(*session, "audio_source", &parsed.audio_source, "session", err) ||
        !AssignValue(*session, "selected_mediapipe_camera", &parsed.selected_mediapipe_camera, "session", err) ||
        !AssignValue(*session, "selected_yolo_camera", &parsed.selected_yolo_camera, "session", err)) {
      return false;
    }
  }

  if (const auto* encoder = root["encoder"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{"gop"};
    if (!RejectUnknownKeys(*encoder, kAllowed, "encoder", err) ||
        !AssignValue(*encoder, "gop", &parsed.gop, "encoder", err)) {
      return false;
    }
  }

  if (const auto* ui = root["ui"].as_table()) {
    static const std::unordered_set<std::string> kAllowed{"preview_rows", "preview_cols"};
    if (!RejectUnknownKeys(*ui, kAllowed, "ui", err) ||
        !AssignValue(*ui, "preview_rows", &parsed.preview_rows, "ui", err) ||
        !AssignValue(*ui, "preview_cols", &parsed.preview_cols, "ui", err)) {
      return false;
    }
  }

  if (!ValidateSessionProfile(parsed, err)) {
    return false;
  }

  *profile = std::move(parsed);
  return true;
}

}  // namespace rkstudio
