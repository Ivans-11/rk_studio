#include "rk_studio/runtime/runtime_manager.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace rkstudio::runtime {
namespace {

std::vector<std::string> ActiveVisionPreviewCameras(
    const SessionProfile& profile,
    bool mediapipe_enabled,
    bool yolo_enabled) {
  std::vector<std::string> cameras;
  auto add_if_needed = [&cameras](const std::string& camera_id) {
    if (camera_id.empty()) {
      return;
    }
    if (std::find(cameras.begin(), cameras.end(), camera_id) == cameras.end()) {
      cameras.push_back(camera_id);
    }
  };

  if (mediapipe_enabled) {
    add_if_needed(profile.selected_mediapipe_camera);
  }
  if (yolo_enabled) {
    add_if_needed(profile.selected_yolo_camera);
  }
  return cameras;
}

}  // namespace

RuntimeManager::RuntimeManager(QObject* parent) : QObject(parent) {
  qRegisterMetaType<rkstudio::AppState>();
  qRegisterMetaType<rkstudio::TelemetryEvent>();
  qRegisterMetaType<rkstudio::vision::MediapipeResult>();
  qRegisterMetaType<rkstudio::vision::YoloResult>();

  media_engine_ = new media::MediaEngine(this);
  vision_engine_ = new media::VisionEngine(this);

  connect(media_engine_, &media::MediaEngine::TelemetryObserved,
          this, &RuntimeManager::TelemetryObserved);
  connect(media_engine_, &media::MediaEngine::PreviewCameraFailed,
          this, &RuntimeManager::PreviewCameraFailed);
  connect(media_engine_, &media::MediaEngine::FatalCameraFailure,
          this, &RuntimeManager::OnFatalCameraFailure);

  connect(vision_engine_, &media::VisionEngine::MediapipeFrameReady,
          this, &RuntimeManager::MediapipeFrameReady);
  connect(vision_engine_, &media::VisionEngine::MediapipeResultReady,
          this, &RuntimeManager::MediapipeResultReady);
  connect(vision_engine_, &media::VisionEngine::YoloFrameReady,
          this, &RuntimeManager::YoloFrameReady);
  connect(vision_engine_, &media::VisionEngine::YoloResultReady,
          this, &RuntimeManager::YoloResultReady);

  vision_engine_->SetCallbacks({
      [this](const TelemetryEvent& event) {
        media_engine_->ObserveTelemetry(event);
      },
      [this](const std::string& camera_id, const std::string& reason, bool fatal) {
        OnVisionCameraError(camera_id, reason, fatal);
      },
  });
  vision_engine_->SetPreviewControls({
      [this](const std::string& camera_id) {
        media_engine_->StopPreviewPipeline(camera_id);
      },
      [this](const std::string& camera_id, std::string* err) {
        return media_engine_->RestorePreviewPipeline(camera_id, err);
      },
  });
}

RuntimeManager::~RuntimeManager() {
  StopAll();
}

void RuntimeManager::LoadBoardConfig(const BoardConfig& board_config) {
  media_engine_->LoadBoardConfig(board_config);
  vision_engine_->LoadBoardConfig(board_config);
}

void RuntimeManager::ApplySessionProfile(const SessionProfile& profile) {
  media_engine_->ApplySessionProfile(profile);
  vision_engine_->ApplySessionProfile(profile);
}

bool RuntimeManager::StartPreview(std::string* err) {
  if (state_ != AppState::kIdle) {
    if (err) *err = "cannot start preview in current state";
    return false;
  }

  const auto excluded_preview_cameras = ActiveVisionPreviewCameras(
      media_engine_->session_profile(),
      vision_engine_->mediapipe_enabled(),
      vision_engine_->yolo_enabled());
  if (!media_engine_->StartPreview(excluded_preview_cameras, err)) {
    EnterErrorState();
    return false;
  }

  vision_engine_->SetSessionWriter(nullptr);
  if (!vision_engine_->SyncForState(AppState::kPreviewing, err)) {
    media_engine_->StopPreview();
    EnterErrorState();
    return false;
  }

  SetState(AppState::kPreviewing);
  return true;
}

bool RuntimeManager::StartRecording(std::string* err) {
  if (state_ != AppState::kIdle && state_ != AppState::kPreviewing) {
    if (err) *err = "cannot start recording in current state";
    return false;
  }

  if (state_ == AppState::kPreviewing) {
    media_engine_->StopPreview();
  }

  if (!media_engine_->StartRecording(err)) {
    EnterErrorState();
    return false;
  }

  vision_engine_->SetSessionWriter(media_engine_->session_writer());
  if (!vision_engine_->SyncForState(AppState::kRecording, err)) {
    vision_engine_->SetSessionWriter(nullptr);
    media_engine_->StopRecording(false);
    EnterErrorState();
    return false;
  }

  SetState(AppState::kRecording);
  return true;
}

bool RuntimeManager::StartRtsp(std::string* err) {
  if (state_ != AppState::kIdle && state_ != AppState::kPreviewing) {
    if (err) *err = "cannot start RTSP in current state";
    return false;
  }

  if (state_ == AppState::kPreviewing) {
    media_engine_->StopPreview();
  }

  if (!media_engine_->StartRtsp(err)) {
    EnterErrorState();
    return false;
  }

  SetState(AppState::kStreaming);
  return true;
}

void RuntimeManager::StopRecording() {
  if (state_ != AppState::kRecording) {
    return;
  }
  vision_engine_->StopPipelines();
  vision_engine_->SetSessionWriter(nullptr);
  media_engine_->StopRecording(true);
  std::string err;
  if (!vision_engine_->SyncForState(AppState::kIdle, &err)) {
    EnterErrorState();
    return;
  }
  SetState(AppState::kIdle);
}

void RuntimeManager::StopPreview() {
  if (state_ != AppState::kPreviewing) {
    return;
  }
  media_engine_->StopPreview();
  SetState(AppState::kIdle);
}

void RuntimeManager::StopRtsp() {
  if (state_ != AppState::kStreaming) {
    return;
  }
  media_engine_->StopRtsp();
  SetState(AppState::kIdle);
}

void RuntimeManager::StopAll() {
  if (state_ == AppState::kRecording) {
    vision_engine_->StopAll();
    vision_engine_->SetSessionWriter(nullptr);
    media_engine_->StopRecording(true);
  } else if (state_ == AppState::kStreaming) {
    media_engine_->StopRtsp();
    vision_engine_->StopAll();
  } else {
    vision_engine_->StopAll();
    media_engine_->StopAll();
  }
  SetState(AppState::kIdle);
}

void RuntimeManager::BindPreviewWindow(const std::string& camera_id, WId window_id) {
  media_engine_->BindPreviewWindow(camera_id, window_id);
}

bool RuntimeManager::ToggleMediapipe(bool enable, std::string* err) {
  vision_engine_->SetState(state_);
  vision_engine_->SetSessionWriter(state_ == AppState::kRecording ? media_engine_->session_writer() : nullptr);
  return vision_engine_->ToggleMediapipe(enable, err);
}

bool RuntimeManager::ToggleYolo(bool enable, std::string* err) {
  vision_engine_->SetState(state_);
  vision_engine_->SetSessionWriter(state_ == AppState::kRecording ? media_engine_->session_writer() : nullptr);
  return vision_engine_->ToggleYolo(enable, err);
}

bool RuntimeManager::mediapipe_enabled() const {
  return vision_engine_->mediapipe_enabled();
}

bool RuntimeManager::yolo_enabled() const {
  return vision_engine_->yolo_enabled();
}

const BoardConfig& RuntimeManager::board_config() const {
  return media_engine_->board_config();
}

const SessionProfile& RuntimeManager::session_profile() const {
  return media_engine_->session_profile();
}

void RuntimeManager::SetState(AppState state) {
  const bool changed = state_ != state;
  state_ = state;
  vision_engine_->SetState(state_);
  if (changed) {
    emit StateChanged(state_);
  }
}

void RuntimeManager::EnterErrorState() {
  vision_engine_->StopPipelines();
  vision_engine_->SetSessionWriter(nullptr);
  SetState(AppState::kError);
}

void RuntimeManager::OnVisionCameraError(
    const std::string& camera_id,
    const std::string& reason,
    bool fatal) {
  emit PreviewCameraFailed(QString::fromStdString(camera_id), QString::fromStdString(reason), fatal);
  if (fatal && state_ == AppState::kRecording) {
    OnFatalCameraFailure();
  }
}

void RuntimeManager::OnFatalCameraFailure() {
  if (state_ != AppState::kRecording) {
    return;
  }
  vision_engine_->StopPipelines();
  vision_engine_->SetSessionWriter(nullptr);
  media_engine_->StopRecording(false);
  SetState(AppState::kError);
}

}  // namespace rkstudio::runtime
