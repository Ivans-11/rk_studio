#include "rk_studio/runtime/runtime_manager.h"

namespace rkstudio::runtime {
RuntimeManager::RuntimeManager(QObject* parent) : QObject(parent) {
  qRegisterMetaType<rkstudio::AppState>();
  qRegisterMetaType<rkstudio::TelemetryEvent>();
  qRegisterMetaType<rkstudio::vision::MediapipeResult>();
  qRegisterMetaType<rkstudio::vision::YoloResult>();

  media_engine_ = new media::MediaEngine(this);
  vision_engine_ = new media::VisionEngine(this);
  vision_engine_->SetZenohPublisher(&zenoh_publisher_);

  connect(media_engine_, &media::MediaEngine::TelemetryObserved,
          this, &RuntimeManager::TelemetryObserved);
  connect(media_engine_, &media::MediaEngine::PreviewCameraFailed,
          this, &RuntimeManager::PreviewCameraFailed);
  connect(media_engine_, &media::MediaEngine::FatalCameraFailure,
          this, &RuntimeManager::OnFatalCameraFailure);

  connect(vision_engine_, &media::VisionEngine::MediapipeResultReady,
          this, &RuntimeManager::MediapipeResultReady);
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

  if (!media_engine_->StartPreview(err)) {
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

bool RuntimeManager::StartZenoh(std::string* err) {
  if (state_ != AppState::kIdle &&
      state_ != AppState::kPreviewing &&
      state_ != AppState::kStreaming) {
    if (err) *err = "cannot start Zenoh in current state";
    return false;
  }
  if (!vision_engine_->mediapipe_enabled() && !vision_engine_->yolo_enabled()) {
    if (err) *err = "start Mediapipe or YOLO before starting Zenoh";
    return false;
  }
  if (!media_engine_->board_config().zenoh.has_value()) {
    if (err) *err = "no [zenoh] section in board config";
    return false;
  }
  return zenoh_publisher_.Start(*media_engine_->board_config().zenoh, err);
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

void RuntimeManager::StopZenoh() {
  if (state_ == AppState::kRecording) {
    return;
  }
  zenoh_publisher_.Stop();
}

void RuntimeManager::StopAll() {
  zenoh_publisher_.Stop();
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
  if (state_ == AppState::kRecording) {
    if (err) *err = "cannot change Mediapipe while recording";
    return false;
  }
  vision_engine_->SetState(state_);
  vision_engine_->SetSessionWriter(nullptr);
  if (!vision_engine_->ToggleMediapipe(enable, err)) {
    return false;
  }
  if (!vision_engine_->mediapipe_enabled() && !vision_engine_->yolo_enabled()) {
    zenoh_publisher_.Stop();
  }
  return true;
}

bool RuntimeManager::ToggleYolo(bool enable, std::string* err) {
  if (state_ == AppState::kRecording) {
    if (err) *err = "cannot change YOLO while recording";
    return false;
  }
  vision_engine_->SetState(state_);
  vision_engine_->SetSessionWriter(nullptr);
  if (!vision_engine_->ToggleYolo(enable, err)) {
    return false;
  }
  if (!vision_engine_->mediapipe_enabled() && !vision_engine_->yolo_enabled()) {
    zenoh_publisher_.Stop();
  }
  return true;
}

bool RuntimeManager::mediapipe_enabled() const {
  return vision_engine_->mediapipe_enabled();
}

bool RuntimeManager::yolo_enabled() const {
  return vision_engine_->yolo_enabled();
}

bool RuntimeManager::zenoh_enabled() const {
  return zenoh_publisher_.active();
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
