#include "rk_studio/runtime/runtime_manager.h"

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

namespace rkstudio::runtime {
namespace {

constexpr const char* kEntityRegistryTopic = "zho/entity/registry";

std::string JsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  out << '"';
  return out.str();
}

}  // namespace

RuntimeManager::RuntimeManager(QObject* parent) : QObject(parent) {
  qRegisterMetaType<rkstudio::AppState>();
  qRegisterMetaType<rkstudio::TelemetryEvent>();
  qRegisterMetaType<rkstudio::vision::MediapipeResult>();
  qRegisterMetaType<rkstudio::vision::YoloResult>();
  qRegisterMetaType<rkstudio::vision::FaceExpressionResult>();

  media_engine_ = new media::MediaEngine(this);
  vision_engine_ = new media::VisionEngine(this);
  vision_engine_->SetZenohPublisher(&zenoh_publisher_);
  entity_registration_timer_ = new QTimer(this);
  entity_registration_timer_->setInterval(5000);
  connect(entity_registration_timer_, &QTimer::timeout, this, [this]() {
    if (!entity_registered_) {
      return;
    }
    std::string err;
    if (!EnsureZenohStarted(&err) ||
        !PublishEntityRegistrationAction("REG_REGISTER", &err)) {
      std::cerr << "[zenoh] entity registration heartbeat failed: " << err << "\n";
    }
  });

  connect(media_engine_, &media::MediaEngine::TelemetryObserved,
          this, &RuntimeManager::TelemetryObserved);
  connect(media_engine_, &media::MediaEngine::PreviewCameraFailed,
          this, &RuntimeManager::PreviewCameraFailed);
  connect(media_engine_, &media::MediaEngine::FatalCameraFailure,
          this, &RuntimeManager::OnFatalCameraFailure);

  connect(vision_engine_, &media::VisionEngine::MediapipeResultReady,
          this, [this](const vision::MediapipeResult& result) {
            media_engine_->UpdateMediapipeResult(result);
            emit MediapipeResultReady(result);
          });
  connect(vision_engine_, &media::VisionEngine::YoloResultReady,
          this, [this](const vision::YoloResult& result) {
            media_engine_->UpdateYoloResult(result);
            emit YoloResultReady(result);
          });
  connect(vision_engine_, &media::VisionEngine::FaceExpressionResultReady,
          this, [this](const vision::FaceExpressionResult& result) {
            media_engine_->UpdateFaceExpressionResult(result);
            emit FaceExpressionResultReady(result);
          });

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

bool RuntimeManager::EnsureZenohStarted(std::string* err) {
  if (zenoh_publisher_.active()) {
    return true;
  }
  if (!media_engine_->board_config().zenoh.has_value()) {
    if (err) *err = "no [zenoh] section in board config";
    return false;
  }
  return zenoh_publisher_.Start(*media_engine_->board_config().zenoh, err);
}

bool RuntimeManager::ToggleResultPublishing(std::string* err) {
  if (state_ != AppState::kIdle &&
      state_ != AppState::kPreviewing &&
      state_ != AppState::kStreaming) {
    if (err) *err = "cannot change result publishing in current state";
    return false;
  }

  if (zenoh_publisher_.result_publishing_enabled()) {
    StopResultPublishing();
    return true;
  }

  if (!vision_engine_->mediapipe_enabled() &&
      !vision_engine_->yolo_enabled() &&
      !vision_engine_->face_expression_enabled()) {
    if (err) *err = "start Mediapipe, YOLO, or face expression before publishing recognition results";
    return false;
  }
  if (!EnsureZenohStarted(err)) {
    return false;
  }
  zenoh_publisher_.SetResultPublishingEnabled(true);
  return true;
}

bool RuntimeManager::ToggleEntityRegistration(std::string* err) {
  if (state_ != AppState::kIdle &&
      state_ != AppState::kPreviewing &&
      state_ != AppState::kStreaming) {
    if (err) *err = "cannot change entity registration in current state";
    return false;
  }
  if (!EnsureZenohStarted(err)) {
    return false;
  }

  if (entity_registered_) {
    StopEntityRegistrationHeartbeat();
    if (!PublishEntityRegistrationAction("REG_UNREGISTER", err)) {
      StartEntityRegistrationHeartbeat();
      StopZenohIfIdle();
      return false;
    }
    entity_registered_ = false;
    StopZenohIfIdle();
    return true;
  }

  if (!PublishEntityRegistrationAction("REG_REGISTER", err)) {
    StopZenohIfIdle();
    return false;
  }
  entity_registered_ = true;
  StartEntityRegistrationHeartbeat();
  return true;
}

bool RuntimeManager::PublishEntityRegistrationAction(
    const std::string& action,
    std::string* err) {
  if (!media_engine_->board_config().zenoh.has_value()) {
    if (err) *err = "no [zenoh] section in board config";
    return false;
  }

  const auto& entity = media_engine_->board_config().entity_registration;
  const bool is_register = action == "REG_REGISTER";

  std::ostringstream payload;
  payload << "{"
          << "\"_type\":" << JsonString("ObjectRegistration") << ","
          << "\"entity_id\":" << JsonString(entity.entity_id) << ","
          << "\"action\":" << JsonString(action);
  if (is_register) {
    payload << ",\"display_name\":" << JsonString(entity.display_name)
            << ",\"metadata\":{"
            << "\"owner\":" << JsonString(entity.owner) << ","
            << "\"device_type\":" << JsonString(entity.device_type) << ","
            << "\"provides_channels\":" << JsonString(entity.provides_channels) << ","
            << "\"video_stream_url\":" << JsonString(entity.video_stream_url)
            << "}";
  }
  payload << "}";

  if (!zenoh_publisher_.PublishJson(kEntityRegistryTopic, payload.str())) {
    if (err) *err = "failed to publish entity registration";
    return false;
  }
  return true;
}

void RuntimeManager::StartEntityRegistrationHeartbeat() {
  if (entity_registration_timer_ && !entity_registration_timer_->isActive()) {
    entity_registration_timer_->start();
  }
}

void RuntimeManager::StopEntityRegistrationHeartbeat() {
  if (entity_registration_timer_) {
    entity_registration_timer_->stop();
  }
}

void RuntimeManager::StopZenohIfIdle() {
  if (!entity_registered_ && !zenoh_publisher_.result_publishing_enabled()) {
    zenoh_publisher_.Stop();
  }
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

void RuntimeManager::StopResultPublishing() {
  if (state_ == AppState::kRecording) {
    return;
  }
  zenoh_publisher_.SetResultPublishingEnabled(false);
  StopZenohIfIdle();
}

void RuntimeManager::StopAll() {
  StopEntityRegistrationHeartbeat();
  entity_registered_ = false;
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
  if (!enable) {
    media_engine_->ClearMediapipeResult(
        media_engine_->session_profile().selected_mediapipe_camera);
  }
  if (!vision_engine_->mediapipe_enabled() &&
      !vision_engine_->yolo_enabled() &&
      !vision_engine_->face_expression_enabled()) {
    StopResultPublishing();
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
  if (!enable) {
    media_engine_->ClearYoloResult(media_engine_->session_profile().selected_yolo_camera);
  }
  if (!vision_engine_->mediapipe_enabled() &&
      !vision_engine_->yolo_enabled() &&
      !vision_engine_->face_expression_enabled()) {
    StopResultPublishing();
  }
  return true;
}

bool RuntimeManager::ToggleFaceExpression(bool enable, std::string* err) {
  if (state_ == AppState::kRecording) {
    if (err) *err = "cannot change face expression while recording";
    return false;
  }
  vision_engine_->SetState(state_);
  vision_engine_->SetSessionWriter(nullptr);
  if (!vision_engine_->ToggleFaceExpression(enable, err)) {
    return false;
  }
  if (!enable) {
    media_engine_->ClearFaceExpressionResult(media_engine_->session_profile().selected_face_camera);
  }
  if (!vision_engine_->mediapipe_enabled() &&
      !vision_engine_->yolo_enabled() &&
      !vision_engine_->face_expression_enabled()) {
    StopResultPublishing();
  }
  return true;
}

bool RuntimeManager::mediapipe_enabled() const {
  return vision_engine_->mediapipe_enabled();
}

bool RuntimeManager::yolo_enabled() const {
  return vision_engine_->yolo_enabled();
}

bool RuntimeManager::face_expression_enabled() const {
  return vision_engine_->face_expression_enabled();
}

bool RuntimeManager::zenoh_enabled() const {
  return zenoh_publisher_.active();
}

bool RuntimeManager::result_publishing_enabled() const {
  return zenoh_publisher_.result_publishing_enabled();
}

bool RuntimeManager::entity_registered() const {
  return entity_registered_;
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
