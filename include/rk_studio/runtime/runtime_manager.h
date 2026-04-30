#pragma once

#include <string>

#include <QObject>
#include <QTimer>
#include <QtGui/qwindowdefs.h>

#include "rk_studio/domain/types.h"
#include "rk_studio/infra/zenoh_publisher.h"
#include "rk_studio/media_core/media_engine.h"
#include "rk_studio/media_core/vision_engine.h"

namespace rkstudio::runtime {

class RuntimeManager : public QObject {
  Q_OBJECT

 public:
  explicit RuntimeManager(QObject* parent = nullptr);
  ~RuntimeManager() override;

  void LoadBoardConfig(const BoardConfig& board_config);
  void ApplySessionProfile(const SessionProfile& profile);
  bool StartPreview(std::string* err);
  bool StartRecording(std::string* err);
  bool StartRtsp(std::string* err);
  bool ToggleResultPublishing(std::string* err);
  bool ToggleEntityRegistration(std::string* err);
  void StopPreview();
  void StopRecording();
  void StopRtsp();
  void StopResultPublishing();
  void StopAll();

  void BindPreviewWindow(const std::string& camera_id, WId window_id);
  bool ToggleMediapipe(bool enable, std::string* err);
  bool ToggleYolo(bool enable, std::string* err);

  bool mediapipe_enabled() const;
  bool yolo_enabled() const;
  bool zenoh_enabled() const;
  bool result_publishing_enabled() const;
  bool entity_registered() const;
  const BoardConfig& board_config() const;
  const SessionProfile& session_profile() const;
  AppState state() const { return state_; }

 signals:
  void StateChanged(rkstudio::AppState state);
  void TelemetryObserved(rkstudio::TelemetryEvent event);
  void PreviewCameraFailed(QString camera_id, QString reason, bool fatal);
  void MediapipeResultReady(rkstudio::vision::MediapipeResult result);
  void YoloResultReady(rkstudio::vision::YoloResult result);

 private:
  void SetState(AppState state);
  bool EnsureZenohStarted(std::string* err);
  bool PublishEntityRegistrationAction(const std::string& action, std::string* err);
  void StartEntityRegistrationHeartbeat();
  void StopEntityRegistrationHeartbeat();
  void StopZenohIfIdle();
  void EnterErrorState();
  void OnVisionCameraError(const std::string& camera_id, const std::string& reason, bool fatal);
  void OnFatalCameraFailure();

  media::MediaEngine* media_engine_ = nullptr;
  media::VisionEngine* vision_engine_ = nullptr;
  rkinfra::ZenohPublisher zenoh_publisher_;
  QTimer* entity_registration_timer_ = nullptr;
  bool entity_registered_ = false;
  AppState state_ = AppState::kIdle;
};

}  // namespace rkstudio::runtime
