#pragma once

#include <string>

#include <QImage>
#include <QObject>
#include <QtGui/qwindowdefs.h>

#include "rk_studio/domain/types.h"
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
  void StopRecording();
  void StopRtsp();
  void StopAll();

  void BindPreviewWindow(const std::string& camera_id, WId window_id);
  bool ToggleMediapipe(bool enable, std::string* err);
  bool ToggleYolo(bool enable, std::string* err);

  bool mediapipe_enabled() const;
  bool yolo_enabled() const;
  const BoardConfig& board_config() const;
  const SessionProfile& session_profile() const;
  AppState state() const { return state_; }

 signals:
  void StateChanged(rkstudio::AppState state);
  void TelemetryObserved(rkstudio::TelemetryEvent event);
  void PreviewCameraFailed(QString camera_id, QString reason, bool fatal);
  void MediapipeFrameReady(QString camera_id, QImage image);
  void MediapipeResultReady(rkstudio::vision::MediapipeResult result);
  void YoloResultReady(rkstudio::vision::YoloResult result);

 private:
  void SetState(AppState state);
  void EnterErrorState();
  void OnVisionCameraError(const std::string& camera_id, const std::string& reason, bool fatal);
  void OnFatalCameraFailure();

  media::MediaEngine* media_engine_ = nullptr;
  media::VisionEngine* vision_engine_ = nullptr;
  AppState state_ = AppState::kIdle;
};

}  // namespace rkstudio::runtime
