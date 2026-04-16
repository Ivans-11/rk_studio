#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <QObject>
#include <QImage>
#include <QTimer>
#include <QtGui/qwindowdefs.h>

#include "rk_studio/domain/types.h"
#include "rk_studio/ai_core/ai_types.h"

#ifndef Q_MOC_RUN
#include "rk_studio/domain/session.h"
#include "rk_studio/media_core/camera_pipeline.h"
#include "rk_studio/media_core/rtsp_server.h"
#endif

namespace rkstudio::ai {
class IAiProcessor;
}

struct _GstSample;
typedef struct _GstSample GstSample;

namespace rkinfra {
class GstAudioRecorder;
struct OutputStreamInfo;
}  // namespace rkinfra

namespace rkstudio::media {

class SessionWriter;

class MediaEngine : public QObject {
  Q_OBJECT

 public:
  explicit MediaEngine(QObject* parent = nullptr);
  ~MediaEngine() override;

  void LoadBoardConfig(const BoardConfig& board_config);
  void ApplySessionProfile(const SessionProfile& profile);
  bool StartPreview(std::string* err);
  bool StartRecording(std::string* err);
  bool StartRtsp(std::string* err);
  void StopRecording();
  void StopRtsp();
  void StopAll();

  void BindPreviewWindow(const std::string& camera_id, WId window_id);
  void SetAiEnabled(bool enabled);
  bool ai_enabled() const { return ai_enabled_; }
  const BoardConfig& board_config() const;
  const SessionProfile& session_profile() const;
  AppState state() const;

 signals:
  void StateChanged(rkstudio::AppState state);
  void TelemetryObserved(rkstudio::TelemetryEvent event);
  void PreviewCameraFailed(QString camera_id, QString reason, bool fatal);
  void AiFrameReady(QString camera_id, QImage image);
  void AiResultReady(rkstudio::ai::AiResult result);

 private:
  using CameraMap = std::map<std::string, std::unique_ptr<CameraPipeline>>;

  bool RebuildPipelines(bool recording, std::string* err);
  void StopPipelines();
  void EmitTelemetry(const TelemetryEvent& event);
  void OnCameraError(const std::string& camera_id, const std::string& reason, bool fatal);
  void FinalizeRecording(bool ok);

  bool StartAiProcessor();
  void StopAiProcessor();
  void OnAiSample(GstSample* sample);
  void PollAiResults();

  BoardConfig board_config_;
  SessionProfile session_profile_;
  AppState state_ = AppState::kIdle;
  CameraMap cameras_;
  std::map<std::string, WId> preview_window_ids_;
  std::unique_ptr<SessionWriter> session_writer_;
  std::unique_ptr<rkinfra::GstAudioRecorder> audio_recorder_;
  std::unique_ptr<rkstudio::ai::IAiProcessor> ai_processor_;
  std::string ai_camera_id_;
  QTimer* ai_poll_timer_ = nullptr;
  QImage latest_ai_frame_;
  std::mutex ai_frame_mu_;
  std::chrono::steady_clock::time_point last_ai_frame_emit_{};
  bool ai_enabled_ = false;
  std::unique_ptr<RtspServer> rtsp_server_;
};

}  // namespace rkstudio::media

Q_DECLARE_METATYPE(rkstudio::AppState)
Q_DECLARE_METATYPE(rkstudio::TelemetryEvent)
Q_DECLARE_METATYPE(rkstudio::ai::AiResult)
