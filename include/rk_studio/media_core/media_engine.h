#pragma once

#include <map>
#include <memory>
#include <string>

#include <QObject>
#include <QImage>
#include <QtGui/qwindowdefs.h>

#include "rk_studio/domain/types.h"
#include "rk_studio/vision_core/vision_types.h"

#ifndef Q_MOC_RUN
#include "rk_studio/domain/session.h"
#include "rk_studio/infra/gst_audio_recorder.h"
#include "rk_studio/media_core/rtsp_server.h"
#include "rk_studio/media_core/v4l2_pipeline.h"
#endif

namespace rkinfra {
struct OutputStreamInfo;
}  // namespace rkinfra

namespace rkstudio::media {

class RtspServer;
class SessionWriter;
class V4l2Pipeline;

class MediaEngine : public QObject {
  Q_OBJECT

 public:
  explicit MediaEngine(QObject* parent = nullptr);
  ~MediaEngine() override;

  void LoadBoardConfig(const BoardConfig& board_config);
  void ApplySessionProfile(const SessionProfile& profile);
  bool StartPreview(std::string* err);
  bool StartRecording(std::string* err);
  bool StartAudioMonitor(std::string* err);
  bool StartRtsp(std::string* err);
  void StopPreview();
  void StopAudioMonitor();
  void StopRecording(bool ok = true);
  void StopRtsp();
  void StopAll();
  void UpdateMediapipeResult(const vision::MediapipeResult& result);
  void UpdateYoloResult(const vision::YoloResult& result);
  void UpdateFaceExpressionResult(const vision::FaceExpressionResult& result);
  void UpdateAudioEventResult(const vision::AudioEventResult& result);
  void ClearMediapipeResult(const std::string& camera_id);
  void ClearYoloResult(const std::string& camera_id);
  void ClearFaceExpressionResult(const std::string& camera_id);
  void ClearAudioEventResult();

  void BindPreviewWindow(const std::string& camera_id, WId window_id);
  void BindPreviewFrameTarget(const std::string& camera_id, bool enabled);
  void SetAudioPcmCallback(rkinfra::GstAudioRecorder::PcmCallback callback);
  void ObserveTelemetry(const TelemetryEvent& event);
  SessionWriter* session_writer() const { return session_writer_.get(); }
  const BoardConfig& board_config() const;
  const SessionProfile& session_profile() const;

 signals:
  void TelemetryObserved(rkstudio::TelemetryEvent event);
  void PreviewCameraFailed(QString camera_id, QString reason, bool fatal);
  void PreviewFrameReady(QString camera_id, QImage frame);
  void FatalCameraFailure();

 private:
  using CameraMap = std::map<std::string, std::unique_ptr<V4l2Pipeline>>;

  bool RebuildPipelines(bool recording, std::string* err);
  std::unique_ptr<V4l2Pipeline> BuildOnePipeline(
      const std::string& camera_id, bool recording, std::string* err);
  void StopPipelines();
  void EmitTelemetry(const TelemetryEvent& event);
  void OnCameraError(const std::string& camera_id, const std::string& reason, bool fatal);
  void FinalizeRecording(bool ok);
  bool StartAudioRecorder(std::string* err);
  void StopAudioRecorder();
  bool BuildAudioRecorder(bool record_to_file, std::string* err);

  BoardConfig board_config_;
  SessionProfile session_profile_;
  CameraMap cameras_;
  std::map<std::string, WId> preview_window_ids_;
  std::map<std::string, bool> preview_frame_targets_;
  std::unique_ptr<SessionWriter> session_writer_;
  std::unique_ptr<rkinfra::GstAudioRecorder> audio_recorder_;
  std::unique_ptr<rkinfra::GstAudioRecorder> audio_monitor_;
  rkinfra::GstAudioRecorder::PcmCallback audio_pcm_callback_;
  std::unique_ptr<RtspServer> rtsp_server_;
};

}  // namespace rkstudio::media

Q_DECLARE_METATYPE(rkstudio::AppState)
Q_DECLARE_METATYPE(rkstudio::TelemetryEvent)
