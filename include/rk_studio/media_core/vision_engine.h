#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <QImage>
#include <QObject>
#include <QTimer>

#include "rk_studio/domain/types.h"
#include "rk_studio/media_core/frame_converter.h"
#include "rk_studio/vision_core/vision_types.h"

#ifndef Q_MOC_RUN
#include "rk_studio/media_core/v4l2_pipeline.h"
#endif

struct _GstSample;
typedef struct _GstSample GstSample;

namespace rkstudio::vision {
class IMediapipeProcessor;
class IYoloProcessor;
}  // namespace rkstudio::vision

namespace rkstudio::media {

class SessionWriter;
class V4l2Pipeline;

class VisionEngine : public QObject {
  Q_OBJECT

 public:
  struct Callbacks {
    std::function<void(const TelemetryEvent&)> emit_telemetry;
    std::function<void(const std::string&, const std::string&, bool)> camera_error;
  };

  struct PreviewControls {
    std::function<void(const std::string&)> stop_preview_pipeline;
    std::function<bool(const std::string&, std::string*)> restore_preview_pipeline;
  };

  explicit VisionEngine(QObject* parent = nullptr);
  ~VisionEngine() override;

  void LoadBoardConfig(const BoardConfig& board_config);
  void ApplySessionProfile(const SessionProfile& profile);
  void SetState(AppState state);
  void SetSessionWriter(SessionWriter* session_writer);
  void SetCallbacks(Callbacks callbacks);
  void SetPreviewControls(PreviewControls controls);

  bool ToggleMediapipe(bool enable, std::string* err);
  bool ToggleYolo(bool enable, std::string* err);
  bool StartActivePipelines(AppState state, std::string* err);
  void StopPipelines();
  void StopAll();

  bool mediapipe_enabled() const { return mediapipe_enabled_; }
  bool yolo_enabled() const { return yolo_enabled_; }

 signals:
  void MediapipeFrameReady(QString camera_id, QImage image);
  void MediapipeResultReady(rkstudio::vision::MediapipeResult result);
  void YoloResultReady(rkstudio::vision::YoloResult result);

 private:
  bool StartMediapipeProcessor();
  void StopMediapipeProcessor();
  std::unique_ptr<V4l2Pipeline> BuildVisionPipeline(
      const std::string& camera_id,
      const std::string& suffix,
      std::function<void(GstSample*)> sample_callback,
      std::string* err);
  std::unique_ptr<V4l2Pipeline> BuildMediapipePipeline(std::string* err);
  bool StartMediapipePipeline(std::string* err);
  void StopMediapipePipeline();
  void OnMediapipeSample(GstSample* sample);
  void PollMediapipeResults();

  bool StartYoloProcessor();
  void StopYoloProcessor();
  std::unique_ptr<V4l2Pipeline> BuildYoloPipeline(std::string* err);
  bool StartYoloPipeline(std::string* err);
  void StopYoloPipeline();
  void OnYoloSample(GstSample* sample);
  void PollYoloResults();

  BoardConfig board_config_;
  SessionProfile session_profile_;
  AppState state_ = AppState::kIdle;
  SessionWriter* session_writer_ = nullptr;
  Callbacks callbacks_;
  PreviewControls preview_controls_;
  FrameConverter frame_converter_;

  std::unique_ptr<V4l2Pipeline> mediapipe_pipeline_;
  std::unique_ptr<rkstudio::vision::IMediapipeProcessor> mediapipe_processor_;
  std::unique_ptr<V4l2Pipeline> yolo_pipeline_;
  std::unique_ptr<rkstudio::vision::IYoloProcessor> yolo_processor_;
  std::string mediapipe_camera_id_;
  std::string yolo_camera_id_;
  QTimer* mediapipe_poll_timer_ = nullptr;
  QTimer* yolo_poll_timer_ = nullptr;
  QImage latest_mediapipe_frame_;
  std::mutex mediapipe_frame_mu_;
  std::atomic<uint64_t> last_mediapipe_submit_ns_{0};
  std::atomic<uint64_t> last_yolo_submit_ns_{0};
  std::chrono::steady_clock::time_point last_mediapipe_frame_emit_{};
  bool mediapipe_enabled_ = false;
  bool yolo_enabled_ = false;
  bool mediapipe_logged_path_ = false;
  bool yolo_logged_path_ = false;
};

}  // namespace rkstudio::media

Q_DECLARE_METATYPE(rkstudio::vision::MediapipeResult)
Q_DECLARE_METATYPE(rkstudio::vision::YoloResult)
