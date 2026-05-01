#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

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
class IFaceExpressionProcessor;
}  // namespace rkstudio::vision

namespace rkinfra {
class ZenohPublisher;
}  // namespace rkinfra

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

  explicit VisionEngine(QObject* parent = nullptr);
  ~VisionEngine() override;

  void LoadBoardConfig(const BoardConfig& board_config);
  void ApplySessionProfile(const SessionProfile& profile);
  void SetState(AppState state);
  void SetSessionWriter(SessionWriter* session_writer);
  void SetZenohPublisher(rkinfra::ZenohPublisher* zenoh_publisher);
  void SetCallbacks(Callbacks callbacks);

  bool ToggleMediapipe(bool enable, std::string* err);
  bool ToggleYolo(bool enable, std::string* err);
  bool ToggleFaceExpression(bool enable, std::string* err);
  bool SyncForState(AppState state, std::string* err);
  void StopPipelines();
  void StopAll();

  bool mediapipe_enabled() const { return mediapipe_enabled_; }
  bool yolo_enabled() const { return yolo_enabled_; }
  bool face_expression_enabled() const { return face_expression_enabled_; }

 signals:
  void MediapipeResultReady(rkstudio::vision::MediapipeResult result);
  void YoloResultReady(rkstudio::vision::YoloResult result);
  void FaceExpressionResultReady(rkstudio::vision::FaceExpressionResult result);

 private:
  bool StartMediapipeProcessor();
  void StopMediapipeProcessor();
  std::unique_ptr<V4l2Pipeline> BuildVisionPipeline(
      const std::string& camera_id,
      const std::string& suffix,
      int fps,
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

  bool StartFaceExpressionProcessor();
  void StopFaceExpressionProcessor();
  std::unique_ptr<V4l2Pipeline> BuildFaceExpressionPipeline(std::string* err);
  bool StartFaceExpressionPipeline(std::string* err);
  void StopFaceExpressionPipeline();
  void OnFaceExpressionSample(GstSample* sample);
  void PollFaceExpressionResults();

  BoardConfig board_config_;
  SessionProfile session_profile_;
  AppState state_ = AppState::kIdle;
  SessionWriter* session_writer_ = nullptr;
  rkinfra::ZenohPublisher* zenoh_publisher_ = nullptr;
  Callbacks callbacks_;
  FrameConverter frame_converter_;

  std::unique_ptr<V4l2Pipeline> mediapipe_pipeline_;
  std::unique_ptr<rkstudio::vision::IMediapipeProcessor> mediapipe_processor_;
  std::unique_ptr<V4l2Pipeline> yolo_pipeline_;
  std::unique_ptr<rkstudio::vision::IYoloProcessor> yolo_processor_;
  std::unique_ptr<V4l2Pipeline> face_expression_pipeline_;
  std::unique_ptr<rkstudio::vision::IFaceExpressionProcessor> face_expression_processor_;
  std::string mediapipe_camera_id_;
  std::string yolo_camera_id_;
  std::string face_expression_camera_id_;
  QTimer* mediapipe_poll_timer_ = nullptr;
  QTimer* yolo_poll_timer_ = nullptr;
  QTimer* face_expression_poll_timer_ = nullptr;
  std::mutex mediapipe_frame_mu_;
  std::mutex yolo_frame_mu_;
  std::mutex face_expression_frame_mu_;
  bool mediapipe_enabled_ = false;
  bool yolo_enabled_ = false;
  bool face_expression_enabled_ = false;
  bool mediapipe_logged_path_ = false;
  bool yolo_logged_path_ = false;
  bool face_expression_logged_path_ = false;
};

}  // namespace rkstudio::media

Q_DECLARE_METATYPE(rkstudio::vision::MediapipeResult)
Q_DECLARE_METATYPE(rkstudio::vision::YoloResult)
Q_DECLARE_METATYPE(rkstudio::vision::FaceExpressionResult)
