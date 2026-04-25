#pragma once

#include <map>

#include <QLabel>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>

#include "rk_studio/runtime/runtime_manager.h"
#include "rk_studio/ui/mediapipe_canvas_widget.h"
#include "rk_studio/ui/preview_tile_widget.h"

namespace rkstudio::ui {

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override;

 private slots:
  void LoadConfigFiles();
  void TogglePreview();
  void ToggleRecording();
  void ToggleRtsp();
  void ToggleMediapipe();
  void ToggleYolo();
  void OnStateChanged(rkstudio::AppState state);
  void OnTelemetryObserved(rkstudio::TelemetryEvent event);
  void OnPreviewFailure(QString camera_id, QString reason, bool fatal);
  void OnTileRebound(QString camera_id, WId window_id);
  void OnMediapipeFrame(QString camera_id, QImage image);
  void OnMediapipeResult(rkstudio::vision::MediapipeResult result);
  void OnYoloResult(rkstudio::vision::YoloResult result);

 private:
  void BuildUi();
  void RebuildTiles();
  void SwapMediapipeTile(bool enabling);
  void SetStatus(const QString& text);
  void AppendLog(const QString& line);

  runtime::RuntimeManager* runtime_manager_ = nullptr;
  QString board_config_path_;
  QString profile_path_;

  QWidget* central_ = nullptr;
  QWidget* grid_container_ = nullptr;
  QLabel* state_label_ = nullptr;
  QLabel* summary_label_ = nullptr;
  QPushButton* preview_button_ = nullptr;
  QPushButton* record_button_ = nullptr;
  QPushButton* rtsp_button_ = nullptr;
  QPushButton* mediapipe_toggle_button_ = nullptr;
  QPushButton* yolo_toggle_button_ = nullptr;
  QPlainTextEdit* log_view_ = nullptr;
  std::map<QString, PreviewTileWidget*> tiles_;
  MediapipeCanvasWidget* mediapipe_canvas_ = nullptr;
  int telemetry_ok_skip_counter_ = 0;
  int yolo_empty_skip_counter_ = 0;
};

}  // namespace rkstudio::ui
