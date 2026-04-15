#pragma once

#include <map>

#include <QLabel>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>

#include "rk_studio/media_core/media_engine.h"
#include "rk_studio/ui/ai_canvas_widget.h"
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
  void ToggleAi();
  void OnStateChanged(rkstudio::AppState state);
  void OnTelemetryObserved(rkstudio::TelemetryEvent event);
  void OnPreviewFailure(QString camera_id, QString reason, bool fatal);
  void OnTileRebound(QString camera_id, WId window_id);
  void OnAiFrame(QString camera_id, QImage image);
  void OnAiResult(rkstudio::ai::AiResult result);

 private:
  void BuildUi();
  void RebuildTiles();
  void SetStatus(const QString& text);
  void AppendLog(const QString& line);

  media::MediaEngine* media_engine_ = nullptr;
  QString board_config_path_;
  QString profile_path_;

  QWidget* central_ = nullptr;
  QWidget* grid_container_ = nullptr;
  QLabel* state_label_ = nullptr;
  QLabel* summary_label_ = nullptr;
  QPushButton* preview_button_ = nullptr;
  QPushButton* record_button_ = nullptr;
  QPushButton* rtsp_button_ = nullptr;
  QPushButton* ai_toggle_button_ = nullptr;
  QPlainTextEdit* log_view_ = nullptr;
  std::map<QString, PreviewTileWidget*> tiles_;
  AiCanvasWidget* ai_canvas_ = nullptr;
  int telemetry_ok_skip_counter_ = 0;
};

}  // namespace rkstudio::ui
