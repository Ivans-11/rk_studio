#include "rk_studio/ui/main_window.h"

#include <algorithm>
#include <map>

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLayoutItem>
#include <QMessageBox>
#include <QStringList>
#include <QVBoxLayout>

#include "rk_studio/domain/config.h"

namespace rkstudio::ui {
namespace {

QString ResolveDefaultConfigPath(const QString& file_name) {
  const QStringList candidates{
      QDir::current().filePath(QStringLiteral("config/%1").arg(file_name)),
      QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../config/%1").arg(file_name)),
      QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("config/%1").arg(file_name)),
  };
  for (const QString& candidate : candidates) {
    const QFileInfo info(candidate);
    if (info.exists() && info.isFile()) {
      return info.canonicalFilePath().isEmpty() ? info.absoluteFilePath() : info.canonicalFilePath();
    }
  }
  return candidates.front();
}

void ClearLayoutWidgets(QWidget* container) {
  if (container == nullptr) {
    return;
  }
  QLayout* layout = container->layout();
  if (layout == nullptr) {
    return;
  }
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->setParent(nullptr);
      delete widget;
    }
    delete item;
  }
  delete layout;
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  runtime_manager_ = new runtime::RuntimeManager(this);

  BuildUi();
  board_config_path_ = ResolveDefaultConfigPath(QStringLiteral("board.toml"));
  profile_path_ = ResolveDefaultConfigPath(QStringLiteral("profile.toml"));
  connect(runtime_manager_, &runtime::RuntimeManager::StateChanged, this, &MainWindow::OnStateChanged);
  connect(runtime_manager_, &runtime::RuntimeManager::TelemetryObserved, this, &MainWindow::OnTelemetryObserved);
  connect(runtime_manager_, &runtime::RuntimeManager::PreviewCameraFailed, this, &MainWindow::OnPreviewFailure);
  connect(runtime_manager_, &runtime::RuntimeManager::MediapipeFrameReady, this, &MainWindow::OnMediapipeFrame);
  connect(runtime_manager_, &runtime::RuntimeManager::MediapipeResultReady, this, &MainWindow::OnMediapipeResult);
  connect(runtime_manager_, &runtime::RuntimeManager::YoloResultReady, this, &MainWindow::OnYoloResult);

  if (QFileInfo::exists(board_config_path_) && QFileInfo::exists(profile_path_)) {
    LoadConfigFiles();
  } else {
    SetStatus(QStringLiteral("未找到默认配置，请确认 config 目录位置"));
    AppendLog(QStringLiteral("默认 board 配置: %1").arg(board_config_path_));
    AppendLog(QStringLiteral("默认 profile 配置: %1").arg(profile_path_));
  }
}

MainWindow::~MainWindow() = default;

void MainWindow::BuildUi() {
  resize(1440, 900);
  central_ = new QWidget(this);
  setCentralWidget(central_);

  auto* root = new QHBoxLayout(central_);
  root->setContentsMargins(16, 16, 16, 16);
  root->setSpacing(16);

  auto* side_panel = new QWidget(central_);
  side_panel->setFixedWidth(320);
  auto* side_layout = new QVBoxLayout(side_panel);
  side_layout->setSpacing(12);

  auto* load_button = new QPushButton(QStringLiteral("加载配置"), side_panel);
  preview_button_ = new QPushButton(QStringLiteral("启动预览"), side_panel);
  record_button_ = new QPushButton(QStringLiteral("启动录制"), side_panel);
  rtsp_button_ = new QPushButton(QStringLiteral("启动 RTSP"), side_panel);
  mediapipe_toggle_button_ = new QPushButton(QStringLiteral("启动 Mediapipe"), side_panel);
  yolo_toggle_button_ = new QPushButton(QStringLiteral("启动 YOLO"), side_panel);
  record_button_->setEnabled(false);
  rtsp_button_->setEnabled(false);
  mediapipe_toggle_button_->setEnabled(false);
  yolo_toggle_button_->setEnabled(false);

  state_label_ = new QLabel(QStringLiteral("状态: Idle"), side_panel);
  summary_label_ = new QLabel(QStringLiteral("等待配置"), side_panel);
  summary_label_->setWordWrap(true);
  log_view_ = new QPlainTextEdit(side_panel);
  log_view_->setReadOnly(true);
  log_view_->setMaximumBlockCount(1000);

  side_layout->addWidget(load_button);
  side_layout->addWidget(preview_button_);
  side_layout->addWidget(record_button_);
  side_layout->addWidget(rtsp_button_);
  side_layout->addWidget(mediapipe_toggle_button_);
  side_layout->addWidget(yolo_toggle_button_);
  side_layout->addWidget(state_label_);
  side_layout->addWidget(summary_label_);
  side_layout->addWidget(log_view_, 1);

  grid_container_ = new QWidget(central_);
  root->addWidget(side_panel);
  root->addWidget(grid_container_, 1);

  connect(load_button, &QPushButton::clicked, this, &MainWindow::LoadConfigFiles);
  connect(preview_button_, &QPushButton::clicked, this, &MainWindow::TogglePreview);
  connect(record_button_, &QPushButton::clicked, this, &MainWindow::ToggleRecording);
  connect(rtsp_button_, &QPushButton::clicked, this, &MainWindow::ToggleRtsp);
  connect(mediapipe_toggle_button_, &QPushButton::clicked, this, &MainWindow::ToggleMediapipe);
  connect(yolo_toggle_button_, &QPushButton::clicked, this, &MainWindow::ToggleYolo);
}

void MainWindow::RebuildTiles() {
  ClearLayoutWidgets(grid_container_);
  tiles_.clear();
  mediapipe_canvas_ = nullptr;

  const auto& profile = runtime_manager_->session_profile();
  auto* grid = new QGridLayout(grid_container_);
  grid->setContentsMargins(0, 0, 0, 0);
  grid->setSpacing(12);

  const int cols = std::max(1, profile.preview_cols);
  for (size_t i = 0; i < profile.preview_cameras.size(); ++i) {
    const QString camera_id = QString::fromStdString(profile.preview_cameras[i]);
    const int row = static_cast<int>(i / cols);
    const int col = static_cast<int>(i % cols);

    if (runtime_manager_->mediapipe_enabled() &&
        profile.preview_cameras[i] == profile.selected_mediapipe_camera) {
      auto* canvas = new MediapipeCanvasWidget(grid_container_);
      canvas->setMinimumSize(320, 180);
      grid->addWidget(canvas, row, col);
      mediapipe_canvas_ = canvas;
      continue;
    }

    auto* tile = new PreviewTileWidget(camera_id, grid_container_);
    connect(tile, &PreviewTileWidget::WindowRebound, this, &MainWindow::OnTileRebound);
    grid->addWidget(tile, row, col);
    tiles_.insert_or_assign(camera_id, tile);
    runtime_manager_->BindPreviewWindow(profile.preview_cameras[i], tile->sink_window_id());
  }
}

void MainWindow::SetStatus(const QString& text) {
  summary_label_->setText(text);
}

void MainWindow::AppendLog(const QString& line) {
  log_view_->appendPlainText(line);
}

void MainWindow::LoadConfigFiles() {
  BoardConfig board_config;
  SessionProfile profile;
  std::string err;
  if (!LoadBoardConfig(board_config_path_.toStdString(), &board_config, &err) ||
      !LoadSessionProfile(profile_path_.toStdString(), &profile, &err)) {
    QMessageBox::critical(this, QStringLiteral("配置错误"), QString::fromStdString(err));
    return;
  }

  if (runtime_manager_->state() != AppState::kIdle) {
    runtime_manager_->StopAll();
  }
  runtime_manager_->LoadBoardConfig(board_config);
  runtime_manager_->ApplySessionProfile(profile);
  RebuildTiles();
  OnStateChanged(runtime_manager_->state());
  SetStatus(QString("已加载 %1 路摄像头").arg(profile.preview_cameras.size()));
}

void MainWindow::TogglePreview() {
  if (runtime_manager_->state() == AppState::kPreviewing) {
    runtime_manager_->StopAll();
  } else if (runtime_manager_->state() == AppState::kIdle) {
    std::string err;
    if (!runtime_manager_->StartPreview(&err)) {
      QMessageBox::warning(this, QStringLiteral("启动失败"), QString::fromStdString(err));
    }
  }
}

void MainWindow::ToggleRecording() {
  if (runtime_manager_->state() == AppState::kRecording) {
    runtime_manager_->StopRecording();
  } else if (runtime_manager_->state() == AppState::kPreviewing ||
             runtime_manager_->state() == AppState::kIdle) {
    std::string err;
    if (!runtime_manager_->StartRecording(&err)) {
      QMessageBox::warning(this, QStringLiteral("录制失败"), QString::fromStdString(err));
    }
  }
}

void MainWindow::ToggleRtsp() {
  if (runtime_manager_->state() == AppState::kStreaming) {
    runtime_manager_->StopRtsp();
  } else if (runtime_manager_->state() == AppState::kPreviewing ||
             runtime_manager_->state() == AppState::kIdle) {
    std::string err;
    if (!runtime_manager_->StartRtsp(&err)) {
      QMessageBox::warning(this, QStringLiteral("RTSP 失败"), QString::fromStdString(err));
    }
  }
}

void MainWindow::OnStateChanged(rkstudio::AppState state) {
  struct StateRow {
    const char* label;
    QString preview_text;
    bool preview_enabled;
    QString record_text;
    bool record_enabled;
    QString rtsp_text;
    bool rtsp_enabled;
    bool mediapipe_enabled;
    bool yolo_enabled;
  };

  static const auto kTable = [] {
    std::map<rkstudio::AppState, StateRow> t;
    t[rkstudio::AppState::kIdle] = {
        "Idle",
        QStringLiteral("启动预览"), true,
        QStringLiteral("启动录制"), true,
        QStringLiteral("启动 RTSP"), true,
        false,
        false};
    t[rkstudio::AppState::kPreviewing] = {
        "Previewing",
        QStringLiteral("关闭预览"), true,
        QStringLiteral("启动录制"), true,
        QStringLiteral("启动 RTSP"), true,
        true,
        true};
    t[rkstudio::AppState::kRecording] = {
        "Recording",
        {}, false,
        QStringLiteral("停止录制"), true,
        {}, false,
        false,
        true};
    t[rkstudio::AppState::kStreaming] = {
        "Streaming",
        {}, false,
        {}, false,
        QStringLiteral("停止 RTSP"), true,
        false,
        false};
    t[rkstudio::AppState::kError] = {
        "Error",
        QStringLiteral("启动预览"), true,
        QStringLiteral("启动录制"), false,
        QStringLiteral("启动 RTSP"), false,
        false,
        false};
    return t;
  }();

  const auto it = kTable.find(state);
  if (it == kTable.end()) return;
  const auto& row = it->second;

  if (!row.preview_text.isEmpty()) preview_button_->setText(row.preview_text);
  preview_button_->setEnabled(row.preview_enabled);
  if (!row.record_text.isEmpty()) record_button_->setText(row.record_text);
  record_button_->setEnabled(row.record_enabled);
  if (!row.rtsp_text.isEmpty()) rtsp_button_->setText(row.rtsp_text);
  rtsp_button_->setEnabled(row.rtsp_enabled);
  mediapipe_toggle_button_->setText(runtime_manager_->mediapipe_enabled() ? QStringLiteral("关闭 Mediapipe")
                                                                       : QStringLiteral("启动 Mediapipe"));
  yolo_toggle_button_->setText(runtime_manager_->yolo_enabled() ? QStringLiteral("关闭 YOLO")
                                                             : QStringLiteral("启动 YOLO"));
  mediapipe_toggle_button_->setEnabled(row.mediapipe_enabled);
  yolo_toggle_button_->setEnabled(row.yolo_enabled);
  state_label_->setText(QString("状态: %1").arg(row.label));
}

void MainWindow::OnTelemetryObserved(rkstudio::TelemetryEvent event) {
  // Always show non-ok events; throttle ok events (4 cams × 30fps = 120/s).
  if (event.status == "ok") {
    if (++telemetry_ok_skip_counter_ % 30 != 0) {
      return;
    }
  }
  AppendLog(QString("[%1] %2/%3 %4 %5")
                .arg(QString::fromStdString(event.category))
                .arg(QString::fromStdString(event.stream_id))
                .arg(QString::fromStdString(event.stage))
                .arg(QString::fromStdString(event.status))
                .arg(QString::fromStdString(event.reason)));
}

void MainWindow::OnPreviewFailure(QString camera_id, QString reason, bool fatal) {
  if (const auto it = tiles_.find(camera_id); it != tiles_.end()) {
    it->second->SetStatusText(fatal ? QStringLiteral("故障: %1").arg(reason)
                                    : QStringLiteral("预览异常: %1").arg(reason));
  }
}

void MainWindow::OnTileRebound(QString camera_id, WId window_id) {
  runtime_manager_->BindPreviewWindow(camera_id.toStdString(), window_id);
}

void MainWindow::ToggleMediapipe() {
  const bool enabling = !runtime_manager_->mediapipe_enabled();
  mediapipe_toggle_button_->setText(enabling ? QStringLiteral("关闭 Mediapipe") : QStringLiteral("启动 Mediapipe"));

  if (!enabling && runtime_manager_->state() == rkstudio::AppState::kPreviewing) {
    SwapMediapipeTile(false);
  }

  std::string err;
  if (!runtime_manager_->ToggleMediapipe(enabling, &err)) {
    QMessageBox::warning(this, QStringLiteral("Mediapipe 切换失败"), QString::fromStdString(err));
    mediapipe_toggle_button_->setText(runtime_manager_->mediapipe_enabled() ? QStringLiteral("关闭 Mediapipe")
                                                           : QStringLiteral("启动 Mediapipe"));
    if (!enabling && runtime_manager_->state() == rkstudio::AppState::kPreviewing) {
      SwapMediapipeTile(true);
    }
    return;
  }

  if (enabling && runtime_manager_->state() == rkstudio::AppState::kPreviewing) {
    SwapMediapipeTile(true);
  }
}

void MainWindow::ToggleYolo() {
  const bool enabling = !runtime_manager_->yolo_enabled();
  yolo_toggle_button_->setText(enabling ? QStringLiteral("关闭 YOLO") : QStringLiteral("启动 YOLO"));

  std::string err;
  if (!runtime_manager_->ToggleYolo(enabling, &err)) {
    QMessageBox::warning(this, QStringLiteral("YOLO 切换失败"), QString::fromStdString(err));
    yolo_toggle_button_->setText(runtime_manager_->yolo_enabled() ? QStringLiteral("关闭 YOLO")
                                                               : QStringLiteral("启动 YOLO"));
  }
}

void MainWindow::SwapMediapipeTile(bool enabling) {
  const auto& profile = runtime_manager_->session_profile();
  const QString mediapipe_cam = QString::fromStdString(profile.selected_mediapipe_camera);
  if (mediapipe_cam.isEmpty()) return;

  QGridLayout* grid = qobject_cast<QGridLayout*>(grid_container_->layout());
  if (!grid) return;

  const int cols = std::max(1, profile.preview_cols);
  int target_row = -1;
  int target_col = -1;
  for (size_t i = 0; i < profile.preview_cameras.size(); ++i) {
    if (profile.preview_cameras[i] == profile.selected_mediapipe_camera) {
      target_row = static_cast<int>(i / cols);
      target_col = static_cast<int>(i % cols);
      break;
    }
  }
  if (target_row < 0) return;

  if (QLayoutItem* item = grid->itemAtPosition(target_row, target_col)) {
    if (QWidget* old_widget = item->widget()) {
      grid->removeWidget(old_widget);
      old_widget->setParent(nullptr);
      delete old_widget;
    }
  }

  tiles_.erase(mediapipe_cam);
  mediapipe_canvas_ = nullptr;

  if (enabling) {
    auto* canvas = new MediapipeCanvasWidget(grid_container_);
    canvas->setMinimumSize(320, 180);
    grid->addWidget(canvas, target_row, target_col);
    mediapipe_canvas_ = canvas;
  } else {
    auto* tile = new PreviewTileWidget(mediapipe_cam, grid_container_);
    connect(tile, &PreviewTileWidget::WindowRebound, this, &MainWindow::OnTileRebound);
    grid->addWidget(tile, target_row, target_col);
    tiles_.insert_or_assign(mediapipe_cam, tile);
    runtime_manager_->BindPreviewWindow(profile.selected_mediapipe_camera, tile->sink_window_id());
  }
}

void MainWindow::OnMediapipeFrame(QString camera_id, QImage image) {
  (void)camera_id;
  if (mediapipe_canvas_) {
    mediapipe_canvas_->SetFrame(image);
  }
}

void MainWindow::OnMediapipeResult(rkstudio::vision::MediapipeResult result) {
  if (mediapipe_canvas_) {
    mediapipe_canvas_->SetResult(result);
  }
}

void MainWindow::OnYoloResult(rkstudio::vision::YoloResult result) {
  if (!result.ok) {
    AppendLog(QString("[yolo] %1 error: %2")
                  .arg(QString::fromStdString(result.camera_id))
                  .arg(QString::fromStdString(result.error)));
    return;
  }
  if (result.detections.empty()) {
    if (++yolo_empty_skip_counter_ % 10 == 0) {
      AppendLog(QString("[yolo] %1 no objects, infer %.1f fps")
                    .arg(QString::fromStdString(result.camera_id))
                    .arg(result.fps));
    }
    return;
  }

  QStringList top;
  const int limit = std::min<int>(3, result.detections.size());
  for (int i = 0; i < limit; ++i) {
    const auto& det = result.detections[static_cast<size_t>(i)];
    top << QString("#%1 %.2f [%2,%3,%4,%5]")
               .arg(det.class_id)
               .arg(det.score)
               .arg(det.box.x1)
               .arg(det.box.y1)
               .arg(det.box.x2)
               .arg(det.box.y2);
  }
  AppendLog(QString("[yolo] %1 %2 objects, infer %.1f fps: %3")
                .arg(QString::fromStdString(result.camera_id))
                .arg(result.detections.size())
                .arg(result.fps)
                .arg(top.join(QStringLiteral("; "))));
}

}  // namespace rkstudio::ui
