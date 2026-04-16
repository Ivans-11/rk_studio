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
  media_engine_ = new media::MediaEngine(this);

  BuildUi();
  board_config_path_ = ResolveDefaultConfigPath(QStringLiteral("board.toml"));
  profile_path_ = ResolveDefaultConfigPath(QStringLiteral("profile.toml"));
  connect(media_engine_, &media::MediaEngine::StateChanged, this, &MainWindow::OnStateChanged);
  connect(media_engine_, &media::MediaEngine::TelemetryObserved, this, &MainWindow::OnTelemetryObserved);
  connect(media_engine_, &media::MediaEngine::PreviewCameraFailed, this, &MainWindow::OnPreviewFailure);
  connect(media_engine_, &media::MediaEngine::AiFrameReady, this, &MainWindow::OnAiFrame);
  connect(media_engine_, &media::MediaEngine::AiResultReady, this, &MainWindow::OnAiResult);

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
  ai_toggle_button_ = new QPushButton(QStringLiteral("启动 Mediapipe"), side_panel);
  record_button_->setEnabled(false);
  rtsp_button_->setEnabled(false);
  ai_toggle_button_->setEnabled(false);

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
  side_layout->addWidget(ai_toggle_button_);
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
  connect(ai_toggle_button_, &QPushButton::clicked, this, &MainWindow::ToggleAi);
}

void MainWindow::RebuildTiles() {
  ClearLayoutWidgets(grid_container_);
  tiles_.clear();
  ai_canvas_ = nullptr;

  const auto& profile = media_engine_->session_profile();
  auto* grid = new QGridLayout(grid_container_);
  grid->setContentsMargins(0, 0, 0, 0);
  grid->setSpacing(12);

  const int cols = std::max(1, profile.preview_cols);
  for (size_t i = 0; i < profile.preview_cameras.size(); ++i) {
    const QString camera_id = QString::fromStdString(profile.preview_cameras[i]);
    const int row = static_cast<int>(i / cols);
    const int col = static_cast<int>(i % cols);

    if (media_engine_->ai_enabled() &&
        profile.preview_cameras[i] == profile.selected_ai_camera) {
      // AI camera: use AiCanvasWidget instead of hardware preview sink
      auto* canvas = new AiCanvasWidget(grid_container_);
      canvas->setMinimumSize(320, 180);
      grid->addWidget(canvas, row, col);
      ai_canvas_ = canvas;
    } else {
      // Normal camera: hardware preview sink
      auto* tile = new PreviewTileWidget(camera_id, grid_container_);
      connect(tile, &PreviewTileWidget::WindowRebound, this, &MainWindow::OnTileRebound);
      grid->addWidget(tile, row, col);
      tiles_.insert_or_assign(camera_id, tile);
      media_engine_->BindPreviewWindow(profile.preview_cameras[i], tile->sink_window_id());
    }
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

  if (media_engine_->state() != AppState::kIdle) {
    media_engine_->StopAll();
  }
  media_engine_->LoadBoardConfig(board_config);
  media_engine_->ApplySessionProfile(profile);
  RebuildTiles();
  OnStateChanged(media_engine_->state());
  SetStatus(QString("已加载 %1 路摄像头").arg(profile.preview_cameras.size()));
}

void MainWindow::TogglePreview() {
  if (media_engine_->state() == AppState::kPreviewing) {
    media_engine_->StopAll();
  } else if (media_engine_->state() == AppState::kIdle) {
    std::string err;
    if (!media_engine_->StartPreview(&err)) {
      QMessageBox::warning(this, QStringLiteral("启动失败"), QString::fromStdString(err));
    }
  }
}

void MainWindow::ToggleRecording() {
  if (media_engine_->state() == AppState::kRecording) {
    media_engine_->StopRecording();
  } else if (media_engine_->state() == AppState::kPreviewing ||
             media_engine_->state() == AppState::kIdle) {
    std::string err;
    if (!media_engine_->StartRecording(&err)) {
      QMessageBox::warning(this, QStringLiteral("录制失败"), QString::fromStdString(err));
    }
  }
}

void MainWindow::ToggleRtsp() {
  if (media_engine_->state() == AppState::kStreaming) {
    media_engine_->StopRtsp();
  } else if (media_engine_->state() == AppState::kPreviewing ||
             media_engine_->state() == AppState::kIdle) {
    std::string err;
    if (!media_engine_->StartRtsp(&err)) {
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
    bool ai_enabled;
  };

  static const auto kTable = [] {
    std::map<rkstudio::AppState, StateRow> t;
    t[rkstudio::AppState::kIdle] = {
        "Idle",
        QStringLiteral("启动预览"), true,
        QStringLiteral("启动录制"), true,
        QStringLiteral("启动 RTSP"), true,
        false};
    t[rkstudio::AppState::kPreviewing] = {
        "Previewing",
        QStringLiteral("关闭预览"), true,
        QStringLiteral("启动录制"), true,
        QStringLiteral("启动 RTSP"), true,
        true};
    t[rkstudio::AppState::kRecording] = {
        "Recording",
        {}, false,
        QStringLiteral("停止录制"), true,
        {}, false,
        false};
    t[rkstudio::AppState::kStreaming] = {
        "Streaming",
        {}, false,
        {}, false,
        QStringLiteral("停止 RTSP"), true,
        false};
    t[rkstudio::AppState::kError] = {
        "Error",
        QStringLiteral("启动预览"), true,
        QStringLiteral("启动录制"), false,
        QStringLiteral("启动 RTSP"), false,
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
  ai_toggle_button_->setEnabled(row.ai_enabled);
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
  media_engine_->BindPreviewWindow(camera_id.toStdString(), window_id);
}

void MainWindow::ToggleAi() {
  const bool enabling = !media_engine_->ai_enabled();
  ai_toggle_button_->setText(enabling ? QStringLiteral("关闭 Mediapipe") : QStringLiteral("启动 Mediapipe"));

  const bool was_previewing = media_engine_->state() == rkstudio::AppState::kPreviewing;
  if (was_previewing) {
    media_engine_->StopAll();
  }
  media_engine_->SetAiEnabled(enabling);
  RebuildTiles();
  if (was_previewing) {
    QCoreApplication::processEvents();
    std::string err;
    if (!media_engine_->StartPreview(&err)) {
      QMessageBox::warning(this, QStringLiteral("AI 切换失败"), QString::fromStdString(err));
    }
  }
}

void MainWindow::OnAiFrame(QString camera_id, QImage image) {
  (void)camera_id;
  if (ai_canvas_) {
    ai_canvas_->SetFrame(image);
  }
}

void MainWindow::OnAiResult(rkstudio::ai::AiResult result) {
  if (ai_canvas_) {
    ai_canvas_->SetResult(result);
  }
}

}  // namespace rkstudio::ui
