#include "rk_studio/ui/preview_tile_widget.h"

#include <utility>

#include <QTimer>
#include <QVBoxLayout>

namespace rkstudio::ui {

PreviewTileWidget::PreviewTileWidget(QString camera_id, QWidget* parent)
    : QWidget(parent), camera_id_(std::move(camera_id)) {
  setStyleSheet("PreviewTileWidget { background: #1c1c1c; border: 2px solid #3b3b3b; border-radius: 10px; }");

  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(8, 8, 8, 8);
  layout->setSpacing(6);

  title_ = new QLabel(camera_id_, this);
  title_->setStyleSheet("font-weight: 600; color: #f3f3f3;");
  status_ = new QLabel(QStringLiteral("未启动"), this);
  status_->setStyleSheet("color: #bdbdbd;");

  sink_host_ = new QFrame(this);
  sink_host_->setFrameShape(QFrame::StyledPanel);
  sink_host_->setStyleSheet("background: #000;");
  sink_host_->setAttribute(Qt::WA_NativeWindow);
  sink_host_->setAttribute(Qt::WA_DontCreateNativeAncestors);
  sink_host_->setMinimumSize(320, 180);
  sink_host_->installEventFilter(this);

  layout->addWidget(title_);
  layout->addWidget(sink_host_, 1);
  layout->addWidget(status_);
}

QString PreviewTileWidget::camera_id() const {
  return camera_id_;
}

WId PreviewTileWidget::sink_window_id() {
  return sink_host_->winId();
}

void PreviewTileWidget::RebindSinkWindow() {
  const WId window_id = sink_host_->winId();
  QTimer::singleShot(0, this, [this, window_id] {
    if (sink_host_ != nullptr && sink_host_->winId() == window_id) {
      emit WindowRebound(camera_id_, window_id);
    }
  });
}

void PreviewTileWidget::SetStatusText(const QString& text) {
  status_->setText(text);
}

bool PreviewTileWidget::eventFilter(QObject* watched, QEvent* event) {
  if (watched == sink_host_ &&
      (event->type() == QEvent::WinIdChange ||
       event->type() == QEvent::Resize ||
       event->type() == QEvent::Show)) {
    RebindSinkWindow();
    return false;
  }
  return QWidget::eventFilter(watched, event);
}

}  // namespace rkstudio::ui
