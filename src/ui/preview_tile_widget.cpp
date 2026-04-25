#include "rk_studio/ui/preview_tile_widget.h"

#include <algorithm>
#include <optional>
#include <utility>

#include <QFont>
#include <QPainter>
#include <QPaintEvent>
#include <QTimer>
#include <QVBoxLayout>

#include "rk_studio/ui/yolo_labels.h"

namespace rkstudio::ui {
namespace {

class ResultOverlayWidget final : public QWidget {
 public:
  explicit ResultOverlayWidget(QWidget* parent = nullptr) : QWidget(parent) {
    setAttribute(Qt::WA_TransparentForMouseEvents);
    setAttribute(Qt::WA_NoSystemBackground);
    setAttribute(Qt::WA_TranslucentBackground);
  }

  void SetMediapipeResult(const vision::MediapipeResult& result) {
    mediapipe_result_ = result;
    update();
  }

  void ClearMediapipeResult() {
    mediapipe_result_.reset();
    update();
  }

  void SetYoloResult(const vision::YoloResult& result) {
    yolo_result_ = result;
    update();
  }

  void ClearYoloResult() {
    yolo_result_.reset();
    update();
  }

 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    DrawMediapipe(painter, rect());
    DrawYolo(painter, rect());
  }

 private:
  static QPointF MapPoint(const QRect& target,
                          int source_w,
                          int source_h,
                          float x,
                          float y) {
    const float scale_x = static_cast<float>(target.width()) /
                          static_cast<float>(std::max(1, source_w));
    const float scale_y = static_cast<float>(target.height()) /
                          static_cast<float>(std::max(1, source_h));
    return QPointF(target.left() + x * scale_x, target.top() + y * scale_y);
  }

  void DrawMediapipe(QPainter& painter, const QRect& target) {
    if (!mediapipe_result_.has_value()) {
      return;
    }

    static const QColor kHandColors[] = {QColor(255, 160, 0), QColor(0, 160, 255)};
    static const QColor kLandmarkColors[] = {QColor(0, 255, 100), QColor(255, 100, 200)};

    const int source_w = mediapipe_result_->frame_width;
    const int source_h = mediapipe_result_->frame_height;
    for (size_t h = 0; h < mediapipe_result_->hands.size(); ++h) {
      const auto& hand = mediapipe_result_->hands[h];
      const QColor& roi_color = kHandColors[h % 2];
      const QColor& lm_color = kLandmarkColors[h % 2];

      if (hand.roi.has_value()) {
        painter.setPen(QPen(roi_color, 2));
        painter.setBrush(Qt::NoBrush);
        const auto& roi = *hand.roi;
        const QRectF roi_rect(
            MapPoint(target, source_w, source_h, static_cast<float>(roi.x1), static_cast<float>(roi.y1)),
            MapPoint(target, source_w, source_h, static_cast<float>(roi.x2), static_cast<float>(roi.y2)));
        painter.drawRect(roi_rect.normalized());
      }

      painter.setPen(Qt::NoPen);
      painter.setBrush(lm_color);
      for (const auto& point : hand.landmarks) {
        painter.drawEllipse(MapPoint(target, source_w, source_h, point.x, point.y), 3.0, 3.0);
      }
    }
  }

  void DrawYolo(QPainter& painter, const QRect& target) {
    if (!yolo_result_.has_value()) {
      return;
    }

    static const QColor kBoxColors[] = {
        QColor(0, 220, 140),
        QColor(255, 180, 0),
        QColor(0, 180, 255),
        QColor(255, 90, 160),
    };

    painter.setFont(QFont(painter.font().family(), 10, QFont::DemiBold));
    const int source_w = yolo_result_->frame_width;
    const int source_h = yolo_result_->frame_height;
    for (size_t i = 0; i < yolo_result_->detections.size(); ++i) {
      const auto& det = yolo_result_->detections[i];
      const QColor color = kBoxColors[i % 4];
      const QRectF box(
          MapPoint(target, source_w, source_h, static_cast<float>(det.box.x1), static_cast<float>(det.box.y1)),
          MapPoint(target, source_w, source_h, static_cast<float>(det.box.x2), static_cast<float>(det.box.y2)));
      const QRectF normalized = box.normalized();

      painter.setPen(QPen(color, 2));
      painter.setBrush(Qt::NoBrush);
      painter.drawRect(normalized);

      const char* class_name = CocoLabel(det.class_id);
      const QString label = class_name != nullptr
                                ? QString("%1  %2").arg(QString::fromLatin1(class_name)).arg(det.score, 0, 'f', 2)
                                : QString("#%1  %2").arg(det.class_id).arg(det.score, 0, 'f', 2);
      const QRect label_rect = painter.fontMetrics().boundingRect(label).adjusted(-4, -2, 4, 2);
      QRectF label_box(normalized.left(), normalized.top() - label_rect.height(),
                       label_rect.width(), label_rect.height());
      if (label_box.top() < target.top()) {
        label_box.moveTop(normalized.top());
      }
      painter.fillRect(label_box, color);
      painter.setPen(Qt::black);
      painter.drawText(label_box.adjusted(4, 0, -4, 0), Qt::AlignVCenter | Qt::AlignLeft, label);
    }
  }

  std::optional<vision::MediapipeResult> mediapipe_result_;
  std::optional<vision::YoloResult> yolo_result_;
};

}  // namespace

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
  overlay_ = new ResultOverlayWidget(sink_host_);
  overlay_->raise();

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

void PreviewTileWidget::SetMediapipeResult(const vision::MediapipeResult& result) {
  if (auto* overlay = static_cast<ResultOverlayWidget*>(overlay_)) {
    overlay->SetMediapipeResult(result);
  }
}

void PreviewTileWidget::ClearMediapipeResult() {
  if (auto* overlay = static_cast<ResultOverlayWidget*>(overlay_)) {
    overlay->ClearMediapipeResult();
  }
}

void PreviewTileWidget::SetYoloResult(const vision::YoloResult& result) {
  if (auto* overlay = static_cast<ResultOverlayWidget*>(overlay_)) {
    overlay->SetYoloResult(result);
  }
}

void PreviewTileWidget::ClearYoloResult() {
  if (auto* overlay = static_cast<ResultOverlayWidget*>(overlay_)) {
    overlay->ClearYoloResult();
  }
}

bool PreviewTileWidget::eventFilter(QObject* watched, QEvent* event) {
  if (watched == sink_host_ &&
      (event->type() == QEvent::WinIdChange ||
       event->type() == QEvent::Resize ||
       event->type() == QEvent::Show)) {
    if (overlay_ != nullptr) {
      overlay_->setGeometry(sink_host_->rect());
      overlay_->raise();
    }
    RebindSinkWindow();
    return false;
  }
  return QWidget::eventFilter(watched, event);
}

}  // namespace rkstudio::ui
