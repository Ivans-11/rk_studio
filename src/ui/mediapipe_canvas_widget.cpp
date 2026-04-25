#include "rk_studio/ui/mediapipe_canvas_widget.h"

#include <utility>

#include <QEvent>
#include <QPainter>
#include <QVBoxLayout>

namespace rkstudio::ui {

MediapipeCanvasWidget::MediapipeCanvasWidget(QWidget* parent)
    : MediapipeCanvasWidget(QStringLiteral("Mediapipe"), parent) {}

MediapipeCanvasWidget::MediapipeCanvasWidget(QString camera_id, QWidget* parent)
    : QWidget(parent) {
  setStyleSheet("MediapipeCanvasWidget { background: #1c1c1c; border: 2px solid #3b3b3b; border-radius: 10px; }");

  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(8, 8, 8, 8);
  layout->setSpacing(6);

  title_ = new QLabel(std::move(camera_id), this);
  title_->setStyleSheet("font-weight: 600; color: #f3f3f3;");
  status_ = new QLabel(QStringLiteral("Mediapipe"), this);
  status_->setStyleSheet("color: #bdbdbd;");

  canvas_ = new QFrame(this);
  canvas_->setFrameShape(QFrame::StyledPanel);
  canvas_->setStyleSheet("background: #000;");
  canvas_->setMinimumSize(320, 180);
  canvas_->installEventFilter(this);

  layout->addWidget(title_);
  layout->addWidget(canvas_, 1);
  layout->addWidget(status_);
}

void MediapipeCanvasWidget::SetFrame(const QImage& image) {
  frame_ = image;
  canvas_->update();
}

void MediapipeCanvasWidget::SetResult(const vision::MediapipeResult& result) {
  result_ = result;
  canvas_->update();
}

void MediapipeCanvasWidget::Clear() {
  frame_ = {};
  result_.reset();
  canvas_->update();
}

bool MediapipeCanvasWidget::eventFilter(QObject* watched, QEvent* event) {
  if (watched == canvas_ && event->type() == QEvent::Paint) {
    QPainter painter(canvas_);
    DrawCanvas(painter, canvas_->rect());
    return true;
  }
  return QWidget::eventFilter(watched, event);
}

void MediapipeCanvasWidget::DrawCanvas(QPainter& painter, const QRect& rect) {
  painter.fillRect(rect, Qt::black);
  if (frame_.isNull()) {
    painter.setPen(Qt::white);
    painter.drawText(rect, Qt::AlignCenter, QStringLiteral("Mediapipe 预览未就绪"));
    return;
  }

  const QRect target = rect;
  painter.drawImage(target, frame_);

  if (!result_.has_value()) {
    return;
  }

  const float scale_x = static_cast<float>(target.width()) / static_cast<float>(frame_.width());
  const float scale_y = static_cast<float>(target.height()) / static_cast<float>(frame_.height());
  auto map_point = [&](float x, float y) {
    return QPointF(target.left() + x * scale_x, target.top() + y * scale_y);
  };

  static const QColor kHandColors[] = {QColor(255, 160, 0), QColor(0, 160, 255)};
  static const QColor kLandmarkColors[] = {QColor(0, 255, 100), QColor(255, 100, 200)};

  for (size_t h = 0; h < result_->hands.size(); ++h) {
    const auto& hand = result_->hands[h];
    const QColor& roi_color = kHandColors[h % 2];
    const QColor& lm_color = kLandmarkColors[h % 2];

    if (hand.roi.has_value()) {
      painter.setPen(QPen(roi_color, 2));
      painter.setBrush(Qt::NoBrush);
      const auto& roi = *hand.roi;
      const QRectF roi_rect(map_point(static_cast<float>(roi.x1), static_cast<float>(roi.y1)),
                            map_point(static_cast<float>(roi.x2), static_cast<float>(roi.y2)));
      painter.drawRect(roi_rect.normalized());
    }

    painter.setPen(Qt::NoPen);
    painter.setBrush(lm_color);
    for (const auto& point : hand.landmarks) {
      painter.drawEllipse(map_point(point.x, point.y), 3.0, 3.0);
    }
  }
}

}  // namespace rkstudio::ui
