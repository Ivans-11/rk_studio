#include "rk_studio/ui/ai_canvas_widget.h"

#include <QPainter>

namespace rkstudio::ui {

AiCanvasWidget::AiCanvasWidget(QWidget* parent) : QWidget(parent) {
  setMinimumSize(320, 180);
  setAutoFillBackground(true);
}

void AiCanvasWidget::SetFrame(const QImage& image) {
  frame_ = image;
  update();
}

void AiCanvasWidget::SetResult(const ai::AiResult& result) {
  result_ = result;
  update();
}

void AiCanvasWidget::Clear() {
  frame_ = {};
  result_.reset();
  update();
}

void AiCanvasWidget::paintEvent(QPaintEvent* event) {
  QWidget::paintEvent(event);

  QPainter painter(this);
  painter.fillRect(rect(), QColor(20, 20, 20));
  if (frame_.isNull()) {
    painter.setPen(Qt::white);
    painter.drawText(rect(), Qt::AlignCenter, QStringLiteral("AI 预览未就绪"));
    return;
  }

  const QSize scaled = frame_.size().scaled(size(), Qt::KeepAspectRatio);
  const QRect target((width() - scaled.width()) / 2, (height() - scaled.height()) / 2,
                     scaled.width(), scaled.height());
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

  painter.setPen(Qt::white);
  painter.drawText(rect().adjusted(8, 8, -8, -8), Qt::AlignTop | Qt::AlignLeft,
                   QString("FPS:%1  Hands:%2")
                       .arg(result_->fps, 0, 'f', 1)
                       .arg(result_->hands.size()));
}

}  // namespace rkstudio::ui
