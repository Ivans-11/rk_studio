#include "rk_studio/ui/yolo_canvas_widget.h"

#include <utility>

#include <QEvent>
#include <QPainter>
#include <QVBoxLayout>

#include "rk_studio/ui/yolo_labels.h"

namespace rkstudio::ui {

YoloCanvasWidget::YoloCanvasWidget(QWidget* parent)
    : YoloCanvasWidget(QStringLiteral("YOLO"), parent) {}

YoloCanvasWidget::YoloCanvasWidget(QString camera_id, QWidget* parent) : QWidget(parent) {
  setStyleSheet("YoloCanvasWidget { background: #1c1c1c; border: 2px solid #3b3b3b; border-radius: 10px; }");

  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(8, 8, 8, 8);
  layout->setSpacing(6);

  title_ = new QLabel(std::move(camera_id), this);
  title_->setStyleSheet("font-weight: 600; color: #f3f3f3;");
  status_ = new QLabel(QStringLiteral("YOLO"), this);
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

void YoloCanvasWidget::SetFrame(const QImage& image) {
  frame_ = image;
  canvas_->update();
}

void YoloCanvasWidget::SetResult(const vision::YoloResult& result) {
  result_ = result;
  canvas_->update();
}

void YoloCanvasWidget::Clear() {
  frame_ = {};
  result_.reset();
  canvas_->update();
}

bool YoloCanvasWidget::eventFilter(QObject* watched, QEvent* event) {
  if (watched == canvas_ && event->type() == QEvent::Paint) {
    QPainter painter(canvas_);
    DrawCanvas(painter, canvas_->rect());
    return true;
  }
  return QWidget::eventFilter(watched, event);
}

void YoloCanvasWidget::DrawCanvas(QPainter& painter, const QRect& rect) {
  painter.fillRect(rect, Qt::black);
  if (frame_.isNull()) {
    painter.setPen(Qt::white);
    painter.drawText(rect, Qt::AlignCenter, QStringLiteral("YOLO 预览未就绪"));
    return;
  }

  const QRect target = rect;
  painter.drawImage(target, frame_);

  if (!result_.has_value()) {
    return;
  }

  const float source_w = result_->frame_width > 0 ? static_cast<float>(result_->frame_width)
                                                  : static_cast<float>(frame_.width());
  const float source_h = result_->frame_height > 0 ? static_cast<float>(result_->frame_height)
                                                   : static_cast<float>(frame_.height());
  const float scale_x = static_cast<float>(target.width()) / source_w;
  const float scale_y = static_cast<float>(target.height()) / source_h;
  auto map_point = [&](float x, float y) {
    return QPointF(target.left() + x * scale_x, target.top() + y * scale_y);
  };

  static const QColor kBoxColors[] = {
      QColor(0, 220, 140),
      QColor(255, 180, 0),
      QColor(0, 180, 255),
      QColor(255, 90, 160),
  };

  painter.setFont(QFont(painter.font().family(), 10, QFont::DemiBold));
  for (size_t i = 0; i < result_->detections.size(); ++i) {
    const auto& det = result_->detections[i];
    const QColor color = kBoxColors[i % 4];
    const QRectF box(map_point(static_cast<float>(det.box.x1), static_cast<float>(det.box.y1)),
                     map_point(static_cast<float>(det.box.x2), static_cast<float>(det.box.y2)));
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

}  // namespace rkstudio::ui
