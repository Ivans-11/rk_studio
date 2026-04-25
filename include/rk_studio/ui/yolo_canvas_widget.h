#pragma once

#include <optional>

#include <QFrame>
#include <QImage>
#include <QLabel>
#include <QPaintEvent>
#include <QWidget>

#include "rk_studio/vision_core/vision_types.h"

namespace rkstudio::ui {

class YoloCanvasWidget : public QWidget {
  Q_OBJECT

 public:
  explicit YoloCanvasWidget(QWidget* parent = nullptr);
  explicit YoloCanvasWidget(QString camera_id, QWidget* parent = nullptr);

  void SetFrame(const QImage& image);
  void SetResult(const vision::YoloResult& result);
  void Clear();

 protected:
  bool eventFilter(QObject* watched, QEvent* event) override;

 private:
  void DrawCanvas(QPainter& painter, const QRect& rect);

  QImage frame_;
  std::optional<vision::YoloResult> result_;
  QLabel* title_ = nullptr;
  QLabel* status_ = nullptr;
  QFrame* canvas_ = nullptr;
};

}  // namespace rkstudio::ui
