#pragma once

#include <optional>

#include <QImage>
#include <QPaintEvent>
#include <QWidget>

#include "rk_studio/vision_core/vision_types.h"

namespace rkstudio::ui {

class MediapipeCanvasWidget : public QWidget {
  Q_OBJECT

 public:
  explicit MediapipeCanvasWidget(QWidget* parent = nullptr);

  void SetFrame(const QImage& image);
  void SetResult(const vision::MediapipeResult& result);
  void Clear();

 protected:
  void paintEvent(QPaintEvent* event) override;

 private:
  QImage frame_;
  std::optional<vision::MediapipeResult> result_;
};

}  // namespace rkstudio::ui
