#pragma once

#include <optional>

#include <QImage>
#include <QPaintEvent>
#include <QWidget>

#include "rk_studio/ai_core/ai_types.h"

namespace rkstudio::ui {

class AiCanvasWidget : public QWidget {
  Q_OBJECT

 public:
  explicit AiCanvasWidget(QWidget* parent = nullptr);

  void SetFrame(const QImage& image);
  void SetResult(const ai::AiResult& result);
  void Clear();

 protected:
  void paintEvent(QPaintEvent* event) override;

 private:
  QImage frame_;
  std::optional<ai::AiResult> result_;
};

}  // namespace rkstudio::ui
