#pragma once

#include <QEvent>
#include <QFrame>
#include <QLabel>
#include <QWidget>

#include "rk_studio/vision_core/vision_types.h"

namespace rkstudio::ui {

class PreviewTileWidget : public QWidget {
  Q_OBJECT

 public:
  explicit PreviewTileWidget(QString camera_id, QWidget* parent = nullptr);

  QString camera_id() const;
  WId sink_window_id();
  void SetStatusText(const QString& text);
  void SetMediapipeResult(const vision::MediapipeResult& result);
  void ClearMediapipeResult();
  void SetYoloResult(const vision::YoloResult& result);
  void ClearYoloResult();

 signals:
  void WindowRebound(QString camera_id, WId window_id);

 protected:
  bool eventFilter(QObject* watched, QEvent* event) override;

 private:
  void RebindSinkWindow();

  QString camera_id_;
  QLabel* title_ = nullptr;
  QLabel* status_ = nullptr;
  QFrame* sink_host_ = nullptr;
  QWidget* overlay_ = nullptr;
};

}  // namespace rkstudio::ui
