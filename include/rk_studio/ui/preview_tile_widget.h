#pragma once

#include <QEvent>
#include <QFrame>
#include <QLabel>
#include <QMouseEvent>
#include <QWidget>

namespace rkstudio::ui {

class PreviewTileWidget : public QWidget {
  Q_OBJECT

 public:
  explicit PreviewTileWidget(QString camera_id, QWidget* parent = nullptr);

  QString camera_id() const;
  WId sink_window_id();
  void SetStatusText(const QString& text);

 signals:
  void WindowRebound(QString camera_id, WId window_id);

 protected:
  bool eventFilter(QObject* watched, QEvent* event) override;

 private:
  QString camera_id_;
  QLabel* title_ = nullptr;
  QLabel* status_ = nullptr;
  QFrame* sink_host_ = nullptr;
};

}  // namespace rkstudio::ui
