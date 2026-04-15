#include <cstdlib>

#include <gst/gst.h>

#include <QApplication>

#include "rk_studio/ui/main_window.h"

int main(int argc, char** argv) {
  setenv("QT_XCB_GL_INTEGRATION", "none", 0);

  gst_init(&argc, &argv);

  QApplication app(argc, argv);
  rkstudio::ui::MainWindow window;
  window.show();
  return app.exec();
}
