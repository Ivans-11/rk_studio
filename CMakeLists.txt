cmake_minimum_required(VERSION 3.18)

project(rk_studio LANGUAGES CXX)

# Project-wide build defaults.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Core dependencies.
find_package(Threads REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(zenohc REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-allocators-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
  gstreamer-audio-1.0
  gstreamer-rtsp-server-1.0
)

set(RK_STUDIO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Infrastructure sources (self-contained, no external project dependency).
set(RK_STUDIO_INFRA_SOURCES
  src/infra/runtime.cpp
  src/infra/gst_util.cpp
  src/infra/gst_audio_recorder.cpp
  src/infra/zenoh_publisher.cpp
  src/infra/telemetry.cpp
  src/infra/session_files.cpp
)
# Application sources.
set(RK_STUDIO_SOURCES
  src/main.cpp
  src/domain/types.cpp
  src/domain/config.cpp
  src/domain/session.cpp
  src/runtime/runtime_manager.cpp
  src/media_core/frame_orientation.cpp
  src/media_core/frame_converter.cpp
  src/media_core/v4l2_pipeline.cpp
  src/media_core/media_engine.cpp
  src/media_core/vision_engine.cpp
  src/media_core/session_writer.cpp
  src/media_core/rtsp_server.cpp
  src/ui/preview_tile_widget.cpp
  src/ui/main_window.cpp
  src/vision_core/coco_labels.cpp
  src/vision_core/mediapipe_processor.cpp
  src/vision_core/yolo_processor.cpp
  src/vision_core/face_expression_processor.cpp
)

set(RK_STUDIO_MEDIAPIPE_SOURCES
  src/mediapipe/common/rknn_model.cpp
  src/mediapipe/detector/palm_decoder.cpp
  src/mediapipe/detector/palm_detector.cpp
  src/mediapipe/landmark/hand_landmark.cpp
  src/mediapipe/preprocess/image_ops.cpp
  src/mediapipe/preprocess/hw_preprocess.cpp
  src/mediapipe/tracking/hand_tracker.cpp
  src/mediapipe/tracking/one_euro_filter.cpp
)

set(RK_STUDIO_HEADERS
  include/rk_studio/domain/types.h
  include/rk_studio/domain/config.h
  include/rk_studio/domain/session.h
  include/rk_studio/runtime/runtime_manager.h
  include/rk_studio/vision_core/coco_labels.h
  include/rk_studio/vision_core/vision_types.h
  include/rk_studio/vision_core/vision_processor.h
  include/rk_studio/media_core/frame_converter.h
  include/rk_studio/media_core/frame_orientation.h
  include/rk_studio/media_core/v4l2_pipeline.h
  include/rk_studio/media_core/media_engine.h
  include/rk_studio/media_core/vision_engine.h
  include/rk_studio/media_core/session_writer.h
  include/rk_studio/ui/preview_tile_widget.h
  include/rk_studio/ui/main_window.h
  include/rk_studio/infra/runtime.h
  include/rk_studio/infra/gst_util.h
  include/rk_studio/infra/config_types.h
  include/rk_studio/infra/telemetry.h
  include/rk_studio/infra/gst_audio_recorder.h
  include/rk_studio/infra/zenoh_publisher.h
  include/rk_studio/infra/session_files.h
)

add_executable(rk_studio)
target_sources(rk_studio PRIVATE
  ${RK_STUDIO_SOURCES}
  ${RK_STUDIO_MEDIAPIPE_SOURCES}
  ${RK_STUDIO_HEADERS}
  ${RK_STUDIO_INFRA_SOURCES}
)

target_include_directories(rk_studio
  PRIVATE
    ${RK_STUDIO_ROOT}/include
    ${RK_STUDIO_ROOT}/include/mediapipe
    ${RK_STUDIO_ROOT}/third_party/rknn
    ${RK_STUDIO_ROOT}/third_party/tomlplusplus
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rk_studio
  PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Threads::Threads
    PkgConfig::GST
    ${OpenCV_LIBS}
    zenohc::lib
    rknnrt
    rga
)

target_compile_options(rk_studio PRIVATE -Wall -Wextra -Wpedantic)
