#pragma once

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include "mediapipe/common/types.h"

namespace mediapipe_demo {

cv::Size TensorSizeFromAttr(const rknn_tensor_attr& attr);

bool PreprocessFrameToRknn(const CameraFrame& frame,
                           const cv::Rect& src_rect,
                           bool keep_aspect,
                           rknn_tensor_mem* dst_mem,
                           const rknn_tensor_attr& dst_attr,
                           PreprocessMeta* meta);

// Full-frame NV12→RGB conversion via RGA hardware.
// On success, writes an RGB888 cv::Mat (h × w × 3) to *rgb_out and returns true.
// Falls back gracefully: returns false if RGA is unavailable or dmabuf_fd < 0.
bool ConvertNv12ToRgb(int dmabuf_fd, int width, int height, int stride,
                      cv::Mat* rgb_out);

}  // namespace mediapipe_demo
