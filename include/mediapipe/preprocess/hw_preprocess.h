#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include "mediapipe/common/types.h"

namespace mediapipe_demo {

struct Nv12RgaInput {
  int dmabuf_fd = -1;
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;
};

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

// Full-frame RGB888→NV12 conversion via RGA hardware.
// On success, writes a tightly packed NV12 cv::Mat ((h * 3 / 2) × w) to
// *nv12_out and returns true.
bool ConvertRgbToNv12(const cv::Mat& rgb, cv::Mat* nv12_out);

// 2D NV12 mosaic via RGA. Inputs can be dmabuf-backed or tightly/strided
// CPU NV12 data. On success, writes a tightly packed NV12 frame to *nv12_out.
bool MosaicNv12ToNv12(const std::vector<Nv12RgaInput>& inputs,
                      int cols,
                      int rows,
                      int tile_width,
                      int tile_height,
                      int output_width,
                      int output_height,
                      cv::Mat* nv12_out);

}  // namespace mediapipe_demo
