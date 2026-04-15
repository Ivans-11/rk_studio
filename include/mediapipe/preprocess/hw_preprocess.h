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

}  // namespace mediapipe_demo
