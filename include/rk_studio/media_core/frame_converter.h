#pragma once

#include <memory>
#include <optional>
#include <string>

#include "rk_studio/vision_core/vision_types.h"

struct _GstSample;
typedef struct _GstSample GstSample;

namespace cv {
class Mat;
}  // namespace cv

namespace rkstudio::media {

struct ConvertedFrame {
  vision::FrameRef frame;
  std::shared_ptr<cv::Mat> rgb_holder;
  bool used_rga = false;
};

class FrameConverter {
 public:
  std::optional<ConvertedFrame> ConvertNv12SampleToRgbFrame(
      GstSample* sample,
      const std::string& camera_id) const;
};

}  // namespace rkstudio::media
