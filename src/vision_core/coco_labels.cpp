#include "rk_studio/vision_core/coco_labels.h"

#include <array>
#include <cstddef>

namespace rkstudio::vision {

const char* CocoLabel(int class_id) {
  static constexpr std::array<const char*, 80> kLabels = {
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
      "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
      "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
      "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
      "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
      "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
      "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
      "hair drier", "toothbrush"};

  if (class_id < 0 || class_id >= static_cast<int>(kLabels.size())) {
    return nullptr;
  }
  return kLabels[static_cast<size_t>(class_id)];
}

}  // namespace rkstudio::vision
