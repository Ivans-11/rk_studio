#include "rk_studio/infra/gst_util.h"

#include <unordered_map>

namespace rkinfra {

std::string Uppercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
  });
  return value;
}

std::string NameWithIndex(const std::string& prefix, size_t index) { return prefix + std::to_string(index); }

bool IsJpegLikeFormat(std::string_view value) {
  const std::string upper = Uppercase(std::string(value));
  return upper == "MJPG" || upper == "JPEG";
}

bool IsNv12Format(std::string_view value) {
  const std::string upper = Uppercase(std::string(value));
  return upper == "NV12";
}

int ToV4l2IoMode(const std::string& value, std::string* err) {
  static const std::unordered_map<std::string, int> kModes{{"AUTO", 0},
                                                           {"RW", 1},
                                                           {"MMAP", 2},
                                                           {"USERPTR", 3},
                                                           {"DMABUF", 4},
                                                           {"DMABUF-IMPORT", 5}};
  const std::string upper = Uppercase(value);
  const auto it = kModes.find(upper);
  if (it != kModes.end()) {
    return it->second;
  }
  if (err) {
    *err = "invalid io_mode: " + value;
  }
  return -1;
}

}  // namespace rkinfra
