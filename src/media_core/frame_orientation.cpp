#include "rk_studio/media_core/frame_orientation.h"

#include <opencv2/imgproc.hpp>

namespace rkstudio::media {
namespace {

cv::Mat FlipMat(const cv::Mat& input, int flip_code) {
  if (input.empty()) {
    return {};
  }
  cv::Mat output;
  cv::flip(input, output, flip_code);
  return output;
}

cv::Mat FlipUvHorizontalPairs(const cv::Mat& uv_plane) {
  if (uv_plane.empty() || (uv_plane.cols % 2) != 0) {
    return {};
  }
  cv::Mat output(uv_plane.rows, uv_plane.cols, uv_plane.type());
  for (int y = 0; y < uv_plane.rows; ++y) {
    const auto* src = uv_plane.ptr<uint8_t>(y);
    auto* dst = output.ptr<uint8_t>(y);
    for (int x = 0; x < uv_plane.cols; x += 2) {
      const int dst_x = uv_plane.cols - x - 2;
      dst[dst_x] = src[x];
      dst[dst_x + 1] = src[x + 1];
    }
  }
  return output;
}

cv::Mat FlipUvVerticalRows(const cv::Mat& uv_plane) {
  return FlipMat(uv_plane, 0);
}

}  // namespace

bool IsOriented(const std::string& orientation) {
  return orientation == "rotate-180" ||
         orientation == "horizontal-flip" ||
         orientation == "vertical-flip";
}

cv::Mat ApplyMatOrientation(const cv::Mat& input, const std::string& orientation) {
  if (input.empty() || orientation == "normal" || orientation.empty()) {
    return input.empty() ? cv::Mat() : input.clone();
  }
  if (orientation == "rotate-180") {
    return FlipMat(input, -1);
  }
  if (orientation == "horizontal-flip") {
    return FlipMat(input, 1);
  }
  if (orientation == "vertical-flip") {
    return FlipMat(input, 0);
  }
  return input.clone();
}

cv::Mat ApplyNv12Orientation(const cv::Mat& input, const std::string& orientation) {
  if (input.empty() || orientation == "normal" || orientation.empty()) {
    return input.empty() ? cv::Mat() : input.clone();
  }

  const int width = input.cols;
  const int total_rows = input.rows;
  const int height = total_rows * 2 / 3;
  if (width <= 0 || height <= 0 || total_rows != height * 3 / 2) {
    return {};
  }

  cv::Mat y_plane = input.rowRange(0, height);
  cv::Mat uv_plane = input.rowRange(height, total_rows);

  cv::Mat y_out;
  cv::Mat uv_out;
  if (orientation == "rotate-180") {
    cv::flip(y_plane, y_out, -1);
    uv_out = FlipUvVerticalRows(FlipUvHorizontalPairs(uv_plane));
  } else if (orientation == "horizontal-flip") {
    cv::flip(y_plane, y_out, 1);
    uv_out = FlipUvHorizontalPairs(uv_plane);
  } else if (orientation == "vertical-flip") {
    cv::flip(y_plane, y_out, 0);
    uv_out = FlipUvVerticalRows(uv_plane);
  } else {
    return input.clone();
  }
  if (y_out.empty() || uv_out.empty()) {
    return {};
  }

  cv::Mat output(total_rows, width, CV_8UC1);
  y_out.copyTo(output.rowRange(0, height));
  uv_out.copyTo(output.rowRange(height, total_rows));
  return output;
}

}  // namespace rkstudio::media
