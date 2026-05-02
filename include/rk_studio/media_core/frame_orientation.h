#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace rkstudio::media {

bool IsOriented(const std::string& orientation);
cv::Mat ApplyMatOrientation(const cv::Mat& input, const std::string& orientation);
cv::Mat ApplyNv12Orientation(const cv::Mat& input, const std::string& orientation);

}  // namespace rkstudio::media
