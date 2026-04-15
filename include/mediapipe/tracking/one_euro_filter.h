#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace mediapipe_demo {

class OneEuroFilter {
 public:
  OneEuroFilter(float min_cutoff, float beta, float d_cutoff);

  void Reset();
  std::vector<cv::Point2f> Filter(const std::vector<cv::Point2f>& points, double now_s);

 private:
  float Alpha(double dt, float cutoff) const;

  float min_cutoff_;
  float beta_;
  float d_cutoff_;
  bool initialized_ = false;
  double last_time_s_ = 0.0;
  std::vector<cv::Point2f> x_prev_;
  std::vector<cv::Point2f> dx_prev_;
};

}  // namespace mediapipe_demo
