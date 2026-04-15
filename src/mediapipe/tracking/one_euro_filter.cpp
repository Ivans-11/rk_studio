#include "mediapipe/tracking/one_euro_filter.h"

#include <algorithm>
#include <cmath>

namespace mediapipe_demo {

OneEuroFilter::OneEuroFilter(float min_cutoff, float beta, float d_cutoff)
    : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff) {}

void OneEuroFilter::Reset() {
  initialized_ = false;
  last_time_s_ = 0.0;
  x_prev_.clear();
  dx_prev_.clear();
}

std::vector<cv::Point2f> OneEuroFilter::Filter(const std::vector<cv::Point2f>& points,
                                               double now_s) {
  if (!initialized_) {
    initialized_ = true;
    last_time_s_ = now_s;
    x_prev_ = points;
    dx_prev_.assign(points.size(), cv::Point2f(0.0f, 0.0f));
    return points;
  }

  const double dt = std::max(now_s - last_time_s_, 1e-6);
  std::vector<cv::Point2f> filtered(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const cv::Point2f dx((points[i].x - x_prev_[i].x) / static_cast<float>(dt),
                         (points[i].y - x_prev_[i].y) / static_cast<float>(dt));
    const float a_d = Alpha(dt, d_cutoff_);
    const cv::Point2f dx_hat(a_d * dx.x + (1.0f - a_d) * dx_prev_[i].x,
                             a_d * dx.y + (1.0f - a_d) * dx_prev_[i].y);

    const float cutoff_x = min_cutoff_ + beta_ * std::abs(dx_hat.x);
    const float cutoff_y = min_cutoff_ + beta_ * std::abs(dx_hat.y);
    const float a_x = Alpha(dt, cutoff_x);
    const float a_y = Alpha(dt, cutoff_y);

    filtered[i].x = a_x * points[i].x + (1.0f - a_x) * x_prev_[i].x;
    filtered[i].y = a_y * points[i].y + (1.0f - a_y) * x_prev_[i].y;
    dx_prev_[i] = dx_hat;
  }

  x_prev_ = filtered;
  last_time_s_ = now_s;
  return filtered;
}

float OneEuroFilter::Alpha(double dt, float cutoff) const {
  const float tau = 1.0f / (2.0f * static_cast<float>(M_PI) * cutoff);
  return 1.0f / (1.0f + tau / static_cast<float>(dt));
}

}  // namespace mediapipe_demo
