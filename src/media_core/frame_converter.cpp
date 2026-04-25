#include "rk_studio/media_core/frame_converter.h"

#include <memory>
#include <utility>

#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/imgproc.hpp>

#include "mediapipe/preprocess/hw_preprocess.h"

namespace rkstudio::media {
namespace {

int ExtractDmabufFd(GstBuffer* buffer) {
  GstMemory* mem = gst_buffer_peek_memory(buffer, 0);
  if (mem != nullptr && gst_is_dmabuf_memory(mem)) {
    return gst_dmabuf_memory_get_fd(mem);
  }
  return -1;
}

}  // namespace

std::optional<ConvertedFrame> FrameConverter::ConvertNv12SampleToRgbFrame(
    GstSample* sample,
    const std::string& camera_id) const {
  if (sample == nullptr || camera_id.empty()) {
    return std::nullopt;
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);
  if (!buffer || !caps) {
    return std::nullopt;
  }

  GstVideoInfo info;
  if (!gst_video_info_from_caps(&info, caps)) {
    return std::nullopt;
  }

  const int w = GST_VIDEO_INFO_WIDTH(&info);
  const int h = GST_VIDEO_INFO_HEIGHT(&info);
  const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
  const uint64_t pts_ns = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))
                              ? GST_BUFFER_PTS(buffer) : 0;

  cv::Mat rgb;
  bool rga_ok = false;
  const int fd = ExtractDmabufFd(buffer);
  if (fd >= 0) {
    rga_ok = mediapipe_demo::ConvertNv12ToRgb(fd, w, h, stride, &rgb);
  }

  if (!rga_ok) {
    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      return std::nullopt;
    }
    if (w > 0 && h > 0 && map.size >= static_cast<size_t>(stride) * h * 3 / 2) {
      cv::Mat nv12(h * 3 / 2, w, CV_8UC1, map.data, stride);
      cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    }
    gst_buffer_unmap(buffer, &map);
  }
  if (rgb.empty()) {
    return std::nullopt;
  }

  auto rgb_holder = std::make_shared<cv::Mat>(std::move(rgb));
  ConvertedFrame output;
  output.rgb_holder = rgb_holder;
  output.used_rga = rga_ok;
  output.frame.camera_id = camera_id;
  output.frame.pts_ns = pts_ns;
  output.frame.width = w;
  output.frame.height = h;
  output.frame.stride = static_cast<int>(rgb_holder->step[0]);
  output.frame.pixel_format = vision::PixelFormat::kRgb;
  output.frame.mapped_ptr = rgb_holder->data;
  output.frame.bytes_used = rgb_holder->total() * rgb_holder->elemSize();
  output.frame.owned_data = rgb_holder;
  return output;
}

}  // namespace rkstudio::media
