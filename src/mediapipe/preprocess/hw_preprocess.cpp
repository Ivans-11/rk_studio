#include "mediapipe/preprocess/hw_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>
#include <rga/im2d.hpp>
#include <rga/im2d_buffer.h>
#include <rga/rga.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace mediapipe_demo {

namespace {

bool g_hw_preprocess_available = true;
std::mutex g_hw_preprocess_mutex;

#if defined(__linux__)
class ScratchDmabuf {
 public:
  ScratchDmabuf() = default;
  ~ScratchDmabuf() {
    Reset();
  }

  ScratchDmabuf(const ScratchDmabuf&) = delete;
  ScratchDmabuf& operator=(const ScratchDmabuf&) = delete;

  bool Ensure(size_t size,
              int width,
              int height,
              int wstride,
              int hstride,
              int format) {
    if (fd_ >= 0 &&
        size_ >= size &&
        width_ == width &&
        height_ == height &&
        wstride_ == wstride &&
        hstride_ == hstride &&
        format_ == format) {
      return true;
    }

    Reset();
    int fd = -1;
    void* addr = nullptr;
    if (!AllocateDmabuf(size, &fd, &addr)) {
      return false;
    }
    fd_ = fd;
    addr_ = addr;
    size_ = size;
    width_ = width;
    height_ = height;
    wstride_ = wstride;
    hstride_ = hstride;
    format_ = format;
    return true;
  }

  bool Matches(size_t size,
               int width,
               int height,
               int wstride,
               int hstride,
               int format) const {
    return fd_ >= 0 &&
           size_ >= size &&
           width_ == width &&
           height_ == height &&
           wstride_ == wstride &&
           hstride_ == hstride &&
           format_ == format;
  }

  int fd() const { return fd_; }
  void* addr() const { return addr_; }
  size_t size() const { return size_; }
  int width() const { return width_; }
  int height() const { return height_; }
  int wstride() const { return wstride_; }
  int hstride() const { return hstride_; }
  int format() const { return format_; }

 private:
  static bool AllocateDmabuf(size_t size, int* fd, void** addr) {
    static const char* kHeaps[] = {
        "/dev/dma_heap/system-uncached-dma32",
        "/dev/dma_heap/system-dma32",
        "/dev/dma_heap/cma-uncached",
        "/dev/dma_heap/cma",
    };

    const size_t aligned_size = (size + 4095U) & ~4095U;
    for (const char* heap_path : kHeaps) {
      const int heap_fd = open(heap_path, O_RDWR | O_CLOEXEC);
      if (heap_fd < 0) {
        continue;
      }

      struct dma_heap_allocation_data alloc;
      std::memset(&alloc, 0, sizeof(alloc));
      alloc.len = aligned_size;
      alloc.fd_flags = O_RDWR | O_CLOEXEC;
      if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) == 0) {
        close(heap_fd);
        void* mapped =
            mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.fd, 0);
        if (mapped == MAP_FAILED) {
          close(alloc.fd);
          continue;
        }
        *fd = alloc.fd;
        *addr = mapped;
        return true;
      }
      close(heap_fd);
    }

    std::cerr << "failed to allocate dma_heap scratch buffer\n";
    return false;
  }

  void Reset() {
    if (addr_ != nullptr && size_ > 0) {
      munmap(addr_, size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = -1;
    addr_ = nullptr;
    size_ = 0;
    width_ = 0;
    height_ = 0;
    wstride_ = 0;
    hstride_ = 0;
    format_ = 0;
  }

  int fd_ = -1;
  void* addr_ = nullptr;
  size_t size_ = 0;
  int width_ = 0;
  int height_ = 0;
  int wstride_ = 0;
  int hstride_ = 0;
  int format_ = 0;
};

ScratchDmabuf* AcquireScratchBuffer(size_t size,
                                    int width,
                                    int height,
                                    int wstride,
                                    int hstride,
                                    int format) {
  static std::vector<std::unique_ptr<ScratchDmabuf>> scratch_buffers;

  for (auto& scratch : scratch_buffers) {
    if (scratch->Matches(size, width, height, wstride, hstride, format)) {
      return scratch.get();
    }
  }

  auto scratch = std::make_unique<ScratchDmabuf>();
  if (!scratch->Ensure(size, width, height, wstride, hstride, format)) {
    return nullptr;
  }
  scratch_buffers.push_back(std::move(scratch));
  return scratch_buffers.back().get();
}
#endif

cv::Rect ClampRect(const cv::Rect& rect, int max_w, int max_h) {
  const int x = std::clamp(rect.x, 0, max_w);
  const int y = std::clamp(rect.y, 0, max_h);
  const int right = std::clamp(rect.x + rect.width, 0, max_w);
  const int bottom = std::clamp(rect.y + rect.height, 0, max_h);
  return cv::Rect(x, y, std::max(0, right - x), std::max(0, bottom - y));
}

cv::Rect AlignRectForNv12(const cv::Rect& rect, int max_w, int max_h) {
  int x = std::clamp(rect.x & ~1, 0, std::max(0, max_w - 2));
  int y = std::clamp(rect.y & ~1, 0, std::max(0, max_h - 2));
  int right = std::clamp((rect.x + rect.width + 1) & ~1, x + 2, max_w);
  int bottom = std::clamp((rect.y + rect.height + 1) & ~1, y + 2, max_h);
  return cv::Rect(x, y, right - x, bottom - y);
}

PreprocessMeta BuildMeta(const cv::Rect& src_rect,
                         const cv::Size& target_size,
                         bool keep_aspect) {
  PreprocessMeta meta;
  meta.input_w = static_cast<float>(target_size.width);
  meta.input_h = static_cast<float>(target_size.height);
  meta.src_w = static_cast<float>(src_rect.width);
  meta.src_h = static_cast<float>(src_rect.height);

  if (!keep_aspect) {
    meta.scale = static_cast<float>(target_size.width) /
                 std::max(static_cast<float>(src_rect.width), 1.0f);
    return meta;
  }

  meta.scale = std::min(static_cast<float>(target_size.width) / src_rect.width,
                        static_cast<float>(target_size.height) / src_rect.height);
  const int resized_w = std::min(target_size.width,
                                 std::max(1, static_cast<int>(std::round(src_rect.width * meta.scale))));
  const int resized_h = std::min(target_size.height,
                                 std::max(1, static_cast<int>(std::round(src_rect.height * meta.scale))));
  meta.scale = static_cast<float>(resized_w) /
               std::max(static_cast<float>(src_rect.width), 1.0f);
  meta.pad_left = static_cast<float>(std::max(0, (target_size.width - resized_w) / 2));
  meta.pad_top = static_cast<float>(std::max(0, (target_size.height - resized_h) / 2));
  return meta;
}

#if defined(__linux__)
int FrameFormatToRga(uint32_t fourcc) {
  if (fourcc == v4l2_fourcc('N', 'V', '1', '2')) {
    return RK_FORMAT_YCbCr_420_SP;
  }
  return 0;
}

bool IsRgaSuccess(IM_STATUS status) {
  return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR;
}

bool ExceedsRgaScaleLimit(const cv::Rect& src_rect, const cv::Size& target_size) {
  return target_size.width > src_rect.width * 16 ||
         target_size.height > src_rect.height * 16 ||
         src_rect.width > target_size.width * 16 ||
         src_rect.height > target_size.height * 16;
}
#endif

}  // namespace

cv::Size TensorSizeFromAttr(const rknn_tensor_attr& attr) {
  if (attr.fmt == RKNN_TENSOR_NCHW) {
    return cv::Size(static_cast<int>(attr.dims[3]), static_cast<int>(attr.dims[2]));
  }
  return cv::Size(static_cast<int>(attr.dims[2]), static_cast<int>(attr.dims[1]));
}

bool PreprocessFrameToRknn(const CameraFrame& frame,
                           const cv::Rect& src_rect,
                           bool keep_aspect,
                           rknn_tensor_mem* dst_mem,
                           const rknn_tensor_attr& dst_attr,
                           PreprocessMeta* meta) {
  std::lock_guard<std::mutex> lock(g_hw_preprocess_mutex);

  if (frame.dmabuf_fd < 0 || dst_mem == nullptr) {
    return false;
  }

  if (!g_hw_preprocess_available) {
    return false;
  }

  cv::Rect clamped = ClampRect(src_rect, frame.width, frame.height);
  if (clamped.width <= 1 || clamped.height <= 1) {
    return false;
  }

  const cv::Size target_size = TensorSizeFromAttr(dst_attr);
  if (target_size.width <= 0 || target_size.height <= 0) {
    return false;
  }

  if (meta != nullptr) {
    *meta = BuildMeta(clamped, target_size, keep_aspect);
  }

#if !defined(__linux__)
  (void)keep_aspect;
  (void)dst_attr;
  return false;
#else
  const int src_format = FrameFormatToRga(frame.fourcc);
  if (src_format == 0) {
    std::cerr << "unsupported camera frame format for RGA: 0x"
              << std::hex << frame.fourcc << std::dec << "\n";
    return false;
  }

  if (src_format == RK_FORMAT_YCbCr_420_SP) {
    clamped = AlignRectForNv12(clamped, frame.width, frame.height);
  }
  if (clamped.width <= 1 || clamped.height <= 1 ||
      ExceedsRgaScaleLimit(clamped, target_size)) {
    return false;
  }

  const int dst_wstride = dst_attr.w_stride > 0 ? static_cast<int>(dst_attr.w_stride)
                                                : target_size.width;
  const int dst_hstride = dst_attr.h_stride > 0 ? static_cast<int>(dst_attr.h_stride)
                                                : target_size.height;
  const size_t dst_size = dst_attr.size_with_stride > 0 ? dst_attr.size_with_stride
                                                        : dst_attr.size;

  ScratchDmabuf* scratch = AcquireScratchBuffer(dst_size,
                                                target_size.width,
                                                target_size.height,
                                                dst_wstride,
                                                dst_hstride,
                                                RK_FORMAT_RGB_888);
  if (scratch == nullptr) {
    g_hw_preprocess_available = false;
    return false;
  }

  rga_buffer_t src = wrapbuffer_fd_t(frame.dmabuf_fd,
                                     frame.width,
                                     frame.height,
                                     frame.stride,
                                     frame.height,
                                     src_format);
  rga_buffer_t dst = wrapbuffer_fd_t(scratch->fd(),
                                     scratch->width(),
                                     scratch->height(),
                                     scratch->wstride(),
                                     scratch->hstride(),
                                     scratch->format());

  std::memset(scratch->addr(), 0, scratch->size());

  const PreprocessMeta current_meta = BuildMeta(clamped, target_size, keep_aspect);
  const int dst_w = keep_aspect
                        ? static_cast<int>(std::round(clamped.width * current_meta.scale))
                        : target_size.width;
  const int dst_h = keep_aspect
                        ? static_cast<int>(std::round(clamped.height * current_meta.scale))
                        : target_size.height;
  const int dst_x = keep_aspect ? static_cast<int>(std::round(current_meta.pad_left)) : 0;
  const int dst_y = keep_aspect ? static_cast<int>(std::round(current_meta.pad_top)) : 0;

  im_rect srect{clamped.x, clamped.y, clamped.width, clamped.height};
  im_rect drect{dst_x, dst_y, dst_w, dst_h};
  im_rect prect{0, 0, 0, 0};
  rga_buffer_t pat;
  std::memset(&pat, 0, sizeof(pat));

  // Some RK BSPs ship an imcheck() variadic macro that rejects an empty
  // __VA_ARGS__ under -Wpedantic, so call the underlying typed API directly.
  const IM_STATUS check = imcheck_t(src, dst, pat, srect, drect, prect, 0);
  if (!IsRgaSuccess(check)) {
    std::cerr << "RGA imcheck failed, disable hardware preprocess: "
              << imStrError(check) << "\n";
    g_hw_preprocess_available = false;
    return false;
  }

  const IM_STATUS status = improcess(src, dst, pat, srect, drect, prect, 0);
  if (!IsRgaSuccess(status)) {
    std::cerr << "RGA improcess failed, disable hardware preprocess: "
              << imStrError(status) << "\n";
    g_hw_preprocess_available = false;
    return false;
  }

  std::memcpy(dst_mem->virt_addr,
              scratch->addr(),
              std::min(static_cast<size_t>(dst_mem->size), scratch->size()));
  return true;
#endif
}

bool ConvertNv12ToRgb(int dmabuf_fd, int width, int height, int stride,
                      cv::Mat* rgb_out) {
  std::lock_guard<std::mutex> lock(g_hw_preprocess_mutex);

  if (dmabuf_fd < 0 || rgb_out == nullptr || width <= 0 || height <= 0) {
    return false;
  }
  if (!g_hw_preprocess_available) {
    return false;
  }

#if !defined(__linux__)
  return false;
#else
  const size_t dst_size = static_cast<size_t>(width) * height * 3;
  ScratchDmabuf* scratch = AcquireScratchBuffer(dst_size,
                                                width, height,
                                                width, height,
                                                RK_FORMAT_RGB_888);
  if (scratch == nullptr) {
    g_hw_preprocess_available = false;
    return false;
  }

  rga_buffer_t src = wrapbuffer_fd_t(dmabuf_fd,
                                     width, height,
                                     stride, height,
                                     RK_FORMAT_YCbCr_420_SP);
  rga_buffer_t dst = wrapbuffer_fd_t(scratch->fd(),
                                     scratch->width(),
                                     scratch->height(),
                                     scratch->wstride(),
                                     scratch->hstride(),
                                     scratch->format());

  im_rect srect{0, 0, width, height};
  im_rect drect{0, 0, width, height};
  im_rect prect{0, 0, 0, 0};
  rga_buffer_t pat;
  std::memset(&pat, 0, sizeof(pat));

  const IM_STATUS check = imcheck_t(src, dst, pat, srect, drect, prect, 0);
  if (!IsRgaSuccess(check)) {
    std::cerr << "RGA ConvertNv12ToRgb imcheck failed: "
              << imStrError(check) << "\n";
    g_hw_preprocess_available = false;
    return false;
  }

  const IM_STATUS status = improcess(src, dst, pat, srect, drect, prect, 0);
  if (!IsRgaSuccess(status)) {
    std::cerr << "RGA ConvertNv12ToRgb improcess failed: "
              << imStrError(status) << "\n";
    g_hw_preprocess_available = false;
    return false;
  }

  // Clone: scratch is a pooled static buffer reused across frames.
  cv::Mat rgb(height, width, CV_8UC3, scratch->addr(), static_cast<size_t>(width) * 3);
  *rgb_out = rgb.clone();
  return true;
#endif
}

}  // namespace mediapipe_demo
