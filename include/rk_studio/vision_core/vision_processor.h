#pragma once

#include <memory>
#include <optional>

#include "rk_studio/vision_core/vision_types.h"

namespace rkstudio::vision {

class IMediapipeProcessor {
 public:
  virtual ~IMediapipeProcessor() = default;

  virtual bool Start(const MediapipeProcessorConfig& config, std::string* err) = 0;
  virtual void Submit(const FrameRef& frame) = 0;
  virtual std::optional<MediapipeResult> PollResult() = 0;
  virtual void Stop() = 0;
};

class IYoloProcessor {
 public:
  virtual ~IYoloProcessor() = default;

  virtual bool Start(const YoloProcessorConfig& config, std::string* err) = 0;
  virtual void Submit(const FrameRef& frame) = 0;
  virtual std::optional<YoloResult> PollResult() = 0;
  virtual void Stop() = 0;
};

std::unique_ptr<IMediapipeProcessor> CreateMediapipeProcessor();
std::unique_ptr<IYoloProcessor> CreateYoloProcessor();

}  // namespace rkstudio::vision
