#pragma once

#include <memory>
#include <optional>

#include "rk_studio/ai_core/ai_types.h"

namespace rkstudio::ai {

class IAiProcessor {
 public:
  virtual ~IAiProcessor() = default;

  virtual bool Start(const AiProcessorConfig& config, std::string* err) = 0;
  virtual void Submit(const FrameRef& frame) = 0;
  virtual std::optional<AiResult> PollResult() = 0;
  virtual void Stop() = 0;
};

std::unique_ptr<IAiProcessor> CreateHandAiProcessor();

}  // namespace rkstudio::ai
