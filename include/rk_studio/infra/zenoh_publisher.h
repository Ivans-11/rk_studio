#pragma once

#include <memory>
#include <string>

#include "rk_studio/domain/types.h"

namespace rkinfra {

class ZenohPublisher {
 public:
  ZenohPublisher();
  ~ZenohPublisher();

  ZenohPublisher(const ZenohPublisher&) = delete;
  ZenohPublisher& operator=(const ZenohPublisher&) = delete;

  bool Start(const rkstudio::ZenohConfig& config, std::string* err);
  void Stop();
  bool active() const;
  void SetResultPublishingEnabled(bool enabled);
  bool result_publishing_enabled() const;

  bool PublishMediapipe(const std::string& camera_id, const std::string& payload);
  bool PublishYolo(const std::string& camera_id, const std::string& payload);
  bool PublishJson(const std::string& key, const std::string& payload);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rkinfra
