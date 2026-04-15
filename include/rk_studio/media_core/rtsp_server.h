#pragma once

#include <string>
#include <vector>

#include "rk_studio/domain/types.h"

struct _GstRTSPServer;

namespace rkstudio::media {

class RtspServer {
 public:
  RtspServer() = default;
  ~RtspServer();

  RtspServer(const RtspServer&) = delete;
  RtspServer& operator=(const RtspServer&) = delete;

  bool Start(const BoardConfig& board, const SessionProfile& profile, std::string* err);
  void Stop();
  bool is_running() const { return server_ != nullptr; }
  int port() const { return port_; }

 private:
  _GstRTSPServer* server_ = nullptr;
  unsigned int attach_id_ = 0;
  int port_ = 0;
};

}  // namespace rkstudio::media
