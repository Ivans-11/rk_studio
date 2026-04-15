#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "rk_studio/infra/config_types.h"
#include "rk_studio/infra/telemetry.h"

namespace rkinfra {

namespace fs = std::filesystem;

struct SessionPaths {
  std::string session_id;
  fs::path session_dir;
  fs::path sidecar_path;
  fs::path meta_path;
  fs::path sync_path;
};

SessionPaths CreateSessionPaths(const RecordConfig& record_config);
bool EnsureSessionDirectory(const SessionPaths& paths, std::string* err);

void WriteSessionMeta(const SessionPaths& paths,
                      const AppConfig& config,
                      const std::string& state,
                      const std::string& started_utc,
                      uint64_t start_monotonic_ns,
                      uint64_t pipeline_base_time_ns,
                      const std::vector<OutputStreamInfo>& streams,
                      const std::string& reference_stream_id);

void WriteSyncReport(const fs::path& path, const SyncReport& report);

}  // namespace rkinfra
