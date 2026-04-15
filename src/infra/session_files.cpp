#include "rk_studio/infra/session_files.h"

#include <fstream>
#include <iomanip>

#include "rk_studio/infra/runtime.h"

namespace rkinfra {

SessionPaths CreateSessionPaths(const RecordConfig& record_config) {
  SessionPaths paths;
  paths.session_id = record_config.prefix + "-" + NowLocalCompact();
  paths.session_dir = fs::path(record_config.output_dir) / paths.session_id;
  paths.sidecar_path = paths.session_dir / "session.sidecar.jsonl";
  paths.meta_path = paths.session_dir / "session.meta.json";
  paths.sync_path = paths.session_dir / "session.sync.json";
  return paths;
}

bool EnsureSessionDirectory(const SessionPaths& paths, std::string* err) {
  try {
    fs::create_directories(paths.session_dir);
    return true;
  } catch (const std::exception& ex) {
    if (err) {
      *err = std::string("failed to create session dir: ") + ex.what();
    }
    return false;
  }
}

void WriteSessionMeta(const SessionPaths& paths,
                      const AppConfig& config,
                      const std::string& state,
                      const std::string& started_utc,
                      uint64_t start_monotonic_ns,
                      uint64_t pipeline_base_time_ns,
                      const std::vector<OutputStreamInfo>& streams,
                      const std::string& reference_stream_id) {
  std::ofstream out(paths.meta_path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return;
  }

  out << "{\n";
  out << "  \"state\": \"" << JsonEscape(state) << "\",\n";
  out << "  \"started_utc\": \"" << JsonEscape(started_utc) << "\",\n";
  out << "  \"start_monotonic_ns\": " << start_monotonic_ns << ",\n";
  out << "  \"pipeline_base_time_ns\": " << pipeline_base_time_ns << ",\n";
  out << "  \"reference_stream_id\": \"" << JsonEscape(reference_stream_id) << "\",\n";
  out << "  \"files\": {\n";
  out << "    \"sidecar\": \"" << JsonEscape(paths.sidecar_path.string()) << "\",\n";
  out << "    \"meta\": \"" << JsonEscape(paths.meta_path.string()) << "\",\n";
  out << "    \"sync\": \"" << JsonEscape(paths.sync_path.string()) << "\"\n";
  out << "  },\n";

  out << "  \"streams\": [\n";
  for (size_t i = 0; i < streams.size(); ++i) {
    const auto& stream = streams[i];
    out << "    {\n";
    out << "      \"id\": \"" << JsonEscape(stream.id) << "\",\n";
    out << "      \"type\": \"" << JsonEscape(stream.type) << "\",\n";
    out << "      \"device\": \"" << JsonEscape(stream.device) << "\",\n";
    out << "      \"codec\": \"" << JsonEscape(stream.codec) << "\",\n";
    out << "      \"output_path\": \"" << JsonEscape(stream.output_path) << "\"\n";
    out << "    }" << (i + 1 < streams.size() ? "," : "") << "\n";
  }
  out << "  ],\n";

  out << "  \"record\": {\n";
  out << "    \"output_dir\": \"" << JsonEscape(config.record.output_dir) << "\",\n";
  out << "    \"prefix\": \"" << JsonEscape(config.record.prefix) << "\"\n";
  out << "  },\n";

  out << "  \"video_config\": [\n";
  for (size_t i = 0; i < config.video_streams.size(); ++i) {
    const auto& video = config.video_streams[i];
    out << "    {\n";
    out << "      \"id\": \"" << JsonEscape(video.id) << "\",\n";
    out << "      \"device\": \"" << JsonEscape(video.device) << "\",\n";
    out << "      \"width\": " << video.width << ",\n";
    out << "      \"height\": " << video.height << ",\n";
    out << "      \"fps\": " << video.fps << ",\n";
    out << "      \"bitrate\": " << video.bitrate << ",\n";
    out << "      \"gop\": " << video.gop << ",\n";
    out << "      \"queue_capture_buffers\": " << video.queue_capture_buffers << ",\n";
    out << "      \"queue_mux_max_time_ns\": " << video.queue_mux_max_time_ns << ",\n";
    out << "      \"io_mode\": \"" << JsonEscape(video.io_mode) << "\",\n";
    out << "      \"input_format\": \"" << JsonEscape(video.input_format) << "\"\n";
    out << "    }" << (i + 1 < config.video_streams.size() ? "," : "") << "\n";
  }
  out << "  ],\n";

  out << "  \"audio\": ";
  if (config.audio) {
    out << "{\n";
    out << "    \"id\": \"" << JsonEscape(config.audio->id) << "\",\n";
    out << "    \"device\": \"" << JsonEscape(config.audio->device) << "\",\n";
    out << "    \"rate\": " << config.audio->rate << ",\n";
    out << "    \"channels\": " << config.audio->channels << "\n";
    out << "  },\n";
  } else {
    out << "null,\n";
  }

  out << "  \"queue\": {\n";
  out << "    \"video_capture_buffers\": " << config.queue.video_capture_buffers << ",\n";
  out << "    \"video_mux_max_time_ns\": " << config.queue.video_mux_max_time_ns << ",\n";
  out << "    \"audio_mux_max_time_ns\": " << config.queue.audio_mux_max_time_ns << "\n";
  out << "  },\n";

  out << "  \"sync\": {\n";
  out << "    \"max_delta_ms\": " << config.sync.max_delta_ms << ",\n";
  out << "    \"window_ns\": " << config.sync.window_ns << "\n";
  out << "  },\n";

  out << "  \"encoder\": {\n";
  out << "    \"gop\": " << config.encoder.gop << ",\n";
  out << "    \"force_key_unit_interval_ms\": " << config.encoder.force_key_unit_interval_ms << "\n";
  out << "  }\n";
  out << "}\n";
}

void WriteSyncReport(const fs::path& path, const SyncReport& report) {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return;
  }

  out << std::fixed << std::setprecision(3);
  out << "{\n";
  out << "  \"reference_stream_id\": \"" << JsonEscape(report.reference_stream_id) << "\",\n";
  out << "  \"window_size_ns\": " << report.window_size_ns << ",\n";
  out << "  \"warning_threshold_ns\": " << report.warning_threshold_ns << ",\n";
  out << "  \"streams\": [\n";
  for (size_t i = 0; i < report.streams.size(); ++i) {
    const auto& stream = report.streams[i];
    out << "    {\n";
    out << "      \"stream_id\": \"" << JsonEscape(stream.stream_id) << "\",\n";
    out << "      \"is_reference\": " << (stream.is_reference ? "true" : "false") << ",\n";
    out << "      \"valid_event_count\": " << stream.valid_event_count << ",\n";
    out << "      \"matched_window_count\": " << stream.matched_window_count << ",\n";
    out << "      \"first_sample_offset_ns\": ";
    if (stream.has_first_sample_offset_ns) {
      out << stream.first_sample_offset_ns;
    } else {
      out << "null";
    }
    out << ",\n";
    out << "      \"min_delta_ns\": ";
    if (stream.has_delta_stats) {
      out << stream.min_delta_ns;
    } else {
      out << "null";
    }
    out << ",\n";
    out << "      \"max_delta_ns\": ";
    if (stream.has_delta_stats) {
      out << stream.max_delta_ns;
    } else {
      out << "null";
    }
    out << ",\n";
    out << "      \"mean_abs_delta_ns\": ";
    if (stream.has_delta_stats) {
      out << stream.mean_abs_delta_ns;
    } else {
      out << "null";
    }
    out << ",\n";
    out << "      \"warning_count\": " << stream.warning_count << "\n";
    out << "    }" << (i + 1 < report.streams.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

}  // namespace rkinfra
