#include "rk_studio/infra/telemetry.h"

#include <cmath>

#include "rk_studio/infra/runtime.h"

namespace rkinfra {

EventQueue::EventQueue(size_t max_size) : max_size_(max_size) {}

bool EventQueue::Push(const StreamEvent& event) {
  std::lock_guard<std::mutex> lock(mu_);
  if (closed_) {
    return false;
  }
  if (events_.size() >= max_size_) {
    events_.pop_front();
  }
  events_.push_back(event);
  cv_.notify_one();
  return true;
}

bool EventQueue::Pop(StreamEvent* out) {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [&] { return closed_ || !events_.empty(); });
  if (events_.empty()) {
    return false;
  }
  *out = events_.front();
  events_.pop_front();
  return true;
}

void EventQueue::Close() {
  std::lock_guard<std::mutex> lock(mu_);
  closed_ = true;
  cv_.notify_all();
}

SidecarWriter::SidecarWriter(EventQueue* queue, std::string path) : queue_(queue), path_(std::move(path)) {}

bool SidecarWriter::Start() {
  out_.open(path_, std::ios::out | std::ios::trunc);
  if (!out_.is_open()) {
    return false;
  }
  worker_ = std::thread([this] { Run(); });
  return true;
}

void SidecarWriter::Stop() {
  if (worker_.joinable()) {
    worker_.join();
  }
  if (out_.is_open()) {
    out_.flush();
    out_.close();
  }
}

void SidecarWriter::Run() {
  StreamEvent event;
  while (queue_->Pop(&event)) {
    out_ << "{\"monotonic_ns\":" << event.monotonic_ns
         << ",\"stream_id\":\"" << JsonEscape(event.stream_id)
         << "\",\"seq\":" << event.seq
         << ",\"pts_ns\":" << event.pts_ns
         << ",\"category\":\"" << JsonEscape(event.category)
         << "\",\"status\":\"" << JsonEscape(event.status)
         << "\",\"reason\":\"" << JsonEscape(event.reason)
         << "\",\"stage\":\"" << JsonEscape(event.stage)
         << "\"}\n";
  }
}

SyncAnalyzer::SyncAnalyzer(SyncConfig config, std::vector<std::string> stream_ids, std::string reference_stream_id)
    : config_(config), stream_ids_(std::move(stream_ids)), reference_stream_id_(std::move(reference_stream_id)) {
  for (const auto& stream_id : stream_ids_) {
    states_.emplace(stream_id, StreamState{});
  }
}

void SyncAnalyzer::Observe(const StreamEvent& event) {
  if (event.status != "ok" || event.pts_ns < 0 || event.stream_id.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mu_);
  auto& state = states_[event.stream_id];
  ++state.valid_event_count;
  if (!state.first_valid_monotonic_ns) {
    state.first_valid_monotonic_ns = event.monotonic_ns;
  }

  const uint64_t window_index = config_.window_ns == 0 ? 0 : event.monotonic_ns / config_.window_ns;
  state.first_sample_by_window.emplace(window_index, event.monotonic_ns);
}

SyncReport SyncAnalyzer::BuildReport() const {
  std::lock_guard<std::mutex> lock(mu_);

  SyncReport report;
  report.reference_stream_id = reference_stream_id_;
  report.window_size_ns = config_.window_ns;
  report.warning_threshold_ns = static_cast<uint64_t>(config_.max_delta_ms) * 1'000'000ULL;

  const auto ref_it = states_.find(reference_stream_id_);
  const StreamState* reference_state = ref_it != states_.end() ? &ref_it->second : nullptr;

  for (const auto& stream_id : stream_ids_) {
    StreamSyncSummary summary;
    summary.stream_id = stream_id;
    summary.is_reference = (stream_id == reference_stream_id_);

    const auto state_it = states_.find(stream_id);
    const StreamState* state = state_it != states_.end() ? &state_it->second : nullptr;
    if (!state) {
      report.streams.push_back(summary);
      continue;
    }

    summary.valid_event_count = state->valid_event_count;

    if (state->first_valid_monotonic_ns && reference_state && reference_state->first_valid_monotonic_ns) {
      summary.has_first_sample_offset_ns = true;
      summary.first_sample_offset_ns = summary.is_reference
                                           ? 0
                                           : static_cast<int64_t>(*state->first_valid_monotonic_ns) -
                                                 static_cast<int64_t>(*reference_state->first_valid_monotonic_ns);
    }

    if (summary.is_reference) {
      summary.has_delta_stats = true;
      summary.min_delta_ns = 0;
      summary.max_delta_ns = 0;
      summary.mean_abs_delta_ns = 0.0;
      summary.matched_window_count = state->first_sample_by_window.size();
      report.streams.push_back(summary);
      continue;
    }

    if (!reference_state) {
      report.streams.push_back(summary);
      continue;
    }

    bool have_delta = false;
    int64_t min_delta = 0;
    int64_t max_delta = 0;
    uint64_t warning_count = 0;
    long double abs_delta_sum = 0.0;

    for (const auto& [window_index, stream_ts] : state->first_sample_by_window) {
      const auto ref_window_it = reference_state->first_sample_by_window.find(window_index);
      if (ref_window_it == reference_state->first_sample_by_window.end()) {
        continue;
      }

      const int64_t delta = static_cast<int64_t>(stream_ts) - static_cast<int64_t>(ref_window_it->second);
      if (!have_delta) {
        min_delta = delta;
        max_delta = delta;
        have_delta = true;
      } else {
        min_delta = std::min(min_delta, delta);
        max_delta = std::max(max_delta, delta);
      }
      abs_delta_sum += std::llabs(delta);
      ++summary.matched_window_count;
      if (static_cast<uint64_t>(std::llabs(delta)) > report.warning_threshold_ns) {
        ++warning_count;
      }
    }

    if (have_delta && summary.matched_window_count > 0) {
      summary.has_delta_stats = true;
      summary.min_delta_ns = min_delta;
      summary.max_delta_ns = max_delta;
      summary.mean_abs_delta_ns = static_cast<double>(abs_delta_sum / summary.matched_window_count);
      summary.warning_count = warning_count;
    }

    report.streams.push_back(summary);
  }

  return report;
}

TelemetrySink::TelemetrySink(size_t queue_size,
                             std::string sidecar_path,
                             SyncConfig sync_config,
                             std::vector<std::string> stream_ids,
                             std::string reference_stream_id)
    : queue_(queue_size),
      sidecar_writer_(&queue_, std::move(sidecar_path)),
      sync_analyzer_(sync_config, std::move(stream_ids), std::move(reference_stream_id)) {}

bool TelemetrySink::Start(std::string* err) {
  if (sidecar_writer_.Start()) {
    return true;
  }
  if (err) {
    *err = "failed to open sidecar";
  }
  return false;
}

void TelemetrySink::Record(const StreamEvent& event) {
  sync_analyzer_.Observe(event);
  queue_.Push(event);
}

void TelemetrySink::Stop() {
  queue_.Close();
  sidecar_writer_.Stop();
}

SyncReport TelemetrySink::BuildSyncReport() const { return sync_analyzer_.BuildReport(); }

}  // namespace rkinfra
