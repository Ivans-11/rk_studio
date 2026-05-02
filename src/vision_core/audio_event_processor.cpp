#include "rk_studio/vision_core/vision_processor.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

namespace rkstudio::vision {
namespace {

struct OwnedAudioFrame {
  std::string source_id;
  uint64_t pts_ns = 0;
  int sample_rate = 16'000;
  int channels = 1;
  std::vector<int16_t> samples;
};

float Clamp01(float value) {
  return std::max(0.0f, std::min(1.0f, value));
}

}  // namespace

class AudioEventProcessor final : public IAudioEventProcessor {
 public:
  ~AudioEventProcessor() override { Stop(); }

  bool Start(const AudioEventProcessorConfig& config, std::string* err) override {
    Stop();
    if (config.sample_rate <= 0 || config.channels <= 0 ||
        config.window_ms <= 0 || config.hop_ms <= 0 || config.top_k <= 0) {
      if (err) *err = "invalid audio event processor config";
      return false;
    }
    config_ = config;
    running_ = true;
    worker_ = std::thread([this] { RunLoop(); });
    return true;
  }

  void Submit(const AudioPcmFrame& frame) override {
    if (frame.samples == nullptr || frame.sample_count == 0) {
      return;
    }
    OwnedAudioFrame owned;
    owned.source_id = frame.source_id;
    owned.pts_ns = frame.pts_ns;
    owned.sample_rate = frame.sample_rate;
    owned.channels = std::max(1, frame.channels);
    owned.samples.assign(frame.samples, frame.samples + frame.sample_count);

    std::lock_guard<std::mutex> lock(mu_);
    if (!running_) {
      return;
    }
    pending_frames_.push_back(std::move(owned));
    while (pending_frames_.size() > std::max<size_t>(1, config_.queue_depth)) {
      pending_frames_.pop_front();
    }
    cv_.notify_one();
  }

  std::optional<AudioEventResult> PollResult() override {
    std::lock_guard<std::mutex> lock(mu_);
    if (results_.empty()) {
      return std::nullopt;
    }
    AudioEventResult result = std::move(results_.front());
    results_.pop_front();
    return result;
  }

  void Stop() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      running_ = false;
      pending_frames_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      results_.clear();
    }
    mono_buffer_.clear();
    next_window_start_ = 0;
    buffer_start_pts_ns_ = 0;
    have_buffer_pts_ = false;
  }

 private:
  void RunLoop() {
    while (true) {
      OwnedAudioFrame frame;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !running_ || !pending_frames_.empty(); });
        if (!running_ && pending_frames_.empty()) {
          break;
        }
        frame = std::move(pending_frames_.front());
        pending_frames_.pop_front();
      }
      ProcessFrame(frame);
    }
  }

  void ProcessFrame(const OwnedAudioFrame& frame) {
    if (frame.sample_rate != config_.sample_rate) {
      AudioEventResult result;
      result.source_id = frame.source_id;
      result.pts_ns = frame.pts_ns;
      result.sample_rate = frame.sample_rate;
      result.window_ms = config_.window_ms;
      result.hop_ms = config_.hop_ms;
      result.error = "audio sample rate does not match audio_event config";
      PushResult(std::move(result));
      return;
    }

    const size_t channels = static_cast<size_t>(std::max(1, frame.channels));
    const size_t frames = frame.samples.size() / channels;
    if (frames == 0) {
      return;
    }
    if (!have_buffer_pts_) {
      buffer_start_pts_ns_ = frame.pts_ns;
      have_buffer_pts_ = true;
    }

    mono_buffer_.reserve(mono_buffer_.size() + frames);
    for (size_t i = 0; i < frames; ++i) {
      int32_t mixed = 0;
      for (size_t ch = 0; ch < channels; ++ch) {
        mixed += frame.samples[i * channels + ch];
      }
      mono_buffer_.push_back(static_cast<float>(mixed) /
                             static_cast<float>(channels * 32768.0f));
    }

    const size_t window_samples = static_cast<size_t>(
        static_cast<int64_t>(config_.sample_rate) * config_.window_ms / 1000);
    const size_t hop_samples = static_cast<size_t>(
        static_cast<int64_t>(config_.sample_rate) * config_.hop_ms / 1000);
    if (window_samples == 0 || hop_samples == 0) {
      return;
    }

    while (next_window_start_ + window_samples <= mono_buffer_.size()) {
      AudioEventResult result = AnalyzeWindow(frame.source_id, next_window_start_, window_samples);
      PushResult(std::move(result));
      next_window_start_ += hop_samples;
    }

    const size_t keep_before = next_window_start_;
    if (keep_before > 0 && keep_before >= hop_samples) {
      mono_buffer_.erase(mono_buffer_.begin(), mono_buffer_.begin() + static_cast<std::ptrdiff_t>(keep_before));
      buffer_start_pts_ns_ += static_cast<uint64_t>(
          (1'000'000'000ULL * keep_before) / static_cast<size_t>(config_.sample_rate));
      next_window_start_ = 0;
    }
  }

  AudioEventResult AnalyzeWindow(const std::string& source_id, size_t start, size_t count) const {
    AudioEventResult result;
    result.source_id = source_id;
    result.pts_ns = buffer_start_pts_ns_ + static_cast<uint64_t>(
        (1'000'000'000ULL * start) / static_cast<size_t>(config_.sample_rate));
    result.center_pts_ns = result.pts_ns + static_cast<uint64_t>(config_.window_ms) * 500'000ULL;
    result.sample_rate = config_.sample_rate;
    result.window_ms = config_.window_ms;
    result.hop_ms = config_.hop_ms;

    float sum_sq = 0.0f;
    float peak = 0.0f;
    int zero_crossings = 0;
    float prev = mono_buffer_[start];
    for (size_t i = 0; i < count; ++i) {
      const float sample = mono_buffer_[start + i];
      sum_sq += sample * sample;
      peak = std::max(peak, std::abs(sample));
      if (i > 0 && ((prev < 0.0f && sample >= 0.0f) ||
                    (prev >= 0.0f && sample < 0.0f))) {
        ++zero_crossings;
      }
      prev = sample;
    }

    result.rms = std::sqrt(sum_sq / static_cast<float>(std::max<size_t>(1, count)));
    result.peak = peak;

    const float zcr = static_cast<float>(zero_crossings) /
                      static_cast<float>(std::max<size_t>(1, count));
    const float loud_score = Clamp01((result.rms - 0.01f) / 0.18f);
    const float transient_score = Clamp01((peak - 0.08f) / 0.35f);
    const float speech_like_score = Clamp01((result.rms - 0.015f) / 0.10f) *
                                    Clamp01((0.18f - zcr) / 0.18f);

    std::vector<AudioEventScore> candidates;
    candidates.push_back({0, "Speech", speech_like_score});
    candidates.push_back({1, "Loud sound", loud_score});
    candidates.push_back({2, "Clicking", transient_score});
    candidates.push_back({3, "Silence", Clamp01((0.02f - result.rms) / 0.02f)});
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
      return a.score > b.score;
    });

    for (const auto& candidate : candidates) {
      if (candidate.score < config_.score_threshold) {
        continue;
      }
      result.events.push_back(candidate);
      if (static_cast<int>(result.events.size()) >= config_.top_k) {
        break;
      }
    }
    if (result.events.empty() && !candidates.empty()) {
      result.events.push_back(candidates.front());
    }
    result.ok = true;
    return result;
  }

  void PushResult(AudioEventResult result) {
    std::lock_guard<std::mutex> lock(mu_);
    results_.push_back(std::move(result));
    while (results_.size() > std::max<size_t>(1, config_.queue_depth * 4)) {
      results_.pop_front();
    }
  }

  AudioEventProcessorConfig config_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<OwnedAudioFrame> pending_frames_;
  std::deque<AudioEventResult> results_;
  std::vector<float> mono_buffer_;
  size_t next_window_start_ = 0;
  uint64_t buffer_start_pts_ns_ = 0;
  bool have_buffer_pts_ = false;
  bool running_ = false;
  std::thread worker_;
};

std::unique_ptr<IAudioEventProcessor> CreateAudioEventProcessor() {
  return std::make_unique<AudioEventProcessor>();
}

}  // namespace rkstudio::vision
