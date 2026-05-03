#include "rk_studio/vision_core/vision_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "mediapipe/common/rknn_model.h"

namespace rkstudio::vision {
namespace {

constexpr int kYamnetSampleRate = 16'000;
constexpr int kYamnetWindowSamples = 15'360;
constexpr int kYamnetFrameLength = 400;
constexpr int kYamnetFrameStep = 160;
constexpr int kYamnetFftBins = 257;
constexpr int kYamnetMelBins = 64;
constexpr int kYamnetPatchFrames = 96;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kYamnetLogOffset = 0.001f;
constexpr float kYamnetLowerHz = 125.0f;
constexpr float kYamnetUpperHz = 7500.0f;

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

float HzToMel(float hz) {
  return 1127.0f * std::log1p(hz / 700.0f);
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> cols;
  std::string current;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      cols.push_back(current);
      current.clear();
    } else {
      current.push_back(ch);
    }
  }
  cols.push_back(current);
  return cols;
}

std::vector<std::string> LoadClassMap(const std::string& path) {
  std::vector<std::string> labels;
  std::ifstream file(path);
  if (!file.is_open()) {
    return labels;
  }

  std::string line;
  bool first = true;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }

    const auto cols = SplitCsvLine(line);
    if (cols.empty()) {
      continue;
    }
    if (first) {
      first = false;
      bool header = false;
      for (const auto& col : cols) {
        if (col == "index" || col == "mid" || col == "display_name" || col == "name") {
          header = true;
          break;
        }
      }
      if (header) {
        continue;
      }
    }

    std::string label;
    if (cols.size() >= 3) {
      label = cols[2];
    } else if (cols.size() >= 2) {
      label = cols[1];
    } else {
      label = cols[0];
    }
    if (!label.empty()) {
      labels.push_back(label);
    }
  }
  return labels;
}

std::vector<std::vector<float>> BuildMelWeightMatrix() {
  std::vector<std::vector<float>> weights(
      kYamnetFftBins, std::vector<float>(kYamnetMelBins, 0.0f));
  const float lower_mel = HzToMel(kYamnetLowerHz);
  const float upper_mel = HzToMel(kYamnetUpperHz);
  std::vector<float> mel_edges(kYamnetMelBins + 2, 0.0f);
  for (int i = 0; i < static_cast<int>(mel_edges.size()); ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(mel_edges.size() - 1);
    mel_edges[i] = lower_mel + t * (upper_mel - lower_mel);
  }

  for (int fft_bin = 0; fft_bin < kYamnetFftBins; ++fft_bin) {
    const float hz = static_cast<float>(fft_bin * kYamnetSampleRate) /
                     static_cast<float>((kYamnetFftBins - 1) * 2);
    const float mel = HzToMel(hz);
    for (int mel_bin = 0; mel_bin < kYamnetMelBins; ++mel_bin) {
      const float lower = mel_edges[mel_bin];
      const float center = mel_edges[mel_bin + 1];
      const float upper = mel_edges[mel_bin + 2];
      float weight = 0.0f;
      if (mel >= lower && mel <= center) {
        weight = (mel - lower) / std::max(1e-6f, center - lower);
      } else if (mel > center && mel <= upper) {
        weight = (upper - mel) / std::max(1e-6f, upper - center);
      }
      weights[fft_bin][mel_bin] = std::max(0.0f, weight);
    }
  }
  return weights;
}

const std::vector<std::vector<float>>& MelWeights() {
  static const auto weights = BuildMelWeightMatrix();
  return weights;
}

std::array<float, kYamnetFrameLength> HannWindow() {
  std::array<float, kYamnetFrameLength> window{};
  for (int i = 0; i < kYamnetFrameLength; ++i) {
    window[static_cast<size_t>(i)] =
        0.5f - 0.5f * std::cos(2.0f * kPi *
                               static_cast<float>(i) /
                               static_cast<float>(kYamnetFrameLength));
  }
  return window;
}

struct DftTables {
  std::array<std::array<float, kYamnetFrameLength>, kYamnetFftBins> cos{};
  std::array<std::array<float, kYamnetFrameLength>, kYamnetFftBins> sin{};
};

DftTables BuildDftTables() {
  DftTables tables;
  for (int bin = 0; bin < kYamnetFftBins; ++bin) {
    for (int n = 0; n < kYamnetFrameLength; ++n) {
      const float angle = -2.0f * kPi *
                          static_cast<float>(bin * n) /
                          static_cast<float>((kYamnetFftBins - 1) * 2);
      tables.cos[static_cast<size_t>(bin)][static_cast<size_t>(n)] = std::cos(angle);
      tables.sin[static_cast<size_t>(bin)][static_cast<size_t>(n)] = std::sin(angle);
    }
  }
  return tables;
}

std::array<float, kYamnetFftBins> MagnitudeSpectrum(const std::vector<float>& samples,
                                                    size_t frame_start,
                                                    size_t window_start) {
  static const auto window = HannWindow();
  static const auto tables = BuildDftTables();
  std::array<float, kYamnetFftBins> power{};
  for (int bin = 0; bin < kYamnetFftBins; ++bin) {
    float real = 0.0f;
    float imag = 0.0f;
    for (int n = 0; n < kYamnetFrameLength; ++n) {
      const size_t sample_index = frame_start + static_cast<size_t>(n);
      const float raw = sample_index < window_start + kYamnetWindowSamples
                            ? samples[sample_index]
                            : 0.0f;
      const float sample = raw * window[static_cast<size_t>(n)];
      real += sample * tables.cos[static_cast<size_t>(bin)][static_cast<size_t>(n)];
      imag += sample * tables.sin[static_cast<size_t>(bin)][static_cast<size_t>(n)];
    }
    power[static_cast<size_t>(bin)] = std::sqrt(real * real + imag * imag);
  }
  return power;
}

cv::Mat BuildYamnetLogMelPatch(const std::vector<float>& samples, size_t start) {
  if (start + kYamnetWindowSamples > samples.size()) {
    return {};
  }

  cv::Mat patch(kYamnetPatchFrames, kYamnetMelBins, CV_32FC1);
  const auto& mel_weights = MelWeights();
  for (int frame_idx = 0; frame_idx < kYamnetPatchFrames; ++frame_idx) {
    const size_t frame_start = start + static_cast<size_t>(frame_idx * kYamnetFrameStep);
    const auto magnitude = MagnitudeSpectrum(samples, frame_start, start);
    for (int mel_bin = 0; mel_bin < kYamnetMelBins; ++mel_bin) {
      float mel_energy = 0.0f;
      for (int fft_bin = 0; fft_bin < kYamnetFftBins; ++fft_bin) {
        mel_energy += magnitude[static_cast<size_t>(fft_bin)] *
                      mel_weights[static_cast<size_t>(fft_bin)][static_cast<size_t>(mel_bin)];
      }
      patch.at<float>(frame_idx, mel_bin) = std::log(mel_energy + kYamnetLogOffset);
    }
  }
  return patch;
}

cv::Mat QuantizePatchToInputType(const cv::Mat& patch,
                                 const rknn_tensor_attr& attr) {
  if (patch.empty() || patch.type() != CV_32FC1) {
    return {};
  }
  if (attr.type == RKNN_TENSOR_FLOAT32) {
    return patch.clone();
  }

  if (attr.type == RKNN_TENSOR_UINT8) {
    const float scale = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                                std::abs(attr.scale) > 1e-9f
                            ? attr.scale
                            : 1.0f;
    const int zp = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ? attr.zp : 0;
    cv::Mat quantized(patch.rows, patch.cols, CV_8UC1);
    for (int y = 0; y < patch.rows; ++y) {
      for (int x = 0; x < patch.cols; ++x) {
        const float value = patch.at<float>(y, x);
        const int q = static_cast<int>(std::lround(value / scale)) + zp;
        quantized.at<uint8_t>(y, x) = static_cast<uint8_t>(std::clamp(q, 0, 255));
      }
    }
    return quantized;
  }

  if (attr.type == RKNN_TENSOR_INT8) {
    const float scale = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                                std::abs(attr.scale) > 1e-9f
                            ? attr.scale
                            : 1.0f;
    const int zp = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ? attr.zp : 0;
    cv::Mat quantized(patch.rows, patch.cols, CV_8SC1);
    for (int y = 0; y < patch.rows; ++y) {
      for (int x = 0; x < patch.cols; ++x) {
        const float value = patch.at<float>(y, x);
        const int q = static_cast<int>(std::lround(value / scale)) + zp;
        quantized.at<int8_t>(y, x) = static_cast<int8_t>(std::clamp(q, -128, 127));
      }
    }
    return quantized;
  }

  std::cerr << "[audio] unsupported RKNN input tensor type for YAMNet: "
            << static_cast<int>(attr.type) << "\n";
  return {};
}

std::vector<AudioEventScore> TopScores(const std::vector<float>& scores,
                                       const std::vector<std::string>& labels,
                                       int top_k,
                                       float threshold) {
  std::vector<AudioEventScore> candidates;
  candidates.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    const float score = scores[i];
    if (!std::isfinite(score) || score < threshold) {
      continue;
    }
    candidates.push_back({static_cast<int>(i),
                          i < labels.size() ? labels[i] : ("class_" + std::to_string(i)),
                          score});
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  if (static_cast<int>(candidates.size()) > top_k) {
    candidates.resize(static_cast<size_t>(top_k));
  }
  return candidates;
}

class YamnetRknnBackend {
 public:
  bool Start(const AudioEventProcessorConfig& config) {
    labels_ = LoadClassMap(config.class_map);
    if (labels_.empty()) {
      std::cerr << "[audio] YAMNet class map unavailable, using rule fallback: "
                << config.class_map << "\n";
      return false;
    }
    if (!model_.Load(config.model)) {
      labels_.clear();
      std::cerr << "[audio] YAMNet RKNN model unavailable, using rule fallback: "
                << config.model << "\n";
      return false;
    }
    std::cerr << "[audio] YAMNet RKNN backend enabled: " << config.model << "\n";
    return true;
  }

  std::vector<AudioEventScore> Infer(const cv::Mat& patch,
                                     const AudioEventProcessorConfig& config) {
    if (patch.empty() || patch.type() != CV_32FC1) {
      return {};
    }
    const cv::Mat input = QuantizePatchToInputType(patch, model_.InputAttr());
    if (input.empty()) {
      return {};
    }
    std::vector<std::vector<float>> outputs;
    if (!model_.Infer(input, &outputs) || outputs.empty()) {
      return {};
    }
    return TopScores(outputs.front(), labels_, config.top_k, config.score_threshold);
  }

 private:
  mediapipe_demo::RknnModel model_;
  std::vector<std::string> labels_;
};

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
    yamnet_enabled_ = config_.sample_rate == kYamnetSampleRate &&
                      config_.window_ms == 960 &&
                      yamnet_.Start(config_);
    if (!yamnet_enabled_) {
      std::cerr << "[audio] rule fallback backend enabled\n";
    }
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
    yamnet_enabled_ = false;
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

  AudioEventResult AnalyzeWindow(const std::string& source_id, size_t start, size_t count) {
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

    if (yamnet_enabled_ && count == static_cast<size_t>(kYamnetWindowSamples)) {
      const cv::Mat patch = BuildYamnetLogMelPatch(mono_buffer_, start);
      result.events = yamnet_.Infer(patch, config_);
      if (!result.events.empty()) {
        result.ok = true;
        return result;
      }
      std::cerr << "[audio] YAMNet RKNN inference produced no events, using rule fallback\n";
    }

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
  YamnetRknnBackend yamnet_;
  bool yamnet_enabled_ = false;
};

std::unique_ptr<IAudioEventProcessor> CreateAudioEventProcessor() {
  return std::make_unique<AudioEventProcessor>();
}

}  // namespace rkstudio::vision
