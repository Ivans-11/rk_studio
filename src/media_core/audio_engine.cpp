#include "rk_studio/media_core/audio_engine.h"

#include <iostream>
#include <sstream>

#include "rk_studio/infra/runtime.h"
#include "rk_studio/infra/zenoh_publisher.h"
#include "rk_studio/media_core/session_writer.h"
#include "rk_studio/vision_core/vision_processor.h"

namespace rkstudio::media {
namespace {

std::string AudioEventResultToJson(const rkstudio::vision::AudioEventResult& r) {
  std::ostringstream o;
  o << "{\"source_id\":\"" << rkinfra::JsonEscape(r.source_id)
    << "\",\"pts_ns\":" << r.pts_ns
    << ",\"center_pts_ns\":" << r.center_pts_ns
    << ",\"sample_rate\":" << r.sample_rate
    << ",\"window_ms\":" << r.window_ms
    << ",\"hop_ms\":" << r.hop_ms
    << ",\"rms\":" << r.rms
    << ",\"peak\":" << r.peak
    << ",\"events\":[";
  for (size_t i = 0; i < r.events.size(); ++i) {
    if (i > 0) o << ',';
    const auto& event = r.events[i];
    o << "{\"class_id\":" << event.class_id
      << ",\"label\":\"" << rkinfra::JsonEscape(event.label)
      << "\",\"score\":" << event.score << '}';
  }
  o << "]}";
  return o.str();
}

}  // namespace

AudioEngine::AudioEngine(QObject* parent) : QObject(parent) {
  qRegisterMetaType<rkstudio::vision::AudioEventResult>();
  audio_event_poll_timer_ = new QTimer(this);
  audio_event_poll_timer_->setInterval(50);
  connect(audio_event_poll_timer_, &QTimer::timeout, this, &AudioEngine::PollAudioEventResults);
}

AudioEngine::~AudioEngine() {
  StopAll();
}

void AudioEngine::LoadBoardConfig(const BoardConfig& board_config) {
  board_config_ = board_config;
}

void AudioEngine::ApplySessionProfile(const SessionProfile& profile) {
  session_profile_ = profile;
}

void AudioEngine::SetSessionWriter(SessionWriter* session_writer) {
  session_writer_ = session_writer;
  if (session_writer_ && session_writer_->session_paths() && audio_event_enabled_) {
    session_writer_->OpenAudioEventWriter(nullptr);
  }
}

void AudioEngine::SetZenohPublisher(rkinfra::ZenohPublisher* zenoh_publisher) {
  zenoh_publisher_ = zenoh_publisher;
}

bool AudioEngine::ToggleAudioEvent(bool enable, std::string* err) {
  if (enable == audio_event_enabled_) {
    return true;
  }
  audio_event_enabled_ = enable;
  if (enable) {
    if (!StartProcessor(err)) {
      audio_event_enabled_ = false;
      return false;
    }
    if (session_writer_ && session_writer_->session_paths()) {
      session_writer_->OpenAudioEventWriter(nullptr);
    }
    audio_event_poll_timer_->start();
    return true;
  }

  audio_event_poll_timer_->stop();
  StopProcessor();
  return true;
}

void AudioEngine::SubmitPcmFrame(const vision::AudioPcmFrame& frame) {
  if (!audio_event_processor_ || !audio_event_enabled_) {
    return;
  }
  audio_event_processor_->Submit(frame);
}

void AudioEngine::StopAll() {
  audio_event_enabled_ = false;
  audio_event_poll_timer_->stop();
  StopProcessor();
}

bool AudioEngine::StartProcessor(std::string* err) {
  StopProcessor();
  if (session_profile_.audio_source.empty()) {
    if (err) *err = "no audio source configured";
    return false;
  }
  const AudioSource* source = FindAudioSource(board_config_, session_profile_.audio_source);
  if (source == nullptr) {
    if (err) *err = "unknown audio source: " + session_profile_.audio_source;
    return false;
  }
  if (!board_config_.audio_event.has_value()) {
    if (err) *err = "no audio_event config available";
    return false;
  }

  const auto& audio_event = *board_config_.audio_event;
  vision::AudioEventProcessorConfig config;
  config.model = audio_event.model;
  config.class_map = audio_event.class_map;
  config.sample_rate = source->rate;
  config.channels = source->channels;
  config.queue_depth = 4;
  config.window_ms = audio_event.window_ms;
  config.hop_ms = audio_event.hop_ms;
  config.top_k = audio_event.top_k;
  config.score_threshold = static_cast<float>(audio_event.score_threshold);

  audio_event_processor_ = vision::CreateAudioEventProcessor();
  std::string audio_err;
  if (!audio_event_processor_->Start(config, &audio_err)) {
    if (err) *err = audio_err;
    audio_event_processor_.reset();
    return false;
  }
  audio_source_id_ = source->id;
  std::cerr << "[audio] event processor started for " << audio_source_id_ << "\n";
  return true;
}

void AudioEngine::StopProcessor() {
  if (audio_event_processor_) {
    audio_event_processor_->Stop();
    audio_event_processor_.reset();
    std::cerr << "[audio] event processor stopped\n";
  }
  audio_source_id_.clear();
}

void AudioEngine::PollAudioEventResults() {
  if (!audio_event_processor_) {
    return;
  }
  while (auto result = audio_event_processor_->PollResult()) {
    std::string payload;
    const bool has_events = result->ok && !result->events.empty();
    if (session_writer_ && has_events) {
      payload = AudioEventResultToJson(*result);
      session_writer_->WriteAudioEventLine(payload);
    }
    if (zenoh_publisher_ && zenoh_publisher_->active() &&
        zenoh_publisher_->result_publishing_enabled() && has_events) {
      bool publishable = false;
      const double threshold = board_config_.audio_event.has_value()
                                   ? board_config_.audio_event->publish_threshold
                                   : 0.0;
      for (const auto& event : result->events) {
        if (event.score >= threshold) {
          publishable = true;
          break;
        }
      }
      if (publishable) {
        if (payload.empty()) {
          payload = AudioEventResultToJson(*result);
        }
        zenoh_publisher_->PublishAudioEvent(result->source_id, payload);
      }
    }
    emit AudioEventResultReady(*result);
  }
}

}  // namespace rkstudio::media
