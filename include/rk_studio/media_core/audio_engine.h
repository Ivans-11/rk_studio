#pragma once

#include <memory>
#include <string>

#include <QObject>
#include <QTimer>

#include "rk_studio/domain/types.h"
#include "rk_studio/vision_core/vision_types.h"

namespace rkstudio::vision {
class IAudioEventProcessor;
}  // namespace rkstudio::vision

namespace rkinfra {
class ZenohPublisher;
}  // namespace rkinfra

namespace rkstudio::media {

class SessionWriter;

class AudioEngine : public QObject {
  Q_OBJECT

 public:
  explicit AudioEngine(QObject* parent = nullptr);
  ~AudioEngine() override;

  void LoadBoardConfig(const BoardConfig& board_config);
  void ApplySessionProfile(const SessionProfile& profile);
  void SetSessionWriter(SessionWriter* session_writer);
  void SetZenohPublisher(rkinfra::ZenohPublisher* zenoh_publisher);

  bool ToggleAudioEvent(bool enable, std::string* err);
  void SubmitPcmFrame(const vision::AudioPcmFrame& frame);
  void StopAll();

  bool audio_event_enabled() const { return audio_event_enabled_; }
  std::string audio_source_id() const { return audio_source_id_; }

 signals:
  void AudioEventResultReady(rkstudio::vision::AudioEventResult result);

 private:
  bool StartProcessor(std::string* err);
  void StopProcessor();
  void PollAudioEventResults();

  BoardConfig board_config_;
  SessionProfile session_profile_;
  SessionWriter* session_writer_ = nullptr;
  rkinfra::ZenohPublisher* zenoh_publisher_ = nullptr;
  std::unique_ptr<rkstudio::vision::IAudioEventProcessor> audio_event_processor_;
  QTimer* audio_event_poll_timer_ = nullptr;
  std::string audio_source_id_;
  bool audio_event_enabled_ = false;
};

}  // namespace rkstudio::media

Q_DECLARE_METATYPE(rkstudio::vision::AudioEventResult)
