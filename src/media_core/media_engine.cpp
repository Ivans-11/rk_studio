#include "rk_studio/media_core/media_engine.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>

#include <gst/gst.h>
#include <gst/video/video.h>
#include <QMetaObject>

#include <opencv2/imgproc.hpp>

#include <gst/allocators/gstdmabuf.h>

#include "rk_studio/ai_core/ai_processor.h"
#include "rk_studio/infra/gst_audio_recorder.h"
#include "rk_studio/infra/runtime.h"
#include "rk_studio/infra/config_types.h"
#include "rk_studio/infra/session_files.h"
#include "rk_studio/infra/telemetry.h"
#include "rk_studio/media_core/session_writer.h"
#include "mediapipe/preprocess/hw_preprocess.h"

namespace rkstudio::media {
namespace {

bool Contains(const std::vector<std::string>& items, const std::string& value) {
  return std::find(items.begin(), items.end(), value) != items.end();
}

int ExtractDmabufFd(GstBuffer* buffer) {
  GstMemory* mem = gst_buffer_peek_memory(buffer, 0);
  if (mem != nullptr && gst_is_dmabuf_memory(mem)) {
    return gst_dmabuf_memory_get_fd(mem);
  }
  return -1;
}

std::vector<rkinfra::OutputStreamInfo> CollectOutputs(
    const std::map<std::string, std::unique_ptr<CameraPipeline>>& cameras,
    const rkinfra::GstAudioRecorder* audio_recorder) {
  std::vector<rkinfra::OutputStreamInfo> outputs;
  for (const auto& [camera_id, pipeline] : cameras) {
    (void)camera_id;
    const std::string path = pipeline->record_output_path();
    if (path.empty()) {
      continue;
    }
    rkinfra::OutputStreamInfo output;
    output.id = pipeline->camera_id();
    output.type = "video";
    output.device = pipeline->resolved_device();
    output.codec = "h265";
    output.output_path = path;
    outputs.push_back(std::move(output));
  }
  if (audio_recorder != nullptr) {
    outputs.push_back(audio_recorder->stream_output());
  }
  return outputs;
}

std::string AiResultToJson(const rkstudio::ai::AiResult& r) {
  std::ostringstream o;
  o << "{\"camera_id\":\"" << rkinfra::JsonEscape(r.camera_id)
    << "\",\"pts_ns\":" << r.pts_ns
    << ",\"frame_width\":" << r.frame_width
    << ",\"frame_height\":" << r.frame_height
    << ",\"hands\":[";
  for (size_t h = 0; h < r.hands.size(); ++h) {
    if (h > 0) o << ',';
    const auto& hand = r.hands[h];
    o << "{\"hand_id\":" << hand.hand_id;
    if (hand.roi.has_value()) {
      o << ",\"roi\":[" << hand.roi->x1 << ',' << hand.roi->y1
        << ',' << hand.roi->x2 << ',' << hand.roi->y2 << ']';
    }
    if (!hand.landmarks.empty()) {
      o << ",\"landmarks\":[";
      for (size_t i = 0; i < hand.landmarks.size(); ++i) {
        if (i > 0) o << ',';
        o << '[' << hand.landmarks[i].x << ',' << hand.landmarks[i].y
          << ',' << hand.landmarks[i].z << ']';
      }
      o << ']';
    }
    const char* mode_str = "no_hand";
    switch (hand.tracking_mode) {
      case rkstudio::ai::TrackingMode::kDetect: mode_str = "detect"; break;
      case rkstudio::ai::TrackingMode::kTrack: mode_str = "track"; break;
      case rkstudio::ai::TrackingMode::kRecover: mode_str = "recover"; break;
      default: break;
    }
    o << ",\"tracking_mode\":\"" << mode_str
      << "\",\"motion_norm\":" << hand.motion_norm << '}';
  }
  o << "],\"fps\":" << r.fps
    << ",\"ok\":" << (r.ok ? "true" : "false");
  if (!r.error.empty()) {
    o << ",\"error\":\"" << rkinfra::JsonEscape(r.error) << '"';
  }
  o << '}';
  return o.str();
}

}  // namespace

MediaEngine::MediaEngine(QObject* parent) : QObject(parent) {
  qRegisterMetaType<rkstudio::AppState>();
  qRegisterMetaType<rkstudio::TelemetryEvent>();
  qRegisterMetaType<rkstudio::ai::AiResult>();

  ai_poll_timer_ = new QTimer(this);
  ai_poll_timer_->setInterval(30);
  connect(ai_poll_timer_, &QTimer::timeout, this, &MediaEngine::PollAiResults);
}

MediaEngine::~MediaEngine() {
  StopAll();
}

void MediaEngine::LoadBoardConfig(const BoardConfig& board_config) {
  board_config_ = board_config;
}

void MediaEngine::ApplySessionProfile(const SessionProfile& profile) {
  session_profile_ = profile;
}

bool MediaEngine::StartPreview(std::string* err) {
  if (state_ != AppState::kIdle) {
    if (err != nullptr) {
      *err = "cannot start preview in current state";
    }
    return false;
  }
  if (board_config_.cameras.empty()) {
    if (err != nullptr) {
      *err = "board config is empty";
    }
    return false;
  }
  if (session_profile_.preview_cameras.empty()) {
    if (err != nullptr) {
      *err = "session profile preview_cameras is empty";
    }
    return false;
  }

  StartAiProcessor();

  if (!RebuildPipelines(false, err)) {
    StopAiProcessor();
    state_ = AppState::kError;
    emit StateChanged(state_);
    return false;
  }

  if (ai_processor_) {
    ai_poll_timer_->start();
  }

  state_ = AppState::kPreviewing;
  emit StateChanged(state_);
  return true;
}

bool MediaEngine::StartRecording(std::string* err) {
  if (state_ != AppState::kIdle && state_ != AppState::kPreviewing) {
    if (err != nullptr) {
      *err = "cannot start recording in current state";
    }
    return false;
  }

  // If previewing, stop pipelines first (modes are mutually exclusive).
  if (state_ == AppState::kPreviewing) {
    ai_poll_timer_->stop();
    StopPipelines();
  }

  if (!ai_processor_) {
    StartAiProcessor();
  }

  session_writer_ = std::make_unique<SessionWriter>();
  if (!session_writer_->Initialize(board_config_, session_profile_, err)) {
    session_writer_.reset();
    return false;
  }

  if (ai_processor_ && session_writer_->session_paths()) {
    session_writer_->OpenAiWriter(nullptr);
  }

  if (!RebuildPipelines(true, err)) {
    FinalizeRecording(false);
    return false;
  }

  const auto* config = session_writer_->compat_config();
  if (config && config->audio.has_value()) {
    audio_recorder_ = std::make_unique<rkinfra::GstAudioRecorder>(
        *config->audio, config->queue.audio_mux_max_time_ns,
        [this](rkinfra::StreamEvent event) {
          event.category = "audio";
          EmitTelemetry(event);
        },
        session_writer_->session_paths()->session_dir);
    std::string audio_err;
    if (!audio_recorder_->Build(&audio_err) || !audio_recorder_->Start(&audio_err)) {
      if (err != nullptr) {
        *err = audio_err;
      }
      FinalizeRecording(false);
      return false;
    }
  } else {
    audio_recorder_.reset();
  }

  const auto outputs = CollectOutputs(cameras_, audio_recorder_.get());
  session_writer_->WriteStartMeta(outputs);

  if (ai_processor_) {
    ai_poll_timer_->start();
  }

  state_ = AppState::kRecording;
  emit StateChanged(state_);
  return true;
}

void MediaEngine::StopRecording() {
  if (state_ != AppState::kRecording) {
    return;
  }
  FinalizeRecording(true);
}

bool MediaEngine::StartRtsp(std::string* err) {
  if (state_ != AppState::kIdle && state_ != AppState::kPreviewing) {
    if (err) *err = "cannot start RTSP in current state";
    return false;
  }

  if (state_ == AppState::kPreviewing) {
    StopAiProcessor();
    ai_poll_timer_->stop();
    StopPipelines();
  }

  rtsp_server_ = std::make_unique<RtspServer>();
  if (!rtsp_server_->Start(board_config_, session_profile_, err)) {
    rtsp_server_.reset();
    state_ = AppState::kError;
    emit StateChanged(state_);
    return false;
  }

  state_ = AppState::kStreaming;
  emit StateChanged(state_);
  return true;
}

void MediaEngine::StopRtsp() {
  if (state_ != AppState::kStreaming) {
    return;
  }
  if (rtsp_server_) {
    rtsp_server_->Stop();
    rtsp_server_.reset();
  }
  state_ = AppState::kIdle;
  emit StateChanged(state_);
}

void MediaEngine::StopAll() {
  ai_poll_timer_->stop();
  if (state_ == AppState::kRecording) {
    FinalizeRecording(true);
    return;
  }
  if (state_ == AppState::kStreaming) {
    StopRtsp();
    return;
  }
  StopAiProcessor();
  StopPipelines();
  state_ = AppState::kIdle;
  emit StateChanged(state_);
}

void MediaEngine::BindPreviewWindow(const std::string& camera_id, WId window_id) {
  preview_window_ids_[camera_id] = window_id;
  auto it = cameras_.find(camera_id);
  if (it != cameras_.end()) {
    it->second->SetPreviewWindow(window_id);
  }
}

const BoardConfig& MediaEngine::board_config() const {
  return board_config_;
}

const SessionProfile& MediaEngine::session_profile() const {
  return session_profile_;
}

AppState MediaEngine::state() const {
  return state_;
}

bool MediaEngine::RebuildPipelines(bool recording, std::string* err) {
  StopPipelines();

  const std::vector<std::string> camera_ids = recording ? UnionCameraIds(session_profile_) : session_profile_.preview_cameras;
  for (const auto& camera_id : camera_ids) {
    const CameraNodeSet* camera = FindCamera(board_config_, camera_id);
    if (camera == nullptr) {
      if (err != nullptr) {
        *err = "unknown camera id: " + camera_id;
      }
      StopPipelines();
      return false;
    }

    auto pipeline = std::make_unique<CameraPipeline>();
    CameraPipeline::BuildOptions options;
    options.camera = *camera;
    options.sink_priority = board_config_.sink_priority;
    options.session_dir = (session_writer_ && session_writer_->session_paths())
        ? session_writer_->session_paths()->session_dir
        : std::filesystem::path(session_profile_.output_dir);
    if (const auto it = preview_window_ids_.find(camera_id); it != preview_window_ids_.end()) {
      options.preview_window_id = it->second;
    }
    options.enable_preview = !recording
                             && preview_window_ids_.count(camera_id) > 0
                             && Contains(session_profile_.preview_cameras, camera_id)
                             && camera_id != ai_camera_id_;
    options.enable_record = recording && Contains(session_profile_.record_cameras, camera_id);
    options.gop = session_profile_.gop;

    if (camera_id == ai_camera_id_ && ai_processor_) {
      options.enable_ai = true;
      options.ai_sample_callback = [this](GstSample* sample) { OnAiSample(sample); };
    }

    if (!pipeline->Build(
            options, [this](const TelemetryEvent& event) { EmitTelemetry(event); },
            [this, camera_id](const std::string& reason, bool fatal) {
              QMetaObject::invokeMethod(
                  this,
                  [this, camera_id, reason, fatal] { OnCameraError(camera_id, reason, fatal); },
                  Qt::QueuedConnection);
            },
            err) ||
        !pipeline->Start(err)) {
      StopPipelines();
      return false;
    }

    cameras_.insert_or_assign(camera_id, std::move(pipeline));
  }

  return true;
}

void MediaEngine::StopPipelines() {
  for (auto& [camera_id, pipeline] : cameras_) {
    (void)camera_id;
    pipeline->Stop();
  }
  cameras_.clear();
}

void MediaEngine::EmitTelemetry(const TelemetryEvent& event) {
  if (session_writer_) {
    session_writer_->WriteEvent(event);

    const bool record_sync_event =
        event.category == "audio" ||
        (event.category == "media" && Contains(session_profile_.record_cameras, event.stream_id) &&
         (event.stage == "capture" || event.stage == "queue"));
    if (record_sync_event) {
      session_writer_->RecordSyncEvent(event);
    }
  }
  emit TelemetryObserved(event);
}

void MediaEngine::OnCameraError(const std::string& camera_id, const std::string& reason, bool fatal) {
  emit PreviewCameraFailed(QString::fromStdString(camera_id), QString::fromStdString(reason), fatal);
  if (fatal && state_ == AppState::kRecording) {
    FinalizeRecording(false);
    state_ = AppState::kError;
    emit StateChanged(state_);
  }
}

void MediaEngine::FinalizeRecording(bool ok) {
  ai_poll_timer_->stop();
  const auto outputs = CollectOutputs(cameras_, audio_recorder_.get());

  if (audio_recorder_) {
    audio_recorder_->RequestStop();
  }
  StopPipelines();
  if (audio_recorder_) {
    audio_recorder_->Stop();
  }
  audio_recorder_.reset();

  if (session_writer_) {
    session_writer_->Finalize(ok, outputs);
    session_writer_.reset();
  }

  StopAiProcessor();

  state_ = ok ? AppState::kIdle : AppState::kError;
  emit StateChanged(state_);
}

void MediaEngine::SetAiEnabled(bool enabled) {
  ai_enabled_ = enabled;
}

bool MediaEngine::StartAiProcessor() {
  ai_camera_id_.clear();
  const std::string& ai_cam = session_profile_.selected_ai_camera;
  if (!ai_enabled_ || ai_cam.empty() || !board_config_.ai.has_value()) {
    return false;
  }

  const auto& ai_hw = *board_config_.ai;

  ai::AiProcessorConfig config;
  config.detector_model = ai_hw.detector_model;
  config.landmark_model = ai_hw.landmark_model;
  config.queue_depth = 1;

  ai_processor_ = ai::CreateHandAiProcessor();
  std::string ai_err;
  if (!ai_processor_->Start(config, &ai_err)) {
    std::cerr << "[ai] failed to start processor: " << ai_err << "\n";
    ai_processor_.reset();
    return false;
  }

  ai_camera_id_ = ai_cam;
  std::cerr << "[ai] processor started for " << ai_camera_id_ << "\n";
  return true;
}

void MediaEngine::StopAiProcessor() {
  ai_poll_timer_->stop();
  if (ai_processor_) {
    ai_processor_->Stop();
    ai_processor_.reset();
    std::cerr << "[ai] processor stopped\n";
  }
  {
    std::lock_guard<std::mutex> lock(ai_frame_mu_);
    ai_camera_id_.clear();
  }
}

void MediaEngine::OnAiSample(GstSample* sample) {
  if (!ai_processor_ || !sample) {
    return;
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);
  if (!buffer || !caps) {
    return;
  }

  GstVideoInfo info;
  if (!gst_video_info_from_caps(&info, caps)) {
    return;
  }

  const int w = GST_VIDEO_INFO_WIDTH(&info);
  const int h = GST_VIDEO_INFO_HEIGHT(&info);
  const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
  const uint64_t pts_ns = GST_CLOCK_TIME_IS_VALID(GST_BUFFER_PTS(buffer))
                              ? GST_BUFFER_PTS(buffer) : 0;

  // Try RGA hardware NV12→RGB conversion (dmabuf path).
  cv::Mat rgb;
  bool rga_ok = false;
  {
    const int fd = ExtractDmabufFd(buffer);
    if (fd >= 0) {
      rga_ok = mediapipe_demo::ConvertNv12ToRgb(fd, w, h, stride, &rgb);
    }
  }

  // Log once which path is active.
  static bool logged_path = false;
  if (!logged_path) {
    std::cerr << "[ai] NV12→RGB path: " << (rga_ok ? "RGA hardware" : "CPU fallback") << "\n";
    logged_path = true;
  }

  // Fallback: CPU NV12→RGB.
  if (!rga_ok) {
    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      return;
    }
    if (w > 0 && h > 0 && map.size >= static_cast<size_t>(stride) * h * 3 / 2) {
      cv::Mat nv12(h * 3 / 2, w, CV_8UC1, map.data, stride);
      cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    }
    gst_buffer_unmap(buffer, &map);
  }

  if (rgb.empty()) {
    return;
  }

  // Submit RGB to AI processor — shared_ptr keeps the Mat alive.
  auto rgb_holder = std::make_shared<cv::Mat>(std::move(rgb));
  ai::FrameRef frame;
  {
    std::lock_guard<std::mutex> lock(ai_frame_mu_);
    frame.camera_id = ai_camera_id_;
  }
  frame.pts_ns = pts_ns;
  frame.width = w;
  frame.height = h;
  frame.stride = static_cast<int>(rgb_holder->step[0]);
  frame.pixel_format = ai::PixelFormat::kRgb;
  frame.mapped_ptr = rgb_holder->data;
  frame.bytes_used = rgb_holder->total() * rgb_holder->elemSize();
  frame.owned_data = rgb_holder;
  ai_processor_->Submit(frame);

  // QImage from the same RGB data (Format_RGB888 matches directly).
  QImage image(rgb_holder->data, w, h,
               static_cast<int>(rgb_holder->step[0]),
               QImage::Format_RGB888);
  {
    std::lock_guard<std::mutex> lock(ai_frame_mu_);
    latest_ai_frame_ = image.copy();
  }
}

void MediaEngine::PollAiResults() {
  if (!ai_processor_) {
    return;
  }

  while (auto result = ai_processor_->PollResult()) {
    if (session_writer_) {
      session_writer_->WriteAiLine(AiResultToJson(*result));
    }

    // Throttle AI frame emission to ~20fps to avoid flooding the UI event queue.
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(ai_frame_mu_);
      if (!latest_ai_frame_.isNull() &&
          now - last_ai_frame_emit_ >= std::chrono::milliseconds(50)) {
        last_ai_frame_emit_ = now;
        emit AiFrameReady(QString::fromStdString(ai_camera_id_), latest_ai_frame_);
      }
    }
    emit AiResultReady(*result);
  }
}

}  // namespace rkstudio::media
