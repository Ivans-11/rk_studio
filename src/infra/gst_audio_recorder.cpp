#include "rk_studio/infra/gst_audio_recorder.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <utility>

#include "rk_studio/infra/runtime.h"

namespace rkinfra {
namespace {

constexpr guint kQueueLeakyDownstream = 2;
constexpr auto kBusStopWait = std::chrono::seconds(5);

template <typename T>
struct GstUnrefDeleter {
  void operator()(T* ptr) const {
    if (ptr) {
      gst_object_unref(ptr);
    }
  }
};

template <typename T>
void SetPropertyIfExists(GstElement* element, const char* property, const T& value) {
  if (!element) {
    return;
  }
  if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), property) != nullptr) {
    g_object_set(G_OBJECT(element), property, value, nullptr);
  }
}

}  // namespace

GstAudioRecorder::GstAudioRecorder(AudioConfig config,
                                   uint64_t queue_mux_max_time_ns,
                                   EventCallback on_event,
                                   const fs::path& session_dir)
    : config_(std::move(config)),
      queue_mux_max_time_ns_(queue_mux_max_time_ns),
      on_event_(std::move(on_event)),
      session_dir_(session_dir) {}

GstAudioRecorder::~GstAudioRecorder() { Stop(); }

bool GstAudioRecorder::Build(std::string* err) {
  output_path_ = session_dir_ / (config_.id + ".wav");
  output_.id = config_.id;
  output_.type = "audio";
  output_.device = config_.device;
  output_.codec = "pcm_s16le";
  output_.output_path = output_path_.string();
  if (err) {
    err->clear();
  }
  return true;
}

bool GstAudioRecorder::Start(std::string* err) {
  stop_requested_.store(false);
  failed_.store(false);
  force_stop_bus_watch_.store(false);
  bus_thread_done_.store(false);
  base_time_ns_.store(GST_CLOCK_TIME_NONE);
  seq_.store(0);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    failure_reason_.clear();
  }

  CleanupPipeline();
  if (!BuildPipeline(err)) {
    CleanupPipeline();
    return false;
  }
  if (!StartPipeline(err)) {
    CleanupPipeline();
    return false;
  }

  bus_thread_ = std::thread([this] { BusLoop(); });
  return true;
}

void GstAudioRecorder::RequestStop() { stop_requested_.store(true); }

void GstAudioRecorder::Stop() {
  stop_requested_.store(true);

  if (pipeline_) {
    gst_element_send_event(pipeline_, gst_event_new_eos());
  }

  const auto deadline = std::chrono::steady_clock::now() + kBusStopWait;
  while (bus_thread_.joinable() && !bus_thread_done_.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (bus_thread_.joinable() && !bus_thread_done_.load()) {
    force_stop_bus_watch_.store(true);
  }
  if (bus_thread_.joinable()) {
    bus_thread_.join();
  }

  CleanupPipeline();
}

bool GstAudioRecorder::failed() const { return failed_.load(); }

const std::atomic<bool>& GstAudioRecorder::failure_flag() const { return failed_; }

std::string GstAudioRecorder::failure_reason() const {
  std::lock_guard<std::mutex> lock(state_mu_);
  return failure_reason_;
}

const OutputStreamInfo& GstAudioRecorder::stream_output() const { return output_; }

bool GstAudioRecorder::BuildPipeline(std::string* err) {
  pipeline_ = gst_pipeline_new("rk-recorder-audio");
  source_ = gst_element_factory_make("alsasrc", "audio_src");
  caps_filter_ = gst_element_factory_make("capsfilter", "audio_caps");
  queue_mux_ = gst_element_factory_make("queue", "audio_queue_mux");
  wavenc_ = gst_element_factory_make("wavenc", "audio_wavenc");
  sink_ = gst_element_factory_make("filesink", "audio_sink");
  if (!pipeline_ || !source_ || !caps_filter_ || !queue_mux_ || !wavenc_ || !sink_) {
    if (err) {
      *err = "failed to create GStreamer audio elements";
    }
    return false;
  }

  SetPropertyIfExists(source_, "device", config_.device.c_str());
  SetPropertyIfExists(source_, "do-timestamp", FALSE);
  SetPropertyIfExists(source_, "use-driver-timestamps", TRUE);

  GstCaps* audio_caps = gst_caps_new_simple("audio/x-raw", "format", G_TYPE_STRING, "S16LE", "layout", G_TYPE_STRING,
                                            "interleaved", "rate", G_TYPE_INT, config_.rate, "channels", G_TYPE_INT,
                                            config_.channels, nullptr);
  g_object_set(G_OBJECT(caps_filter_), "caps", audio_caps, nullptr);
  gst_caps_unref(audio_caps);

  g_object_set(G_OBJECT(queue_mux_), "leaky", kQueueLeakyDownstream, "max-size-buffers", 0u, "max-size-bytes", 0u,
               "max-size-time", queue_mux_max_time_ns_, nullptr);
  SetPropertyIfExists(sink_, "location", output_path_.c_str());

  gst_bin_add_many(GST_BIN(pipeline_), source_, caps_filter_, queue_mux_, wavenc_, sink_, nullptr);
  if (!gst_element_link_many(source_, caps_filter_, queue_mux_, wavenc_, sink_, nullptr)) {
    if (err) {
      *err = "failed to link GStreamer audio pipeline";
    }
    return false;
  }

  InstallProbes();
  InstallQueueSignals();
  return true;
}

bool GstAudioRecorder::StartPipeline(std::string* err) {
  using GstClockPtr = std::unique_ptr<GstClock, GstUnrefDeleter<GstClock>>;

  GstClockPtr clock(gst_system_clock_obtain());
  if (!clock) {
    if (err) {
      *err = "failed to obtain system clock for audio pipeline";
    }
    return false;
  }

  g_object_set(clock.get(), "clock-type", GST_CLOCK_TYPE_MONOTONIC, nullptr);
  gst_pipeline_use_clock(GST_PIPELINE(pipeline_), clock.get());

  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  const GstStateChangeReturn state_change = gst_element_get_state(pipeline_, nullptr, nullptr, 5 * GST_SECOND);
  if (state_change == GST_STATE_CHANGE_FAILURE) {
    if (err) {
      *err = "audio pipeline failed to enter PLAYING state";
    }
    return false;
  }

  base_time_ns_.store(gst_element_get_base_time(pipeline_));
  return true;
}

void GstAudioRecorder::CleanupPipeline() {
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }

  pipeline_ = nullptr;
  source_ = nullptr;
  caps_filter_ = nullptr;
  queue_mux_ = nullptr;
  wavenc_ = nullptr;
  sink_ = nullptr;
  base_time_ns_.store(GST_CLOCK_TIME_NONE);
  probe_contexts_.clear();
  queue_contexts_.clear();
}

void GstAudioRecorder::InstallProbes() {
  using GstPadPtr = std::unique_ptr<GstPad, GstUnrefDeleter<GstPad>>;

  GstPadPtr pad(gst_element_get_static_pad(queue_mux_, "src"));
  if (!pad) {
    return;
  }

  auto ctx = std::make_unique<ProbeContext>();
  ctx->self = this;
  ctx->stream_id = config_.id;
  ctx->seq = &seq_;
  gst_pad_add_probe(pad.get(), GST_PAD_PROBE_TYPE_BUFFER, &GstAudioRecorder::OnBufferProbe, ctx.get(), nullptr);
  probe_contexts_.push_back(std::move(ctx));
}

void GstAudioRecorder::InstallQueueSignals() {
  auto ctx = std::make_unique<QueueSignalContext>();
  ctx->self = this;
  ctx->stream_id = config_.id;
  ctx->queue_name = "audio_queue_mux";
  g_signal_connect(queue_mux_, "overrun", G_CALLBACK(&GstAudioRecorder::OnQueueOverrun), ctx.get());
  queue_contexts_.push_back(std::move(ctx));
}

void GstAudioRecorder::BusLoop() {
  using GstBusPtr = std::unique_ptr<GstBus, GstUnrefDeleter<GstBus>>;

  GstBusPtr bus(pipeline_ ? gst_element_get_bus(pipeline_) : nullptr);
  if (!bus) {
    Fail("failed to get audio pipeline bus");
    bus_thread_done_.store(true);
    return;
  }

  while (!force_stop_bus_watch_.load()) {
    GstMessage* message = gst_bus_timed_pop_filtered(
        bus.get(), 200 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!message) {
      continue;
    }

    switch (GST_MESSAGE_TYPE(message)) {
      case GST_MESSAGE_ERROR: {
        GError* error = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &error, &debug);
        const std::string reason =
            "audio_gstreamer_error:" + std::string(error ? error->message : "unknown");
        std::cerr << "audio pipeline error: " << (error ? error->message : "unknown") << "\n";
        if (debug) {
          std::cerr << "audio debug: " << debug << "\n";
        }
        g_clear_error(&error);
        g_free(debug);
        gst_message_unref(message);
        Fail(reason);
        bus_thread_done_.store(true);
        return;
      }
      case GST_MESSAGE_EOS:
        gst_message_unref(message);
        if (!stop_requested_.load()) {
          Fail("audio_unexpected_eos");
        }
        bus_thread_done_.store(true);
        return;
      default:
        gst_message_unref(message);
        break;
    }
  }

  bus_thread_done_.store(true);
}

uint64_t GstAudioRecorder::ComputeMonotonicNsFromPts(GstClockTime pts) const {
  const GstClockTime base = base_time_ns_.load();
  if (GST_CLOCK_TIME_IS_VALID(base) && GST_CLOCK_TIME_IS_VALID(pts)) {
    return static_cast<uint64_t>(base + pts);
  }

  if (pipeline_) {
    using GstClockPtr = std::unique_ptr<GstClock, GstUnrefDeleter<GstClock>>;
    GstClockPtr clock(gst_pipeline_get_clock(GST_PIPELINE(pipeline_)));
    if (clock) {
      return static_cast<uint64_t>(gst_clock_get_time(clock.get()));
    }
  }

  return ClockMonotonicNs();
}

void GstAudioRecorder::Fail(const std::string& reason) {
  const bool already_failed = failed_.exchange(true);
  stop_requested_.store(true);
  {
    std::lock_guard<std::mutex> lock(state_mu_);
    if (failure_reason_.empty()) {
      failure_reason_ = reason;
    }
  }

  if (!already_failed) {
    StreamEvent event;
    event.monotonic_ns = ClockMonotonicNs();
    event.stream_id = config_.id;
    event.seq = seq_.fetch_add(1) + 1;
    event.pts_ns = -1;
    event.status = "missing";
    event.reason = reason;
    event.stage = "capture";
    PushEvent(std::move(event));
  }
}

void GstAudioRecorder::PushEvent(StreamEvent event) {
  if (on_event_) {
    on_event_(event);
  }
}

GstPadProbeReturn GstAudioRecorder::OnBufferProbe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* ctx = static_cast<ProbeContext*>(user_data);
  if (!ctx || !ctx->self || !(info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
    return GST_PAD_PROBE_OK;
  }

  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buffer) {
    return GST_PAD_PROBE_OK;
  }

  StreamEvent event;
  event.monotonic_ns = ctx->self->ComputeMonotonicNsFromPts(GST_BUFFER_PTS(buffer));
  event.stream_id = ctx->stream_id;
  event.seq = ctx->seq->fetch_add(1) + 1;
  event.stage = "capture";

  const GstClockTime pts = GST_BUFFER_PTS(buffer);
  if (!GST_CLOCK_TIME_IS_VALID(pts)) {
    event.pts_ns = -1;
    event.status = "missing";
    event.reason = "invalid_pts";
  } else {
    event.pts_ns = static_cast<int64_t>(pts);
    event.status = "ok";
  }

  ctx->self->PushEvent(std::move(event));
  return GST_PAD_PROBE_OK;
}

void GstAudioRecorder::OnQueueOverrun(GstElement*, gpointer user_data) {
  auto* ctx = static_cast<QueueSignalContext*>(user_data);
  if (!ctx || !ctx->self) {
    return;
  }

  StreamEvent event;
  event.monotonic_ns = ctx->self->ComputeMonotonicNsFromPts(GST_CLOCK_TIME_NONE);
  event.stream_id = ctx->stream_id;
  event.seq = 0;
  event.pts_ns = -1;
  event.status = "missing";
  event.reason = "queue_overrun:" + ctx->queue_name;
  event.stage = "queue";
  ctx->self->PushEvent(std::move(event));
}

}  // namespace rkinfra
