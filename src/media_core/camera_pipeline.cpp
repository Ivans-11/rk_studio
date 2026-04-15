#include "rk_studio/media_core/camera_pipeline.h"

#include <chrono>
#include <memory>

#include <gst/video/video-event.h>
#include <gst/video/video.h>
#include <gst/video/videooverlay.h>

#include "rk_studio/infra/runtime.h"
#include "rk_studio/infra/gst_util.h"

namespace rkstudio::media {
namespace {

constexpr guint kQueueLeakyDownstream = 2;

template <typename T>
struct GstUnrefDeleter {
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      gst_object_unref(ptr);
    }
  }
};

GstCaps* MakeSourceCaps(const CameraNodeSet& camera, int width, int height) {
  if (rkinfra::IsJpegLikeFormat(camera.input_format)) {
    return gst_caps_new_simple("image/jpeg", "width", G_TYPE_INT, width, "height", G_TYPE_INT, height, "framerate",
                               GST_TYPE_FRACTION, camera.fps, 1, nullptr);
  }
  return gst_caps_new_simple("video/x-raw", "width", G_TYPE_INT, width, "height", G_TYPE_INT, height, "format",
                             G_TYPE_STRING, camera.input_format.c_str(), "framerate", GST_TYPE_FRACTION, camera.fps, 1,
                             nullptr);
}

GstCaps* MakeNormalizedCaps(const CameraNodeSet& camera, int width, int height) {
  return gst_caps_new_simple("video/x-raw", "width", G_TYPE_INT, width, "height", G_TYPE_INT, height, "format",
                             G_TYPE_STRING, "NV12", "framerate", GST_TYPE_FRACTION, camera.fps, 1, nullptr);
}

}  // namespace

CameraPipeline::CameraPipeline() = default;

CameraPipeline::~CameraPipeline() {
  Stop();
}

bool CameraPipeline::Build(const BuildOptions& options,
                          TelemetryCallback telemetry_callback,
                          ErrorCallback error_callback,
                          std::string* err) {
  Stop();
  options_ = options;
  telemetry_callback_ = std::move(telemetry_callback);
  error_callback_ = std::move(error_callback);

  resolved_device_ = options_.camera.record_device;

  if (options_.enable_record) {
    record_output_path_ = (options_.session_dir / (options_.camera.id + ".mkv")).string();
  } else {
    record_output_path_.clear();
  }

  return BuildPipeline(err);
}

bool CameraPipeline::BuildPipeline(std::string* err) {
  const bool jpeg_input = rkinfra::IsJpegLikeFormat(options_.camera.input_format);
  const bool nv12_input = rkinfra::IsNv12Format(options_.camera.input_format);
  const bool needs_normalize = jpeg_input || !nv12_input;

  pipeline_ = gst_pipeline_new(("rk-studio-" + options_.camera.id).c_str());
  source_ = gst_element_factory_make("v4l2src", ("source_" + options_.camera.id).c_str());
  source_caps_ = gst_element_factory_make("capsfilter", ("source_caps_" + options_.camera.id).c_str());

  if (needs_normalize) {
    if (jpeg_input) {
      jpeg_decoder_ = gst_element_factory_make("jpegdec", ("jpegdec_" + options_.camera.id).c_str());
    }
    normalize_convert_ = gst_element_factory_make("videoconvert", ("normalize_convert_" + options_.camera.id).c_str());
    normalize_caps_ = gst_element_factory_make("capsfilter", ("normalize_caps_" + options_.camera.id).c_str());
  }

  if (options_.enable_preview) {
    preview_convert_ = gst_element_factory_make("videoconvert", ("preview_convert_" + options_.camera.id).c_str());
    if (!CreatePreviewSink(err)) {
      return false;
    }
  }

  if (options_.enable_ai) {
    ai_queue_ = gst_element_factory_make("queue", ("ai_queue_" + options_.camera.id).c_str());
    ai_sink_ = gst_element_factory_make("appsink", ("ai_sink_" + options_.camera.id).c_str());
  }

  if (options_.enable_record) {
    encoder_ = gst_element_factory_make("mpph265enc", ("encoder_" + options_.camera.id).c_str());
    parser_ = gst_element_factory_make("h265parse", ("parser_" + options_.camera.id).c_str());
    mux_ = gst_element_factory_make("matroskamux", ("mux_" + options_.camera.id).c_str());
    record_sink_ = gst_element_factory_make("filesink", ("record_sink_" + options_.camera.id).c_str());
  }

  // tee is needed when more than one output branch is active.
  const int branch_count = (options_.enable_preview ? 1 : 0) +
                           (options_.enable_record ? 1 : 0) +
                           (options_.enable_ai ? 1 : 0);
  const bool needs_tee = branch_count > 1;
  if (needs_tee) {
    tee_ = gst_element_factory_make("tee", ("tee_" + options_.camera.id).c_str());
    if (options_.enable_record) {
      record_queue_ = gst_element_factory_make("queue", ("record_queue_" + options_.camera.id).c_str());
    }
    if (options_.enable_preview) {
      preview_queue_ = gst_element_factory_make("queue", ("preview_queue_" + options_.camera.id).c_str());
    }
  }

  if (!pipeline_ || !source_ || !source_caps_ ||
      (needs_normalize && !normalize_convert_) ||
      (needs_normalize && !normalize_caps_) || (jpeg_input && !jpeg_decoder_) ||
      (options_.enable_preview && (!preview_convert_ || !preview_sink_)) ||
      (options_.enable_record && (!encoder_ || !parser_ || !mux_ || !record_sink_)) ||
      (options_.enable_ai && (!ai_queue_ || !ai_sink_)) ||
      (needs_tee && !tee_) ||
      (needs_tee && options_.enable_record && !record_queue_) ||
      (needs_tee && options_.enable_preview && !preview_queue_)) {
    if (err != nullptr) {
      *err = "failed to create camera pipeline elements for " + options_.camera.id;
    }
    return false;
  }

  rkinfra::SetPropertyIfExists(source_, "device", resolved_device_.c_str());
  rkinfra::SetPropertyIfExists(source_, "do-timestamp", TRUE);
  std::string io_mode_error;
  const int io_mode = rkinfra::ToV4l2IoMode(options_.camera.io_mode, &io_mode_error);
  if (io_mode < 0) {
    if (err != nullptr) {
      *err = io_mode_error;
    }
    return false;
  }
  rkinfra::SetPropertyIfExists(source_, "io-mode", io_mode);

  const int source_width = options_.enable_record ? options_.camera.record_width : options_.camera.preview_width;
  const int source_height = options_.enable_record ? options_.camera.record_height : options_.camera.preview_height;

  GstCaps* source_caps = MakeSourceCaps(options_.camera, source_width, source_height);
  g_object_set(G_OBJECT(source_caps_), "caps", source_caps, nullptr);
  gst_caps_unref(source_caps);

  if (needs_normalize) {
    GstCaps* normalize_caps = MakeNormalizedCaps(options_.camera, source_width, source_height);
    g_object_set(G_OBJECT(normalize_caps_), "caps", normalize_caps, nullptr);
    gst_caps_unref(normalize_caps);
  }

  if (options_.enable_preview) {
    rkinfra::SetPropertyIfExists(preview_sink_, "sync", FALSE);
  }

  if (options_.enable_ai) {
    g_object_set(G_OBJECT(ai_sink_), "emit-signals", FALSE, "max-buffers", 2u, "drop", TRUE, "sync", FALSE, nullptr);
    g_object_set(G_OBJECT(ai_queue_), "leaky", kQueueLeakyDownstream, "max-size-buffers", 2u,
                 "max-size-bytes", 0u, "max-size-time", static_cast<guint64>(0), nullptr);
    GstAppSinkCallbacks callbacks{};
    callbacks.new_sample = &CameraPipeline::OnAiSample;
    gst_app_sink_set_callbacks(GST_APP_SINK(ai_sink_), &callbacks, this, nullptr);
  }

  if (needs_tee) {
    if (record_queue_) {
      g_object_set(G_OBJECT(record_queue_), "leaky", kQueueLeakyDownstream, "max-size-buffers", 0u,
                   "max-size-bytes", 0u, "max-size-time", static_cast<guint64>(2'000'000'000ULL), nullptr);
    }
    if (preview_queue_) {
      g_object_set(G_OBJECT(preview_queue_), "leaky", kQueueLeakyDownstream, "max-size-buffers", 2u,
                   "max-size-bytes", 0u, "max-size-time", static_cast<guint64>(0), nullptr);
    }
  }

  if (options_.enable_record) {
    rkinfra::SetPropertyIfExists(encoder_, "bps", static_cast<guint>(options_.camera.bitrate));
    rkinfra::SetPropertyIfExists(encoder_, "gop", options_.gop);
    rkinfra::SetPropertyIfExists(encoder_, "header-mode", 1);
    rkinfra::SetPropertyIfExists(record_sink_, "location", record_output_path_.c_str());
  }

  // Add elements to pipeline.
  gst_bin_add_many(GST_BIN(pipeline_), source_, source_caps_, nullptr);
  if (jpeg_decoder_ != nullptr) {
    gst_bin_add(GST_BIN(pipeline_), jpeg_decoder_);
  }
  if (normalize_convert_ != nullptr) {
    gst_bin_add_many(GST_BIN(pipeline_), normalize_convert_, normalize_caps_, nullptr);
  }
  if (options_.enable_preview) {
    gst_bin_add_many(GST_BIN(pipeline_), preview_convert_, preview_sink_, nullptr);
    if (preview_queue_) {
      gst_bin_add(GST_BIN(pipeline_), preview_queue_);
    }
  }
  if (options_.enable_ai) {
    gst_bin_add_many(GST_BIN(pipeline_), ai_queue_, ai_sink_, nullptr);
  }
  if (needs_tee) {
    gst_bin_add(GST_BIN(pipeline_), tee_);
    if (record_queue_) {
      gst_bin_add(GST_BIN(pipeline_), record_queue_);
    }
  }
  if (options_.enable_record) {
    gst_bin_add_many(GST_BIN(pipeline_), encoder_, parser_, mux_, record_sink_, nullptr);
  }

  // Link source chain: v4l2src → capsfilter → [normalize]
  // source_tail tracks the last element in the source chain for linking the output branch.
  GstElement* source_tail = nullptr;
  bool link_ok = false;
  if (jpeg_input) {
    link_ok = gst_element_link_many(source_, source_caps_, jpeg_decoder_, normalize_convert_, normalize_caps_, nullptr);
    source_tail = normalize_caps_;
  } else if (needs_normalize) {
    link_ok = gst_element_link_many(source_, source_caps_, normalize_convert_, normalize_caps_, nullptr);
    source_tail = normalize_caps_;
  } else {
    link_ok = gst_element_link(source_, source_caps_);
    source_tail = source_caps_;
  }
  if (!link_ok) {
    if (err != nullptr) {
      *err = "failed to link source chain for " + options_.camera.id;
    }
    return false;
  }

  // --- Link output branches ---

  auto LinkTeeBranch = [&](GstElement* sink_element, const char* label) -> bool {
    GstPad* tee_src = gst_element_get_request_pad(tee_, "src_%u");
    GstPad* branch_sink = gst_element_get_static_pad(sink_element, "sink");
    const bool ok = tee_src && branch_sink &&
                    gst_pad_link(tee_src, branch_sink) == GST_PAD_LINK_OK;
    if (branch_sink) gst_object_unref(branch_sink);
    if (tee_src) gst_object_unref(tee_src);
    if (!ok && err != nullptr) {
      *err = std::string("failed to link tee → ") + label + " for " + options_.camera.id;
    }
    return ok;
  };

  auto LinkRecordChain = [&]() -> bool {
    if (record_queue_ && !gst_element_link(record_queue_, encoder_)) {
      if (err) *err = "failed to link record_queue → encoder for " + options_.camera.id;
      return false;
    }
    if (!gst_element_link_many(encoder_, parser_, nullptr) || !gst_element_link(mux_, record_sink_)) {
      if (err) *err = "failed to link record chain for " + options_.camera.id;
      return false;
    }
    GstPad* mux_pad = gst_element_get_request_pad(mux_, "video_%u");
    GstPad* parser_src = gst_element_get_static_pad(parser_, "src");
    const bool mux_ok = mux_pad && parser_src && gst_pad_link(parser_src, mux_pad) == GST_PAD_LINK_OK;
    if (parser_src) gst_object_unref(parser_src);
    if (mux_pad) gst_object_unref(mux_pad);
    if (!mux_ok) {
      if (err) *err = "failed to link parser → mux for " + options_.camera.id;
      return false;
    }
    return true;
  };

  auto LinkPreviewChain = [&]() -> bool {
    if (preview_queue_ && !gst_element_link(preview_queue_, preview_convert_)) {
      if (err) *err = "failed to link preview_queue → convert for " + options_.camera.id;
      return false;
    }
    if (!gst_element_link(preview_convert_, preview_sink_)) {
      if (err) *err = "failed to link preview chain for " + options_.camera.id;
      return false;
    }
    return true;
  };

  if (needs_tee) {
    if (!gst_element_link(source_tail, tee_)) {
      if (err) *err = "failed to link source → tee for " + options_.camera.id;
      return false;
    }
    if (options_.enable_record) {
      GstElement* record_entry = record_queue_ ? record_queue_ : encoder_;
      if (!LinkTeeBranch(record_entry, "record") || !LinkRecordChain()) return false;
    }
    if (options_.enable_preview) {
      GstElement* preview_entry = preview_queue_ ? preview_queue_ : preview_convert_;
      if (!LinkTeeBranch(preview_entry, "preview") || !LinkPreviewChain()) return false;
    }
    if (options_.enable_ai) {
      if (!LinkTeeBranch(ai_queue_, "ai") || !gst_element_link(ai_queue_, ai_sink_)) {
        if (err) *err = "failed to link AI branch for " + options_.camera.id;
        return false;
      }
    }
  } else if (options_.enable_ai) {
    if (!gst_element_link_many(source_tail, ai_queue_, ai_sink_, nullptr)) {
      if (err) *err = "failed to link AI branch for " + options_.camera.id;
      return false;
    }
  } else if (options_.enable_preview) {
    if (!gst_element_link_many(source_tail, preview_convert_, preview_sink_, nullptr)) {
      if (err) *err = "failed to link preview branch for " + options_.camera.id;
      return false;
    }
  } else if (options_.enable_record) {
    if (!gst_element_link(source_tail, encoder_) || !LinkRecordChain()) return false;
  }

  InstallProbes(source_tail);
  return true;
}

bool CameraPipeline::CreatePreviewSink(std::string* err) {
  for (const auto& sink_name : options_.sink_priority) {
    preview_sink_ = gst_element_factory_make(sink_name.c_str(), ("preview_sink_" + options_.camera.id).c_str());
    if (preview_sink_ != nullptr) {
      return true;
    }
  }
  if (err != nullptr) {
    *err = "failed to create preview sink for " + options_.camera.id;
  }
  return false;
}

void CameraPipeline::InstallProbes(GstElement* source_tail) {
  using GstPadPtr = std::unique_ptr<GstPad, GstUnrefDeleter<GstPad>>;

  GstPadPtr pad(gst_element_get_static_pad(source_tail, "src"));
  if (!pad) {
    return;
  }
  auto ctx = std::make_unique<ProbeContext>();
  ctx->self = this;
  ctx->stream_id = options_.camera.id;
  ctx->seq = &seq_;
  gst_pad_add_probe(pad.get(), GST_PAD_PROBE_TYPE_BUFFER, &CameraPipeline::OnCaptureProbe, ctx.get(), nullptr);
  probe_contexts_.push_back(std::move(ctx));
}

bool CameraPipeline::Start(std::string* err) {
  if (pipeline_ == nullptr) {
    if (err != nullptr) {
      *err = "pipeline not built for " + options_.camera.id;
    }
    return false;
  }

  stop_requested_.store(false);
  bus_done_.store(false);
  base_time_ns_.store(GST_CLOCK_TIME_NONE);

  using GstClockPtr = std::unique_ptr<GstClock, GstUnrefDeleter<GstClock>>;
  GstClockPtr clock(gst_system_clock_obtain());
  if (!clock) {
    if (err != nullptr) {
      *err = "failed to obtain GStreamer clock";
    }
    return false;
  }
  g_object_set(clock.get(), "clock-type", GST_CLOCK_TYPE_MONOTONIC, nullptr);
  gst_pipeline_use_clock(GST_PIPELINE(pipeline_), clock.get());

  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  const GstStateChangeReturn state = gst_element_get_state(pipeline_, nullptr, nullptr, 5 * GST_SECOND);
  if (state == GST_STATE_CHANGE_FAILURE) {
    if (err != nullptr) {
      *err = "pipeline failed to enter PLAYING state for " + options_.camera.id;
    }
    return false;
  }

  base_time_ns_.store(gst_element_get_base_time(pipeline_));
  BindPreviewOverlay();
  bus_thread_ = std::thread([this] { RunBusLoop(); });

  return true;
}

void CameraPipeline::Stop() {
  stop_requested_.store(true);

  if (pipeline_ != nullptr) {
    gst_element_send_event(pipeline_, gst_event_new_eos());
  }

  if (bus_thread_.joinable()) {
    bus_thread_.join();
  }

  if (pipeline_ != nullptr) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
  }

  pipeline_ = nullptr;
  source_ = nullptr;
  source_caps_ = nullptr;
  jpeg_decoder_ = nullptr;
  normalize_convert_ = nullptr;
  normalize_caps_ = nullptr;
  preview_convert_ = nullptr;
  preview_sink_ = nullptr;
  encoder_ = nullptr;
  parser_ = nullptr;
  mux_ = nullptr;
  record_sink_ = nullptr;
  tee_ = nullptr;
  ai_queue_ = nullptr;
  ai_sink_ = nullptr;
  record_queue_ = nullptr;
  preview_queue_ = nullptr;
  base_time_ns_.store(GST_CLOCK_TIME_NONE);
  probe_contexts_.clear();
}

void CameraPipeline::SetPreviewWindow(WId window_id) {
  options_.preview_window_id = window_id;
  BindPreviewOverlay();
}

const std::string& CameraPipeline::camera_id() const {
  return options_.camera.id;
}

const std::string& CameraPipeline::resolved_device() const {
  return resolved_device_;
}

std::string CameraPipeline::record_output_path() const {
  return record_output_path_;
}

void CameraPipeline::RunBusLoop() {
  using GstBusPtr = std::unique_ptr<GstBus, GstUnrefDeleter<GstBus>>;

  GstBusPtr bus(pipeline_ != nullptr ? gst_element_get_bus(pipeline_) : nullptr);
  if (!bus) {
    Fail("missing_bus", options_.enable_record);
    bus_done_.store(true);
    return;
  }

  while (!stop_requested_.load()) {
    GstMessage* message =
        gst_bus_timed_pop_filtered(bus.get(), 200 * GST_MSECOND,
                                   static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (message == nullptr) {
      continue;
    }

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
      GError* error = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(message, &error, &debug);
      const std::string reason = error != nullptr ? error->message : "unknown";
      g_clear_error(&error);
      g_free(debug);
      gst_message_unref(message);
      Fail(reason, options_.enable_record);
      bus_done_.store(true);
      return;
    }

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
      gst_message_unref(message);
      break;
    }
    gst_message_unref(message);
  }

  bus_done_.store(true);
}

void CameraPipeline::EmitTelemetry(TelemetryEvent event) {
  if (telemetry_callback_) {
    telemetry_callback_(std::move(event));
  }
}

void CameraPipeline::Fail(const std::string& reason, bool fatal) {
  TelemetryEvent event;
  event.monotonic_ns = rkinfra::ClockMonotonicNs();
  event.stream_id = options_.camera.id;
  event.category = "media";
  event.stage = "pipeline";
  event.status = "error";
  event.reason = reason;
  EmitTelemetry(std::move(event));

  if (error_callback_) {
    error_callback_(reason, fatal);
  }
}

void CameraPipeline::BindPreviewOverlay() {
  if (!options_.enable_preview || preview_sink_ == nullptr || options_.preview_window_id == 0) {
    return;
  }
  if (!GST_IS_VIDEO_OVERLAY(preview_sink_)) {
    return;
  }
  gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY(preview_sink_),
                                      static_cast<guintptr>(options_.preview_window_id));
  gst_video_overlay_handle_events(GST_VIDEO_OVERLAY(preview_sink_), FALSE);
  gst_video_overlay_expose(GST_VIDEO_OVERLAY(preview_sink_));
}

GstPadProbeReturn CameraPipeline::OnCaptureProbe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* ctx = static_cast<ProbeContext*>(user_data);
  if (ctx == nullptr || ctx->self == nullptr || !(info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
    return GST_PAD_PROBE_OK;
  }

  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer == nullptr) {
    return GST_PAD_PROBE_OK;
  }

  const GstClockTime pts = GST_BUFFER_PTS(buffer);
  const GstClockTime base = ctx->self->base_time_ns_.load();

  TelemetryEvent event;
  event.stream_id = ctx->stream_id;
  event.seq = ctx->seq->fetch_add(1) + 1;
  event.category = "media";
  event.stage = "capture";
  event.status = GST_CLOCK_TIME_IS_VALID(pts) ? "ok" : "missing";
  event.pts_ns = GST_CLOCK_TIME_IS_VALID(pts) ? static_cast<int64_t>(pts) : -1;
  event.monotonic_ns = GST_CLOCK_TIME_IS_VALID(base) && GST_CLOCK_TIME_IS_VALID(pts)
                           ? static_cast<uint64_t>(base + pts)
                           : rkinfra::ClockMonotonicNs();
  if (!GST_CLOCK_TIME_IS_VALID(pts)) {
    event.reason = "invalid_pts";
  }
  ctx->self->EmitTelemetry(std::move(event));
  return GST_PAD_PROBE_OK;
}

GstFlowReturn CameraPipeline::OnAiSample(GstAppSink* sink, gpointer user_data) {
  auto* self = static_cast<CameraPipeline*>(user_data);
  if (!self || !self->options_.ai_sample_callback) {
    return GST_FLOW_OK;
  }

  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (!sample) {
    return GST_FLOW_OK;
  }

  self->options_.ai_sample_callback(sample);
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

}  // namespace rkstudio::media
