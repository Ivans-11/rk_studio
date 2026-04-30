#include "rk_studio/infra/zenoh_publisher.h"

#include <cstddef>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include <zenoh.h>

namespace rkinfra {
namespace {

std::string JsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  out << '"';
  return out.str();
}

std::string JsonStringArray(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) out << ',';
    out << JsonString(values[i]);
  }
  out << ']';
  return out.str();
}

std::string NormalizePrefix(std::string prefix) {
  while (!prefix.empty() && prefix.front() == '/') {
    prefix.erase(prefix.begin());
  }
  while (!prefix.empty() && prefix.back() == '/') {
    prefix.pop_back();
  }
  return prefix.empty() ? "rk_studio" : prefix;
}

std::vector<std::string> EffectiveConnectEndpoints(const rkstudio::ZenohConfig& config) {
  std::vector<std::string> endpoints = config.connect;
  if (!config.server_ip.empty()) {
    endpoints.push_back("tcp/" + config.server_ip + ":" + std::to_string(config.server_port));
  }
  return endpoints;
}

}  // namespace

class ZenohPublisher::Impl {
 public:
  ~Impl() { Stop(); }

  bool Start(const rkstudio::ZenohConfig& config, std::string* err) {
    Stop();
    zc_init_log_from_env_or("error");

    z_owned_config_t zenoh_config;
    z_config_default(&zenoh_config);
    const std::vector<std::string> connect = EffectiveConnectEndpoints(config);
    if (!InsertConfig(zenoh_config, Z_CONFIG_MODE_KEY, JsonString(config.mode), err) ||
        (!connect.empty() &&
         !InsertConfig(zenoh_config, Z_CONFIG_CONNECT_KEY, JsonStringArray(connect), err)) ||
        (!config.listen.empty() &&
         !InsertConfig(zenoh_config, Z_CONFIG_LISTEN_KEY, JsonStringArray(config.listen), err))) {
      z_drop(z_move(zenoh_config));
      return false;
    }

    z_owned_session_t session;
    if (z_open(&session, z_move(zenoh_config), nullptr) < 0) {
      if (err) *err = "failed to open Zenoh session";
      return false;
    }

    std::lock_guard<std::mutex> lock(mu_);
    session_ = session;
    active_ = true;
    key_prefix_ = NormalizePrefix(config.key_prefix);
    return true;
  }

  void Stop() {
    std::lock_guard<std::mutex> lock(mu_);
    if (active_) {
      z_drop(z_move(session_));
      active_ = false;
    }
    result_publishing_enabled_ = false;
    key_prefix_.clear();
  }

  bool active() const {
    std::lock_guard<std::mutex> lock(mu_);
    return active_;
  }

  void SetResultPublishingEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mu_);
    result_publishing_enabled_ = enabled;
  }

  bool result_publishing_enabled() const {
    std::lock_guard<std::mutex> lock(mu_);
    return result_publishing_enabled_;
  }

  bool PublishMediapipe(const std::string& camera_id, const std::string& payload) {
    (void)camera_id;
    return Publish("halmet/mediapipe", payload);
  }

  bool PublishYolo(const std::string& camera_id, const std::string& payload) {
    return Publish(key_prefix_ + "/yolo/" + camera_id + "/objects", payload);
  }

  bool PublishJson(const std::string& key, const std::string& payload) {
    return Publish(key, payload);
  }

 private:
  bool InsertConfig(z_owned_config_t& config,
                    const char* key,
                    const std::string& value,
                    std::string* err) {
    if (zc_config_insert_json5(z_loan_mut(config), key, value.c_str()) < 0) {
      if (err) *err = std::string("failed to set Zenoh config: ") + key;
      return false;
    }
    return true;
  }

  bool Publish(const std::string& key, const std::string& payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!active_) {
      return false;
    }

    z_view_keyexpr_t keyexpr;
    z_view_keyexpr_from_str(&keyexpr, key.c_str());

    z_put_options_t options;
    z_put_options_default(&options);
    z_owned_encoding_t encoding;
    z_encoding_clone(&encoding, z_encoding_application_json());
    options.encoding = z_move(encoding);

    z_owned_bytes_t bytes;
    z_bytes_copy_from_str(&bytes, payload.c_str());
    return z_put(z_loan(session_), z_loan(keyexpr), z_move(bytes), &options) >= 0;
  }

  mutable std::mutex mu_;
  z_owned_session_t session_;
  bool active_ = false;
  bool result_publishing_enabled_ = false;
  std::string key_prefix_;
};

ZenohPublisher::ZenohPublisher() : impl_(std::make_unique<Impl>()) {}

ZenohPublisher::~ZenohPublisher() = default;

bool ZenohPublisher::Start(const rkstudio::ZenohConfig& config, std::string* err) {
  return impl_->Start(config, err);
}

void ZenohPublisher::Stop() {
  impl_->Stop();
}

bool ZenohPublisher::active() const {
  return impl_->active();
}

void ZenohPublisher::SetResultPublishingEnabled(bool enabled) {
  impl_->SetResultPublishingEnabled(enabled);
}

bool ZenohPublisher::result_publishing_enabled() const {
  return impl_->result_publishing_enabled();
}

bool ZenohPublisher::PublishMediapipe(const std::string& camera_id, const std::string& payload) {
  return impl_->PublishMediapipe(camera_id, payload);
}

bool ZenohPublisher::PublishYolo(const std::string& camera_id, const std::string& payload) {
  return impl_->PublishYolo(camera_id, payload);
}

bool ZenohPublisher::PublishJson(const std::string& key, const std::string& payload) {
  return impl_->PublishJson(key, payload);
}

}  // namespace rkinfra
