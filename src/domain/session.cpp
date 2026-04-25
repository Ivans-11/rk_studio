#include "rk_studio/domain/session.h"

#include <chrono>
#include <ctime>
#include <exception>
#include <iomanip>
#include <sstream>

namespace rkstudio {
namespace {

std::string NowLocalCompact() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&now_time, &tm);

  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d-%H%M%S");
  return out.str();
}

}  // namespace

SessionArtifacts CreateSessionArtifacts(const std::string& output_dir, const std::string& prefix) {
  SessionArtifacts artifacts;
  artifacts.session_id = prefix + "-" + NowLocalCompact();
  artifacts.session_dir = fs::path(output_dir) / artifacts.session_id;
  artifacts.studio_event_path = artifacts.session_dir / "studio.events.jsonl";
  return artifacts;
}

bool EnsureSessionDirectory(const SessionArtifacts& artifacts, std::string* err) {
  try {
    fs::create_directories(artifacts.session_dir);
    return true;
  } catch (const std::exception& ex) {
    if (err) {
      *err = std::string("failed to create session dir: ") + ex.what();
    }
    return false;
  }
}

JsonlFileWriter::~JsonlFileWriter() {
  Close();
}

bool JsonlFileWriter::Open(const fs::path& path, std::string* err) {
  Close();
  out_.open(path, std::ios::out | std::ios::app);
  if (!out_.is_open()) {
    if (err) {
      *err = "failed to open " + path.string();
    }
    return false;
  }
  return true;
}

void JsonlFileWriter::WriteLine(const std::string& line) {
  if (out_.is_open()) {
    out_ << line << "\n";
    out_.flush();
  }
}

void JsonlFileWriter::Close() {
  if (out_.is_open()) {
    out_.close();
  }
}

}  // namespace rkstudio
