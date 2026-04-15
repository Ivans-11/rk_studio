#include "rk_studio/domain/session.h"

#include <exception>

#include "rk_studio/infra/runtime.h"

namespace rkstudio {

SessionArtifacts CreateSessionArtifacts(const std::string& output_dir, const std::string& prefix) {
  SessionArtifacts artifacts;
  artifacts.session_id = prefix + "-" + rkinfra::NowLocalCompact();
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
