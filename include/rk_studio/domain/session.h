#pragma once

#include <filesystem>
#include <fstream>
#include <string>

namespace rkstudio {

namespace fs = std::filesystem;

struct SessionArtifacts {
  std::string session_id;
  fs::path session_dir;
  fs::path studio_event_path;
};

SessionArtifacts CreateSessionArtifacts(const std::string& output_dir, const std::string& prefix);
bool EnsureSessionDirectory(const SessionArtifacts& artifacts, std::string* err);

class JsonlFileWriter {
 public:
  JsonlFileWriter() = default;
  ~JsonlFileWriter();

  JsonlFileWriter(const JsonlFileWriter&) = delete;
  JsonlFileWriter& operator=(const JsonlFileWriter&) = delete;

  bool Open(const fs::path& path, std::string* err);
  void WriteLine(const std::string& line);
  void Close();

 private:
  std::ofstream out_;
};

}  // namespace rkstudio
