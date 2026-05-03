#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <exception>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gst/gst.h>
#include <unistd.h>

#include <QCoreApplication>
#include <QSocketNotifier>
#include <QTimer>

#include "rk_studio/domain/config.h"
#include "rk_studio/runtime/runtime_manager.h"

namespace {

enum class Command {
  kRecord,
  kRtsp,
};

struct RecognizeSelection {
  bool hand = false;
  bool yolo = false;
  bool face = false;
  bool audio = false;
};

struct CliOptions {
  Command command = Command::kRecord;
  std::optional<uint64_t> duration_seconds;
  std::optional<std::string> output_dir;
  RecognizeSelection recognize;
};

struct ConfigPaths {
  std::filesystem::path board;
  std::filesystem::path profile;
};

rkstudio::runtime::RuntimeManager* g_manager = nullptr;
Command g_command = Command::kRecord;
bool g_stopping = false;
int g_signal_pipe_fds[2] = {-1, -1};

void PrintHelp(std::ostream& out, const char* argv0) {
  out << "Usage:\n"
      << "  " << argv0 << " record [--duration <seconds>] [--output <dir>] [--recognize <list>]\n"
      << "  " << argv0 << " rtsp [--recognize <list>]\n"
      << "\n"
      << "Recognize list:\n"
      << "  none | all | hand | yolo | face | audio | comma-separated combinations\n"
      << "\n"
      << "Examples:\n"
      << "  " << argv0 << " record --duration 600\n"
      << "  " << argv0 << " record --duration 600 --output records/test01 --recognize face,audio\n"
      << "  " << argv0 << " rtsp --recognize all\n";
}

bool SplitCommaList(const std::string& value, std::vector<std::string>* parts) {
  if (parts == nullptr) {
    return false;
  }
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      return false;
    }
    parts->push_back(item);
  }
  return !parts->empty();
}

bool ParseDuration(const std::string& value, uint64_t* out, std::string* err) {
  if (value.empty()) {
    if (err) *err = "--duration requires a value";
    return false;
  }
  try {
    size_t parsed = 0;
    const auto duration = std::stoull(value, &parsed, 10);
    if (parsed != value.size() || duration == 0) {
      if (err) *err = "--duration must be a positive integer";
      return false;
    }
    constexpr uint64_t kMaxTimerSeconds =
        static_cast<uint64_t>(std::numeric_limits<int>::max()) / 1000ULL;
    if (duration > kMaxTimerSeconds) {
      if (err) *err = "--duration is too large";
      return false;
    }
    *out = duration;
    return true;
  } catch (const std::exception&) {
    if (err) *err = "--duration must be a positive integer";
    return false;
  }
}

bool ParseRecognize(const std::string& value, RecognizeSelection* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "recognize output is null";
    return false;
  }
  RecognizeSelection parsed;
  if (value.empty() || value == "none") {
    *out = parsed;
    return true;
  }
  if (value == "all") {
    parsed.hand = true;
    parsed.yolo = true;
    parsed.face = true;
    parsed.audio = true;
    *out = parsed;
    return true;
  }

  std::vector<std::string> parts;
  if (!SplitCommaList(value, &parts)) {
    if (err) *err = "recognize list contains an empty item";
    return false;
  }

  std::set<std::string> seen;
  for (const auto& item : parts) {
    if (!seen.insert(item).second) {
      if (err) *err = "duplicate recognize item: " + item;
      return false;
    }
    if (item == "hand") {
      parsed.hand = true;
    } else if (item == "yolo") {
      parsed.yolo = true;
    } else if (item == "face") {
      parsed.face = true;
    } else if (item == "audio") {
      parsed.audio = true;
    } else if (item == "all" || item == "none") {
      if (err) *err = "'all' and 'none' cannot be combined with other recognize items";
      return false;
    } else {
      if (err) *err = "unknown recognize item: " + item;
      return false;
    }
  }
  *out = parsed;
  return true;
}

void SetNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags >= 0) {
    const int ignored = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    (void)ignored;
  }
}

std::string RecognizeToString(const RecognizeSelection& recognize) {
  std::vector<std::string> items;
  if (recognize.hand) items.emplace_back("hand");
  if (recognize.yolo) items.emplace_back("yolo");
  if (recognize.face) items.emplace_back("face");
  if (recognize.audio) items.emplace_back("audio");
  if (items.empty()) {
    return "none";
  }
  std::ostringstream out;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) out << ',';
    out << items[i];
  }
  return out.str();
}

bool ParseArgs(int argc, char** argv, CliOptions* options, std::string* err) {
  if (options == nullptr) {
    if (err) *err = "options output is null";
    return false;
  }
  if (argc < 2) {
    if (err) *err = "missing command";
    return false;
  }

  const std::string command = argv[1];
  if (command == "--help" || command == "-h" || command == "help") {
    return false;
  }
  if (command == "record") {
    options->command = Command::kRecord;
  } else if (command == "rtsp") {
    options->command = Command::kRtsp;
  } else {
    if (err) *err = "unknown command: " + command;
    return false;
  }

  std::set<std::string> seen;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name, std::string* value) -> bool {
      if (i + 1 >= argc) {
        if (err) *err = name + " requires a value";
        return false;
      }
      *value = argv[++i];
      return true;
    };

    if (arg == "--duration") {
      if (options->command != Command::kRecord) {
        if (err) *err = "--duration is only supported by record";
        return false;
      }
      if (!seen.insert(arg).second) {
        if (err) *err = "duplicate option: " + arg;
        return false;
      }
      std::string value;
      uint64_t duration = 0;
      if (!require_value(arg, &value) || !ParseDuration(value, &duration, err)) {
        return false;
      }
      options->duration_seconds = duration;
    } else if (arg == "--output") {
      if (options->command != Command::kRecord) {
        if (err) *err = "--output is only supported by record";
        return false;
      }
      if (!seen.insert(arg).second) {
        if (err) *err = "duplicate option: " + arg;
        return false;
      }
      std::string value;
      if (!require_value(arg, &value)) {
        return false;
      }
      if (value.empty()) {
        if (err) *err = "--output cannot be empty";
        return false;
      }
      options->output_dir = value;
    } else if (arg == "--recognize") {
      if (!seen.insert(arg).second) {
        if (err) *err = "duplicate option: " + arg;
        return false;
      }
      std::string value;
      if (!require_value(arg, &value) || !ParseRecognize(value, &options->recognize, err)) {
        return false;
      }
    } else if (arg == "--help" || arg == "-h") {
      return false;
    } else {
      if (err) *err = "unknown option: " + arg;
      return false;
    }
  }

  return true;
}

std::filesystem::path CanonicalIfExists(const std::filesystem::path& path) {
  std::error_code ec;
  if (std::filesystem::exists(path, ec)) {
    const auto canonical = std::filesystem::canonical(path, ec);
    if (!ec) {
      return canonical;
    }
  }
  return path;
}

std::optional<std::filesystem::path> FirstExisting(const std::vector<std::filesystem::path>& candidates) {
  std::error_code ec;
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate, ec) && std::filesystem::is_regular_file(candidate, ec)) {
      return CanonicalIfExists(candidate);
    }
  }
  return std::nullopt;
}

ConfigPaths ResolveConfigPaths(const char* argv0) {
  const auto cwd = std::filesystem::current_path();
  std::filesystem::path app_dir = cwd;
  if (argv0 != nullptr && std::string(argv0).find('/') != std::string::npos) {
    std::error_code ec;
    app_dir = std::filesystem::absolute(argv0, ec).parent_path();
    if (ec) {
      app_dir = cwd;
    }
  }

  const std::vector<std::filesystem::path> board_candidates{
      cwd / "config" / "board.toml",
      app_dir / "../config/board.toml",
      app_dir / "config/board.toml",
  };
  const std::vector<std::filesystem::path> profile_candidates{
      cwd / "config" / "profile.toml",
      app_dir / "../config/profile.toml",
      app_dir / "config/profile.toml",
  };

  ConfigPaths paths;
  paths.board = FirstExisting(board_candidates).value_or(board_candidates.front());
  paths.profile = FirstExisting(profile_candidates).value_or(profile_candidates.front());
  return paths;
}

void StopAndQuit() {
  if (g_stopping) {
    return;
  }
  g_stopping = true;
  if (g_manager != nullptr) {
    if (g_command == Command::kRecord) {
      std::cout << "[cli] stopping recording\n";
      g_manager->StopRecording();
    } else {
      std::cout << "[cli] stopping RTSP\n";
      g_manager->StopRtsp();
    }
    g_manager->StopAll();
  }
  QCoreApplication::quit();
}

void HandleSignal(int) {
  const uint8_t byte = 1;
  if (g_signal_pipe_fds[1] >= 0) {
    const ssize_t ignored = write(g_signal_pipe_fds[1], &byte, sizeof(byte));
    (void)ignored;
  }
}

bool EnableRecognizers(rkstudio::runtime::RuntimeManager* manager,
                       const RecognizeSelection& recognize,
                       std::string* err) {
  if (manager == nullptr) {
    if (err) *err = "runtime manager is null";
    return false;
  }
  if (recognize.hand && !manager->ToggleMediapipe(true, err)) {
    return false;
  }
  if (recognize.yolo && !manager->ToggleYolo(true, err)) {
    return false;
  }
  if (recognize.face && !manager->ToggleFaceExpression(true, err)) {
    return false;
  }
  if (recognize.audio && !manager->ToggleAudioEvent(true, err)) {
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gst_init(nullptr, nullptr);

  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName(QStringLiteral("rk_studio_cli"));

  CliOptions options;
  std::string err;
  if (!ParseArgs(argc, argv, &options, &err)) {
    if (!err.empty()) {
      std::cerr << "[cli] error: " << err << "\n\n";
    }
    PrintHelp(err.empty() ? std::cout : std::cerr, argv[0]);
    return err.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  const ConfigPaths paths = ResolveConfigPaths(argv[0]);
  std::cout << "[cli] loading config: " << paths.board.string() << "\n";
  std::cout << "[cli] loading profile: " << paths.profile.string() << "\n";

  rkstudio::BoardConfig board_config;
  rkstudio::SessionProfile profile;
  if (!rkstudio::LoadBoardConfig(paths.board.string(), &board_config, &err) ||
      !rkstudio::LoadSessionProfile(paths.profile.string(), &profile, &err)) {
    std::cerr << "[cli] error: failed to load config: " << err << "\n";
    return EXIT_FAILURE;
  }

  if (options.output_dir.has_value()) {
    profile.output_dir = *options.output_dir;
  }

  rkstudio::runtime::RuntimeManager manager;
  g_manager = &manager;
  g_command = options.command;

  manager.LoadBoardConfig(board_config);
  manager.ApplySessionProfile(profile);

  if (pipe(g_signal_pipe_fds) == 0) {
    SetNonBlocking(g_signal_pipe_fds[0]);
    SetNonBlocking(g_signal_pipe_fds[1]);
    auto* signal_notifier = new QSocketNotifier(g_signal_pipe_fds[0], QSocketNotifier::Read, &app);
    QObject::connect(signal_notifier, &QSocketNotifier::activated, &app, [](int fd) {
      uint8_t buffer[32];
      const ssize_t ignored = read(fd, buffer, sizeof(buffer));
      (void)ignored;
      StopAndQuit();
    });
  } else {
    std::cerr << "[cli] warning: failed to install signal pipe: " << errno << "\n";
  }
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  std::cout << "[cli] recognize: " << RecognizeToString(options.recognize) << "\n";
  if (!EnableRecognizers(&manager, options.recognize, &err)) {
    std::cerr << "[cli] error: failed to enable recognizer: " << err << "\n";
    manager.StopAll();
    g_manager = nullptr;
    return EXIT_FAILURE;
  }

  bool started = false;
  if (options.command == Command::kRecord) {
    std::cout << "[cli] starting recording\n";
    started = manager.StartRecording(&err);
  } else {
    std::cout << "[cli] starting RTSP\n";
    started = manager.StartRtsp(&err);
  }
  if (!started) {
    std::cerr << "[cli] error: " << err << "\n";
    manager.StopAll();
    g_manager = nullptr;
    return EXIT_FAILURE;
  }

  if (options.command == Command::kRecord && options.duration_seconds.has_value()) {
    const uint64_t duration_ms = *options.duration_seconds * 1000ULL;
    QTimer::singleShot(static_cast<int>(duration_ms), []() {
      StopAndQuit();
    });
  }

  const int rc = app.exec();
  if (!g_stopping) {
    StopAndQuit();
  }
  g_manager = nullptr;
  return rc;
}
