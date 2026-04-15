#pragma once

#include <string>

#include "rk_studio/domain/types.h"

namespace rkstudio {

bool LoadBoardConfig(const std::string& path, BoardConfig* config, std::string* err);
bool LoadSessionProfile(const std::string& path, SessionProfile* profile, std::string* err);

}  // namespace rkstudio
