#pragma once

#include <cstdint>
#include <string>

namespace rkinfra {

std::string JsonEscape(const std::string& input);
std::string NowUtcIso8601();
std::string NowLocalCompact();
uint64_t ClockMonotonicNs();

}  // namespace rkinfra
