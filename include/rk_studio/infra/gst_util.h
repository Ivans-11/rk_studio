#pragma once

#include <gst/gst.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace rkinfra {

std::string Uppercase(std::string value);
std::string NameWithIndex(const std::string& prefix, size_t index);
bool IsJpegLikeFormat(std::string_view value);
bool IsNv12Format(std::string_view value);
int ToV4l2IoMode(const std::string& value, std::string* err);

template <typename T>
void SetPropertyIfExists(GstElement* element, const char* property, const T& value) {
  if (!element) {
    return;
  }
  if (g_object_class_find_property(G_OBJECT_GET_CLASS(element), property) != nullptr) {
    g_object_set(G_OBJECT(element), property, value, nullptr);
  }
}

}  // namespace rkinfra
