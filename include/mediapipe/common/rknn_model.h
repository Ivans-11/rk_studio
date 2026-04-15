#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>

namespace mediapipe_demo {

class RknnModel {
 public:
  RknnModel() = default;
  ~RknnModel();

  RknnModel(const RknnModel&) = delete;
  RknnModel& operator=(const RknnModel&) = delete;

  bool Load(const std::string& model_path, int core_mask = -1);
  bool CopyInput(const cv::Mat& input);
  bool SyncInputMemory(size_t index = 0);
  bool Run(std::vector<std::vector<float>>* outputs);
  bool Infer(const cv::Mat& input, std::vector<std::vector<float>>* outputs);

  const std::vector<rknn_tensor_attr>& InputAttrs() const;
  const std::vector<rknn_tensor_attr>& OutputAttrs() const;
  const rknn_tensor_attr& InputAttr(size_t index = 0) const;
  rknn_tensor_mem* InputMemory(size_t index = 0);

 private:
  bool QueryAttrs(rknn_query_cmd query,
                  uint32_t tensor_count,
                  std::vector<rknn_tensor_attr>* attrs);
  bool PrepareIoMemories();
  void Reset();

  std::vector<uint8_t> model_data_;
  rknn_context context_ = 0;
  rknn_input_output_num io_num_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
  std::vector<rknn_tensor_mem*> input_mems_;
  std::vector<rknn_tensor_mem*> output_mems_;
  std::vector<std::vector<float>> output_buffers_;
};

}  // namespace mediapipe_demo
