#include "mediapipe/common/rknn_model.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

namespace mediapipe_demo {

namespace {

bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>* data) {
  if (data == nullptr) {
    return false;
  }

  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    std::cerr << "failed to open model: " << path << "\n";
    return false;
  }

  stream.unsetf(std::ios::skipws);
  stream.seekg(0, std::ios::end);
  const std::streampos size = stream.tellg();
  stream.seekg(0, std::ios::beg);
  data->reserve(static_cast<size_t>(size));
  data->insert(data->begin(),
               std::istream_iterator<uint8_t>(stream),
               std::istream_iterator<uint8_t>());
  return !data->empty();
}

}  // namespace

RknnModel::~RknnModel() {
  Reset();
}

bool RknnModel::Load(const std::string& model_path, int core_mask) {
  Reset();
  if (!ReadBinaryFile(model_path, &model_data_)) {
    return false;
  }

  int ret = rknn_init(&context_, model_data_.data(),
                      static_cast<uint32_t>(model_data_.size()), 0, nullptr);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_init failed for " << model_path << ", ret=" << ret << "\n";
    Reset();
    return false;
  }

  const int requested_core_mask =
      core_mask >= 0 ? core_mask : static_cast<int>(RKNN_NPU_CORE_0_1_2);
  ret = rknn_set_core_mask(context_, static_cast<rknn_core_mask>(requested_core_mask));
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_set_core_mask failed, fallback to driver default, ret=" << ret << "\n";
  }

  ret = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM failed, ret=" << ret << "\n";
    Reset();
    return false;
  }

  if (!QueryAttrs(RKNN_QUERY_INPUT_ATTR, io_num_.n_input, &input_attrs_) ||
      !QueryAttrs(RKNN_QUERY_OUTPUT_ATTR, io_num_.n_output, &output_attrs_)) {
    Reset();
    return false;
  }
  if (!PrepareIoMemories()) {
    Reset();
    return false;
  }

  return true;
}

bool RknnModel::CopyInput(const cv::Mat& input) {
  if (context_ == 0 || input.empty() || !input.isContinuous() || input_mems_.empty()) {
    return false;
  }
  rknn_tensor_mem* const input_mem = input_mems_[0];
  if (input_mem == nullptr || input_mem->virt_addr == nullptr) {
    return false;
  }

  const size_t copy_size = input.total() * input.elemSize();
  if (copy_size > input_mem->size) {
    std::cerr << "input buffer too small, required=" << copy_size
              << ", available=" << input_mem->size << "\n";
    return false;
  }
  std::memcpy(input_mem->virt_addr, input.ptr<unsigned char>(0), copy_size);
  return SyncInputMemory();
}

bool RknnModel::SyncInputMemory(size_t index) {
  if (context_ == 0 || index >= input_mems_.size()) {
    return false;
  }
  rknn_tensor_mem* const input_mem = input_mems_[index];
  if (input_mem == nullptr) {
    return false;
  }
  const int ret = rknn_mem_sync(context_, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_mem_sync input failed, index=" << index << ", ret=" << ret << "\n";
    return false;
  }
  return true;
}

bool RknnModel::Run(std::vector<std::vector<float>>* outputs) {
  if (outputs == nullptr || context_ == 0) {
    return false;
  }

  int ret = rknn_run(context_, nullptr);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_run failed, ret=" << ret << "\n";
    return false;
  }

  outputs->resize(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    rknn_tensor_mem* const output_mem = output_mems_[i];
    if (output_mem == nullptr || output_mem->virt_addr == nullptr) {
      return false;
    }
    ret = rknn_mem_sync(context_, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_mem_sync output failed, index=" << i << ", ret=" << ret << "\n";
      return false;
    }
    std::memcpy(output_buffers_[i].data(),
                output_mem->virt_addr,
                output_buffers_[i].size() * sizeof(float));
    (*outputs)[i] = output_buffers_[i];
  }
  return true;
}

bool RknnModel::Infer(const cv::Mat& input, std::vector<std::vector<float>>* outputs) {
  return CopyInput(input) && Run(outputs);
}

const std::vector<rknn_tensor_attr>& RknnModel::InputAttrs() const {
  return input_attrs_;
}

const std::vector<rknn_tensor_attr>& RknnModel::OutputAttrs() const {
  return output_attrs_;
}

const rknn_tensor_attr& RknnModel::InputAttr(size_t index) const {
  return input_attrs_.at(index);
}

rknn_tensor_mem* RknnModel::InputMemory(size_t index) {
  return input_mems_.at(index);
}

bool RknnModel::QueryAttrs(rknn_query_cmd query,
                           uint32_t tensor_count,
                           std::vector<rknn_tensor_attr>* attrs) {
  if (attrs == nullptr) {
    return false;
  }
  attrs->assign(tensor_count, {});
  for (uint32_t i = 0; i < tensor_count; ++i) {
    (*attrs)[i].index = i;
    const int ret = rknn_query(context_, query, &(*attrs)[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_query attr failed, index=" << i << ", ret=" << ret << "\n";
      return false;
    }
  }
  return true;
}

bool RknnModel::PrepareIoMemories() {
  input_mems_.assign(input_attrs_.size(), nullptr);
  for (size_t i = 0; i < input_attrs_.size(); ++i) {
    input_attrs_[i].type = RKNN_TENSOR_UINT8;
    input_attrs_[i].fmt = RKNN_TENSOR_NHWC;
    const uint32_t mem_size =
        input_attrs_[i].size_with_stride > 0 ? input_attrs_[i].size_with_stride
                                             : input_attrs_[i].size;
    input_mems_[i] = rknn_create_mem(context_, mem_size);
    if (input_mems_[i] == nullptr) {
      std::cerr << "rknn_create_mem input failed, index=" << i << "\n";
      return false;
    }
    const int ret = rknn_set_io_mem(context_, input_mems_[i], &input_attrs_[i]);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_set_io_mem input failed, index=" << i << ", ret=" << ret << "\n";
      return false;
    }
  }

  output_mems_.assign(output_attrs_.size(), nullptr);
  output_buffers_.assign(output_attrs_.size(), {});
  for (size_t i = 0; i < output_attrs_.size(); ++i) {
    output_buffers_[i].assign(output_attrs_[i].n_elems, 0.0f);
    output_attrs_[i].type = RKNN_TENSOR_FLOAT32;
    const uint32_t output_size =
        static_cast<uint32_t>(output_buffers_[i].size() * sizeof(float));
    output_mems_[i] = rknn_create_mem(context_, output_size);
    if (output_mems_[i] == nullptr) {
      std::cerr << "rknn_create_mem output failed, index=" << i << "\n";
      return false;
    }
    const int ret = rknn_set_io_mem(context_, output_mems_[i], &output_attrs_[i]);
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_set_io_mem output failed, index=" << i << ", ret=" << ret << "\n";
      return false;
    }
  }
  return true;
}

void RknnModel::Reset() {
  for (auto* mem : input_mems_) {
    if (mem != nullptr && context_ != 0) {
      rknn_destroy_mem(context_, mem);
    }
  }
  input_mems_.clear();
  for (auto* mem : output_mems_) {
    if (mem != nullptr && context_ != 0) {
      rknn_destroy_mem(context_, mem);
    }
  }
  output_mems_.clear();
  if (context_ != 0) {
    rknn_destroy(context_);
    context_ = 0;
  }
  io_num_ = {};
  input_attrs_.clear();
  output_attrs_.clear();
  output_buffers_.clear();
  model_data_.clear();
}

}  // namespace mediapipe_demo
