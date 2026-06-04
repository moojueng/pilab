#pragma once

#ifdef USE_ONNX_MOCK
#include "s_nav_core/onnx_mock.hpp"
#else
#include <onnxruntime_cxx_api.h>
#endif

#include <memory>
#include <string>
#include <vector>

class SsmInference {
public:
    explicit SsmInference(const std::string& model_path);
    ~SsmInference();

    std::vector<float> predictNextWaypoint(
        const std::vector<float>& graph_features,
        const std::string& command);

    int predictAction(const std::vector<float>& policy_features);

private:
    static constexpr int kActionInputDim = 14;
    static constexpr int kNumActions = 4;

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
};
