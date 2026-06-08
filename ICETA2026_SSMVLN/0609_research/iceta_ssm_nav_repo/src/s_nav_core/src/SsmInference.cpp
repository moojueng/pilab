#include "s_nav_core/SsmInference.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

SsmInference::SsmInference(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SSM_Inference"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    try {
        Ort::SessionOptions session_options;

#ifndef USE_ONNX_MOCK
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#endif

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

#ifndef USE_ONNX_MOCK
        Ort::AllocatorWithDefaultOptions allocator;

        std::size_t num_inputs = session_->GetInputCount();
        std::size_t num_outputs = session_->GetOutputCount();

        input_name_storage_.clear();
        output_name_storage_.clear();
        input_node_names_.clear();
        output_node_names_.clear();

        for (std::size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_name_storage_.push_back(name.get());
        }

        for (std::size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_name_storage_.push_back(name.get());
        }

        for (const std::string& name : input_name_storage_) {
            input_node_names_.push_back(name.c_str());
        }

        for (const std::string& name : output_name_storage_) {
            output_node_names_.push_back(name.c_str());
        }

        std::cout << "[SsmInference] ONNX model loaded from: " << model_path << std::endl;
        if (!input_name_storage_.empty() && !output_name_storage_.empty()) {
            std::cout << "[SsmInference] input=" << input_name_storage_.front()
                      << ", output=" << output_name_storage_.front() << std::endl;
        }
#else
        std::cout << "[SsmInference] Mock model loaded from: " << model_path << std::endl;
#endif
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    }
}

SsmInference::~SsmInference() {}

std::vector<float> SsmInference::predictNextWaypoint(
    const std::vector<float>& graph_features,
    const std::string& command) {
    std::cout << "[SsmInference] Predicting next waypoint for command: "
              << command << std::endl;

    std::cout << "[SsmInference] graph_features dim="
              << graph_features.size() << std::endl;

    float first_feature = graph_features.empty() ? 0.0f : graph_features[0];

    return {
        0.5f + static_cast<float>(command.length()) * 0.01f,
        0.8f - first_feature * 0.1f
    };
}

int SsmInference::predictAction(const std::vector<float>& policy_features) {
    if (policy_features.size() != static_cast<std::size_t>(kActionInputDim)) {
        std::cerr << "[SsmInference] Invalid policy feature dim: "
                  << policy_features.size()
                  << ", expected " << kActionInputDim << std::endl;
        return -1;
    }

#ifdef USE_ONNX_MOCK
    const float delta_goal_x = policy_features[4];
    const float delta_goal_y = policy_features[5];

    const bool free_up = policy_features[10] > 0.5f;
    const bool free_down = policy_features[11] > 0.5f;
    const bool free_left = policy_features[12] > 0.5f;
    const bool free_right = policy_features[13] > 0.5f;

    if (std::abs(delta_goal_x) > std::abs(delta_goal_y)) {
        if (delta_goal_x > 0.0f && free_right) {
            return 3;
        }
        if (delta_goal_x < 0.0f && free_left) {
            return 2;
        }
    }

    if (delta_goal_y > 0.0f && free_down) {
        return 1;
    }
    if (delta_goal_y < 0.0f && free_up) {
        return 0;
    }

    if (free_right) {
        return 3;
    }
    if (free_down) {
        return 1;
    }
    if (free_left) {
        return 2;
    }
    if (free_up) {
        return 0;
    }

    return -1;
#else
    if (!session_) {
        std::cerr << "[SsmInference] ONNX session is not initialized." << std::endl;
        return -1;
    }

    if (input_node_names_.empty() || output_node_names_.empty()) {
        std::cerr << "[SsmInference] ONNX input/output names are empty." << std::endl;
        return -1;
    }

    std::vector<float> input_data = policy_features;
    std::vector<int64_t> input_shape = {1, kActionInputDim};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    std::vector<float> logits(kNumActions, 0.0f);

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor,
        1,
        output_node_names_.data(),
        1);

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        std::cerr << "[SsmInference] Invalid ONNX output tensor." << std::endl;
        return -1;
    }

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    for (int i = 0; i < kNumActions; ++i) {
        logits[i] = output_data[i];
    }

    int best_action = static_cast<int>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));

    std::cout << "[SsmInference] action logits=["
              << logits[0] << ", "
              << logits[1] << ", "
              << logits[2] << ", "
              << logits[3] << "], action=" << best_action << std::endl;

    return best_action;
#endif
}
