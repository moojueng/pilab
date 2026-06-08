#pragma once

#include "s_nav_core/GraphManager.hpp"
#include "s_nav_core/SsmInference.hpp"
#include <vector>

/**
 * @brief SSM 추론 결과를 바탕으로 최종 경로를 생성하는 클래스
 */
class PathPlanner {
public:
    PathPlanner(SsmInference& inference_engine);
    
    /**
     * @brief 목적지까지의 최적 경로(노드 ID 리스트)를 생성합니다.
     * @param graph 구축된 그래프 데이터
     * @param start_id 시작 노드 ID
     * @param target_id 목표 노드 ID (또는 의미론적 목표)
     * @param command 자연어 명령
     */
    std::vector<int> generatePath(const std::vector<GraphNode>& graph, int start_id, int target_id, const std::string& command);

private:
    SsmInference& inference_engine_;
};
