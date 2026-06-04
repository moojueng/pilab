#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "s_nav_msgs/msg/graph_node.hpp"

/**
 * @brief Vision-Graph 표현체를 관리하는 클래스
 */
struct GraphNode {
    int node_id;
    cv::Point2f position;
    std::string semantic_label;
    std::vector<float> feature_vector;
    std::vector<int> neighbors;

    // ROS 2 메시지로 변환
    s_nav_msgs::msg::GraphNode to_msg() const {
        s_nav_msgs::msg::GraphNode msg;
        msg.node_id = node_id;
        msg.position.x = position.x;
        msg.position.y = position.y;
        msg.semantic_label = semantic_label;
        msg.feature_vector = feature_vector;
        msg.neighbors = neighbors;
        return msg;
    }
};

class GraphManager {
public:
    GraphManager();
    ~GraphManager();

    /**
     * @brief Grid World 데이터를 기반으로 그래프를 구축합니다.
     * @param grid 2D 격자 정보 (0: 빈공간, 1: 장애물)
     */
    void buildGraphFromGrid(const std::vector<std::vector<int>>& grid);

    std::vector<GraphNode> getGraph() const { return graph_nodes_; }

private:
    std::vector<GraphNode> graph_nodes_;
};
