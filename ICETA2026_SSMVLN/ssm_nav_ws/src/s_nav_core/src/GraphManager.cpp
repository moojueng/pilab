#include "s_nav_core/GraphManager.hpp"

#include <iostream>

GraphManager::GraphManager() {}
GraphManager::~GraphManager() {}

void GraphManager::buildGraphFromGrid(const std::vector<std::vector<int>>& grid) {
    graph_nodes_.clear();

    if (grid.empty() || grid[0].empty()) {
        std::cerr << "[GraphManager] Cannot build graph: grid is empty." << std::endl;
        return;
    }

    const int height = static_cast<int>(grid.size());
    const int width = static_cast<int>(grid[0].size());

    std::vector<std::vector<int>> cell_to_node(
        height,
        std::vector<int>(width, -1));

    int node_count = 0;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (grid[y][x] != 0) {
                continue;
            }

            GraphNode node;
            node.node_id = node_count;
            node.position = cv::Point2f(
                static_cast<float>(x),
                static_cast<float>(y));
            node.semantic_label = "free_space";

            node.feature_vector = {
                static_cast<float>(x) / static_cast<float>(width),
                static_cast<float>(y) / static_cast<float>(height),
                1.0f
            };

            cell_to_node[y][x] = node_count;
            graph_nodes_.push_back(node);
            ++node_count;
        }
    }

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int node_id = cell_to_node[y][x];
            if (node_id < 0) {
                continue;
            }

            for (int dir = 0; dir < 4; ++dir) {
                int nx = x + dx[dir];
                int ny = y + dy[dir];

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                int neighbor_id = cell_to_node[ny][nx];
                if (neighbor_id >= 0) {
                    graph_nodes_[node_id].neighbors.push_back(neighbor_id);
                }
            }
        }
    }

    std::cout << "[GraphManager] Built graph with "
              << graph_nodes_.size()
              << " nodes." << std::endl;
}
