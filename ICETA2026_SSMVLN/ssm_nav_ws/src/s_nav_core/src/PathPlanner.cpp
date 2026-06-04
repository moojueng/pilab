#include "s_nav_core/PathPlanner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

namespace {

float heuristic(const GraphNode& a, const GraphNode& b) {
    float dx = a.position.x - b.position.x;
    float dy = a.position.y - b.position.y;
    return std::sqrt(dx * dx + dy * dy);
}

float edge_cost(const GraphNode& a, const GraphNode& b) {
    return heuristic(a, b);
}

struct QueueItem {
    int node_id;
    float priority;

    bool operator>(const QueueItem& other) const {
        return priority > other.priority;
    }
};

}  // namespace

PathPlanner::PathPlanner(SsmInference& inference_engine)
    : inference_engine_(inference_engine) {}

std::vector<int> PathPlanner::generatePath(
    const std::vector<GraphNode>& graph,
    int start_id,
    int target_id,
    const std::string& command) {
    std::vector<int> path;

    if (graph.empty()) {
        std::cerr << "[PathPlanner] Cannot generate path: graph is empty." << std::endl;
        return path;
    }

    if (start_id < 0 || start_id >= static_cast<int>(graph.size())) {
        std::cerr << "[PathPlanner] Cannot generate path: invalid start_id "
                  << start_id << std::endl;
        return path;
    }

    if (target_id < 0 || target_id >= static_cast<int>(graph.size())) {
        std::cerr << "[PathPlanner] Cannot generate path: invalid target_id "
                  << target_id << std::endl;
        return path;
    }

    std::vector<float> predicted_waypoint =
        inference_engine_.predictNextWaypoint(graph[start_id].feature_vector, command);

    if (predicted_waypoint.size() >= 2) {
        std::cout << "[PathPlanner] SSM predicted waypoint: x="
                  << predicted_waypoint[0]
                  << ", y=" << predicted_waypoint[1] << std::endl;
    }

    const float inf = std::numeric_limits<float>::infinity();

    std::vector<float> g_score(graph.size(), inf);
    std::vector<int> parent(graph.size(), -1);
    std::vector<bool> closed(graph.size(), false);

    std::priority_queue<
        QueueItem,
        std::vector<QueueItem>,
        std::greater<QueueItem>> open_set;

    g_score[start_id] = 0.0f;
    open_set.push({start_id, heuristic(graph[start_id], graph[target_id])});

    while (!open_set.empty()) {
        int current = open_set.top().node_id;
        open_set.pop();

        if (closed[current]) {
            continue;
        }

        closed[current] = true;

        if (current == target_id) {
            break;
        }

        for (int neighbor : graph[current].neighbors) {
            if (neighbor < 0 || neighbor >= static_cast<int>(graph.size())) {
                continue;
            }

            if (closed[neighbor]) {
                continue;
            }

            float tentative_g =
                g_score[current] + edge_cost(graph[current], graph[neighbor]);

            if (tentative_g < g_score[neighbor]) {
                parent[neighbor] = current;
                g_score[neighbor] = tentative_g;

                float f_score =
                    tentative_g + heuristic(graph[neighbor], graph[target_id]);

                open_set.push({neighbor, f_score});
            }
        }
    }

    if (parent[target_id] == -1 && start_id != target_id) {
        std::cerr << "[PathPlanner] No path found from "
                  << start_id << " to " << target_id << std::endl;
        return path;
    }

    int current = target_id;
    while (current != -1) {
        path.push_back(current);

        if (current == start_id) {
            break;
        }

        current = parent[current];
    }

    std::reverse(path.begin(), path.end());

    std::cout << "[PathPlanner] A* path generated with "
              << path.size() << " nodes, cost="
              << g_score[target_id] << std::endl;

    return path;
}
