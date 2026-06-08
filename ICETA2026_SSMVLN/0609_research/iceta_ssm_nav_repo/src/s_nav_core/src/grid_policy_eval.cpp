#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "onnxruntime_cxx_api.h"

struct Pos {
    int r = 0;
    int c = 0;

    bool operator<(const Pos& o) const {
        return std::tie(r, c) < std::tie(o.r, o.c);
    }

    bool operator==(const Pos& o) const {
        return r == o.r && c == o.c;
    }

    bool operator!=(const Pos& o) const {
        return !(*this == o);
    }
};

struct Metrics {
    bool success = false;
    int steps = 0;
    int collisions = 0;
    int revisits = 0;
    int observed_nodes = 0;
    int observed_edges = 0;
    int target_seen_step = -1;
    int fallback_count = 0;
    int frontier_switches = 0;
    int unique_visited = 0;
    double revisit_ratio = 0.0;
};

class GridPolicy {
public:
    GridPolicy(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "grid_policy_eval"),
          session_options_(),
          session_(nullptr) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::cout << "Loaded ONNX policy: " << model_path << std::endl;
    }

    int predictAction(const std::vector<float>& input) {
        std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(input.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"action_logits", "reward_values"};

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2);

        float* action_logits = outputs[0].GetTensorMutableData<float>();
        float* reward_values = outputs[1].GetTensorMutableData<float>();

        int best_action = 0;
        float best_score = -1e9f;

        for (int a = 0; a < 5; ++a) {
            float score = reward_values[a] + 0.2f * action_logits[a];
            if (score > best_score) {
                best_score = score;
                best_action = a;
            }
        }

        return best_action;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
};

class GridPolicyEval {
public:
    GridPolicyEval(const std::string& model_path, const std::string& planner_mode = "utility")
        : policy_(model_path), planner_mode_(planner_mode) {
        if (planner_mode_ != "nearest" && planner_mode_ != "utility" &&
            planner_mode_ != "utility_commit" && planner_mode_ != "policy_only") {
            std::cerr << "unknown planner_mode=" << planner_mode_
                      << ", fallback to utility_commit" << std::endl;
            planner_mode_ = "utility";
        }
    }

    bool loadHiddenMap(const std::string& path) {
        hidden_map_.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "failed to open map: " << path << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }

            std::vector<int> row;
            std::stringstream ss(line);
            std::string cell;

            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stoi(cell));
            }

            hidden_map_.push_back(row);
        }

        rows_ = static_cast<int>(hidden_map_.size());
        cols_ = rows_ > 0 ? static_cast<int>(hidden_map_[0].size()) : 0;
        observed_map_.assign(rows_, std::vector<int>(cols_, -1));
        visit_count_.assign(rows_, std::vector<int>(cols_, 0));

        return rows_ > 0 && cols_ > 0;
    }

    Metrics run(Pos start, int max_steps, int vision_radius) {
        robot_ = start;
        max_steps_ = max_steps;
        vision_radius_ = vision_radius;

        trajectory_.clear();
        visited_.clear();
        node_id_.clear();
        observed_edges_ = 0;
        fallback_count_ = 0;
        frontier_switches_ = 0;
        commitment_remaining_ = 0;
        has_subgoal_ = false;

        Metrics m;

        if (!inBounds(robot_) || isObstacleHidden(robot_)) {
            std::cerr << "invalid start" << std::endl;
            m.collisions = 1;
            return m;
        }

        int prev_action = 4;

        for (int step = 0; step < max_steps_; ++step) {
            observeLocalPatch();
            buildObservedGraph();
            trajectory_.push_back(robot_);
            visit_count_[robot_.r][robot_.c]++;

            if (targetVisibleInLocalPatch()) {
                m.success = true;
                m.steps = step;
                m.target_seen_step = step;
                break;
            }

            std::vector<float> input = buildFeature(prev_action);
            int policy_action = policy_.predictAction(input);
            int action = chooseHybridAction(policy_action, prev_action);

            Pos next = actionToNext(action);

            if (action == 4) {
                m.steps = step;
                break;
            }

            if (!move(next)) {
                m.collisions++;
                int fallback = selectFrontierAction(prev_action);
                fallback_count_++;
                next = actionToNext(fallback);
                if (!move(next)) {
                    m.steps = step;
                    break;
                }
                action = fallback;
            }

            if (visited_.count(robot_) > 0) {
                m.revisits++;
            }

            visited_.insert(robot_);
            prev_action = action;
        }

        if (!m.success && m.steps == 0) {
            m.steps = max_steps_;
        }

        m.observed_nodes = static_cast<int>(node_id_.size());
        m.observed_edges = observed_edges_;
        m.fallback_count = fallback_count_;
        m.frontier_switches = frontier_switches_;
        m.unique_visited = static_cast<int>(visited_.size());
        m.revisit_ratio = m.steps > 0 ? static_cast<double>(m.revisits) / static_cast<double>(m.steps) : 0.0;

        return m;
    }

    void saveTrajectory(const std::string& path) const {
        std::ofstream f(path);
        f << "step,r,c\n";
        for (size_t i = 0; i < trajectory_.size(); ++i) {
            f << i << "," << trajectory_[i].r << "," << trajectory_[i].c << "\n";
        }
    }

    void saveObservedMapCsv(const std::string& path) const {
        std::ofstream f(path);
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                if (c > 0) {
                    f << ",";
                }
                f << observed_map_[r][c];
            }
            f << "\n";
        }
    }

    void saveImagePpm(const std::string& path) const {
        const int scale = 50;
        const int w = cols_ * scale;
        const int h = rows_ * scale;
        std::vector<unsigned char> img(w * h * 3, 255);

        auto paint = [&](int r, int c, int R, int G, int B) {
            for (int y = r * scale; y < (r + 1) * scale; ++y) {
                for (int x = c * scale; x < (c + 1) * scale; ++x) {
                    int idx = (y * w + x) * 3;
                    bool grid_line = (x % scale < 2) || (y % scale < 2);
                    img[idx + 0] = static_cast<unsigned char>(grid_line ? 120 : R);
                    img[idx + 1] = static_cast<unsigned char>(grid_line ? 120 : G);
                    img[idx + 2] = static_cast<unsigned char>(grid_line ? 120 : B);
                }
            }
        };

        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                int v = observed_map_[r][c];

                if (v == -1) {
                    paint(r, c, 170, 170, 170);
                } else if (v == 1) {
                    paint(r, c, 0, 0, 0);
                } else if (v == 2) {
                    paint(r, c, 255, 0, 0);
                } else {
                    paint(r, c, 255, 255, 255);
                }
            }
        }

        for (const Pos& p : visited_) {
            paint(p.r, p.c, 110, 180, 255);
        }

        for (const Pos& p : trajectory_) {
            paint(p.r, p.c, 255, 220, 0);
        }

        if (!trajectory_.empty()) {
            paint(trajectory_.front().r, trajectory_.front().c, 180, 0, 255);
        }

        paint(robot_.r, robot_.c, 0, 255, 0);

        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(img.data()), img.size());
    }

private:
    bool inBounds(Pos p) const {
        return p.r >= 0 && p.r < rows_ && p.c >= 0 && p.c < cols_;
    }

    bool isObstacleHidden(Pos p) const {
        return !inBounds(p) || hidden_map_[p.r][p.c] == 1;
    }

    bool isFreeObserved(Pos p) const {
        return inBounds(p) && (observed_map_[p.r][p.c] == 0 || observed_map_[p.r][p.c] == 2);
    }

    std::vector<Pos> neighbors4(Pos p) const {
        return {
            Pos{p.r - 1, p.c},
            Pos{p.r, p.c + 1},
            Pos{p.r + 1, p.c},
            Pos{p.r, p.c - 1}
        };
    }

    void observeLocalPatch() {
        for (int dr = -vision_radius_; dr <= vision_radius_; ++dr) {
            for (int dc = -vision_radius_; dc <= vision_radius_; ++dc) {
                Pos p{robot_.r + dr, robot_.c + dc};
                if (inBounds(p)) {
                    observed_map_[p.r][p.c] = hidden_map_[p.r][p.c];
                }
            }
        }
    }

    bool targetVisibleInLocalPatch() const {
        for (int dr = -vision_radius_; dr <= vision_radius_; ++dr) {
            for (int dc = -vision_radius_; dc <= vision_radius_; ++dc) {
                Pos p{robot_.r + dr, robot_.c + dc};
                if (inBounds(p) && observed_map_[p.r][p.c] == 2) {
                    return true;
                }
            }
        }
        return false;
    }

    bool isFrontier(Pos p) const {
        if (!isFreeObserved(p)) {
            return false;
        }

        for (Pos n : neighbors4(p)) {
            if (inBounds(n) && observed_map_[n.r][n.c] == -1) {
                return true;
            }
        }
        return false;
    }

    void buildObservedGraph() {
        node_id_.clear();
        observed_edges_ = 0;

        int id = 0;
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                Pos p{r, c};
                if (isFreeObserved(p)) {
                    node_id_[p] = id++;
                }
            }
        }

        for (const auto& kv : node_id_) {
            Pos p = kv.first;
            for (Pos n : neighbors4(p)) {
                if (node_id_.count(n) > 0) {
                    observed_edges_++;
                }
            }
        }
    }

    std::vector<float> buildFeature(int prev_action) const {
        std::vector<float> x;
        x.reserve(32);

        for (int dr = -2; dr <= 2; ++dr) {
            for (int dc = -2; dc <= 2; ++dc) {
                Pos p{robot_.r + dr, robot_.c + dc};
                if (!inBounds(p)) {
                    x.push_back(-1.0f);
                } else {
                    x.push_back(static_cast<float>(observed_map_[p.r][p.c]));
                }
            }
        }

        int frontier_count = 0;
        for (const auto& kv : node_id_) {
            Pos p = kv.first;
            for (Pos n : neighbors4(p)) {
                if (inBounds(n) && observed_map_[n.r][n.c] == -1) {
                    frontier_count++;
                }
            }
        }

        x.push_back(static_cast<float>(robot_.r));
        x.push_back(static_cast<float>(robot_.c));
        x.push_back(static_cast<float>(prev_action));
        x.push_back(static_cast<float>(node_id_.size()));
        x.push_back(static_cast<float>(observed_edges_));
        x.push_back(static_cast<float>(frontier_count));
        x.push_back(static_cast<float>(visited_.size()));

        return x;
    }

    Pos actionToNext(int action) const {
        if (action == 0) {
            return Pos{robot_.r - 1, robot_.c};
        }
        if (action == 1) {
            return Pos{robot_.r, robot_.c + 1};
        }
        if (action == 2) {
            return Pos{robot_.r + 1, robot_.c};
        }
        if (action == 3) {
            return Pos{robot_.r, robot_.c - 1};
        }
        return robot_;
    }

    bool actionValid(int action) const {
        if (action < 0 || action > 3) {
            return false;
        }

        Pos n = actionToNext(action);
        return isFreeObserved(n) && !isObstacleHidden(n);
    }

    bool actionLoops(int action) const {
        if (action < 0 || action > 3) {
            return true;
        }

        Pos n = actionToNext(action);
        if (!inBounds(n) || !isFreeObserved(n)) {
            return true;
        }

        // Softened loop handling: do not block revisits absolutely.
        // Only treat a very heavily revisited immediate move as a loop.
        return visit_count_[n.r][n.c] >= 6;
    }

    int chooseHybridAction(int policy_action, int prev_action) {
        if (targetVisibleInLocalPatch()) {
            return 4;
        }

        if (planner_mode_ == "policy_only") {
            if (actionValid(policy_action) && !actionLoops(policy_action)) {
                return policy_action;
            }
            fallback_count_++;
            return nearestFrontierAction();
        }

        if (planner_mode_ == "nearest") {
            if (actionValid(policy_action) && visitCountOfAction(policy_action) == 0) {
                return policy_action;
            }
            fallback_count_++;
            return nearestFrontierAction();
        }

        // In utility modes, the policy remains useful for local motion, but
        // repeated/low-information moves are redirected by utility frontier selection.
        if (actionValid(policy_action) && !actionLoops(policy_action) && visitCountOfAction(policy_action) == 0) {
            return policy_action;
        }

        fallback_count_++;
        return selectFrontierAction(prev_action);
    }

    int visitCountOfAction(int action) const {
        if (action < 0 || action > 3) {
            return 999;
        }
        Pos n = actionToNext(action);
        if (!inBounds(n)) {
            return 999;
        }
        return visit_count_[n.r][n.c];
    }

    int selectFrontierAction(int prev_action) {
        if (planner_mode_ == "nearest") {
            return nearestFrontierAction();
        }
        if (planner_mode_ == "utility_commit") {
            if (has_subgoal_ && commitment_remaining_ > 0 && isFreeObserved(current_subgoal_)) {
                int a = firstActionTo(current_subgoal_);
                if (a >= 0 && a < 4) {
                    commitment_remaining_--;
                    return a;
                }
            }
        }

        Pos best;
        if (selectBestFrontier(best, prev_action)) {
            if (!has_subgoal_ || best != current_subgoal_) {
                frontier_switches_++;
            }
            current_subgoal_ = best;
            has_subgoal_ = true;
            commitment_remaining_ = (planner_mode_ == "utility_commit") ? 4 : 0;
            int a = firstActionTo(best);
            if (a >= 0 && a < 4) {
                return a;
            }
        }

        return nearestFrontierAction();
    }

    int nearestFrontierAction() const {
        std::queue<Pos> q;
        std::set<Pos> seen;
        std::map<Pos, Pos> parent;

        q.push(robot_);
        seen.insert(robot_);

        bool found = false;
        Pos frontier;

        while (!q.empty()) {
            Pos cur = q.front();
            q.pop();

            if (cur != robot_ && isFrontier(cur)) {
                frontier = cur;
                found = true;
                break;
            }

            for (Pos n : neighbors4(cur)) {
                if (!isFreeObserved(n) || seen.count(n) > 0) {
                    continue;
                }
                seen.insert(n);
                parent[n] = cur;
                q.push(n);
            }
        }

        if (found) {
            Pos cur = frontier;
            while (parent.count(cur) > 0 && parent.at(cur) != robot_) {
                cur = parent.at(cur);
            }
            return actionFromTo(robot_, cur);
        }

        for (int a = 0; a < 4; ++a) {
            Pos n = actionToNext(a);
            if (isFreeObserved(n) && visit_count_[n.r][n.c] == 0) {
                return a;
            }
        }

        for (int a = 0; a < 4; ++a) {
            Pos n = actionToNext(a);
            if (isFreeObserved(n)) {
                return a;
            }
        }

        return 4;
    }

    std::vector<Pos> collectFrontiers() const {
        std::vector<Pos> frontiers;
        for (const auto& kv : node_id_) {
            Pos p = kv.first;
            if (p != robot_ && isFrontier(p)) {
                frontiers.push_back(p);
            }
        }
        return frontiers;
    }

    int unknownCountAround(Pos p, int radius) const {
        int cnt = 0;
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                Pos q{p.r + dr, p.c + dc};
                if (inBounds(q) && observed_map_[q.r][q.c] == -1) {
                    cnt++;
                }
            }
        }
        return cnt;
    }

    int freeDegreeObserved(Pos p) const {
        int degree = 0;
        for (Pos n : neighbors4(p)) {
            if (isFreeObserved(n)) {
                degree++;
            }
        }
        return degree;
    }

    float localVisitPenalty(Pos p, int radius = 1) const {
        float penalty = 0.0f;
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                Pos q{p.r + dr, p.c + dc};
                if (inBounds(q)) {
                    penalty += static_cast<float>(std::min(visit_count_[q.r][q.c], 5));
                }
            }
        }
        return penalty;
    }

    float directionCost(Pos target, int prev_action) const {
        if (prev_action < 0 || prev_action > 3) {
            return 0.0f;
        }
        int best_dir = 4;
        int dr = target.r - robot_.r;
        int dc = target.c - robot_.c;
        if (std::abs(dr) > std::abs(dc)) {
            best_dir = dr < 0 ? 0 : 2;
        } else if (dc != 0) {
            best_dir = dc > 0 ? 1 : 3;
        }
        return best_dir == prev_action ? 0.0f : 1.0f;
    }

    int bfsDistance(Pos start, Pos goal) const {
        std::queue<Pos> q;
        std::map<Pos, int> dist;
        q.push(start);
        dist[start] = 0;
        while (!q.empty()) {
            Pos cur = q.front();
            q.pop();
            if (cur == goal) {
                return dist[cur];
            }
            for (Pos n : neighbors4(cur)) {
                if (!isFreeObserved(n) || dist.count(n) > 0) {
                    continue;
                }
                dist[n] = dist[cur] + 1;
                q.push(n);
            }
        }
        return 1000000;
    }

    float frontierScore(Pos f, int prev_action) const {
        const float travel_cost = static_cast<float>(bfsDistance(robot_, f));
        if (travel_cost >= 1000000.0f) {
            return -1e9f;
        }

        const float information_gain = static_cast<float>(unknownCountAround(f, vision_radius_ + 1));
        const float prior_map_score = static_cast<float>(unknownCountAround(f, 3));
        const float revisit_penalty = localVisitPenalty(f, 1);
        const float deadend_penalty = freeDegreeObserved(f) <= 1 ? 1.0f : 0.0f;
        const float turn_cost = directionCost(f, prev_action);

        // 2D-grid proxy for semantic target likelihood:
        // if the current partial map has expanded in one direction, prefer frontier
        // candidates that continue opening more unknown area rather than near dead ends.
        const float semantic_proxy = information_gain / 10.0f;

        return
            1.0f * semantic_proxy +
            0.8f * information_gain +
            0.6f * prior_map_score -
            1.4f * travel_cost -
            2.6f * revisit_penalty -
            1.4f * deadend_penalty -
            0.6f * turn_cost;
    }

    bool selectBestFrontier(Pos& best, int prev_action) const {
        std::vector<Pos> frontiers = collectFrontiers();
        float best_score = -1e9f;
        bool found = false;
        for (Pos f : frontiers) {
            float s = frontierScore(f, prev_action);
            if (s > best_score) {
                best_score = s;
                best = f;
                found = true;
            }
        }
        return found;
    }

    int firstActionTo(Pos goal) const {
        if (goal == robot_) {
            return 4;
        }

        std::queue<Pos> q;
        std::set<Pos> seen;
        std::map<Pos, Pos> parent;
        q.push(robot_);
        seen.insert(robot_);

        bool found = false;
        while (!q.empty()) {
            Pos cur = q.front();
            q.pop();
            if (cur == goal) {
                found = true;
                break;
            }
            for (Pos n : neighbors4(cur)) {
                if (!isFreeObserved(n) || seen.count(n) > 0) {
                    continue;
                }
                seen.insert(n);
                parent[n] = cur;
                q.push(n);
            }
        }

        if (!found) {
            return 4;
        }

        Pos cur = goal;
        while (parent.count(cur) > 0 && parent.at(cur) != robot_) {
            cur = parent.at(cur);
        }
        return actionFromTo(robot_, cur);
    }

    int actionFromTo(Pos from, Pos to) const {
        int dr = to.r - from.r;
        int dc = to.c - from.c;

        if (dr == -1 && dc == 0) {
            return 0;
        }
        if (dr == 0 && dc == 1) {
            return 1;
        }
        if (dr == 1 && dc == 0) {
            return 2;
        }
        if (dr == 0 && dc == -1) {
            return 3;
        }

        return 4;
    }

    bool move(Pos next) {
        if (isObstacleHidden(next)) {
            return false;
        }

        robot_ = next;
        return true;
    }

    GridPolicy policy_;
    std::string planner_mode_ = "utility";

    std::vector<std::vector<int>> hidden_map_;
    std::vector<std::vector<int>> observed_map_;
    std::vector<std::vector<int>> visit_count_;

    int rows_ = 0;
    int cols_ = 0;
    int max_steps_ = 200;
    int vision_radius_ = 2;

    Pos robot_;
    std::set<Pos> visited_;
    std::vector<Pos> trajectory_;

    std::map<Pos, int> node_id_;
    int observed_edges_ = 0;
    int fallback_count_ = 0;
    int frontier_switches_ = 0;
    int commitment_remaining_ = 0;
    bool has_subgoal_ = false;
    Pos current_subgoal_;
};

int main(int argc, char** argv) {
    std::string map_path = "maps/grid_unseen/unseen_001.csv";
    std::string model_path = "models/grid_ssm_policy.onnx";

    if (argc > 1) {
        map_path = argv[1];
    }
    std::string planner_mode = "utility";
    int max_steps = 200;
    int vision_radius = 2;
    Pos start{14, 0};
    std::string out_dir = "results/grid_sim";

    if (argc > 2) {
        model_path = argv[2];
    }
    if (argc > 3) {
        planner_mode = argv[3];
    }
    if (argc > 4) {
        max_steps = std::stoi(argv[4]);
    }
    if (argc > 5) {
        vision_radius = std::stoi(argv[5]);
    }
    if (argc > 7) {
        start = Pos{std::stoi(argv[6]), std::stoi(argv[7])};
    }
    if (argc > 8) {
        out_dir = argv[8];
    }

    GridPolicyEval eval(model_path, planner_mode);

    if (!eval.loadHiddenMap(map_path)) {
        return 1;
    }

    Metrics m = eval.run(start, max_steps, vision_radius);
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    std::system(mkdir_cmd.c_str());

    eval.saveTrajectory(out_dir + "/policy_trajectory.csv");
    eval.saveObservedMapCsv(out_dir + "/policy_observed_map.csv");
    eval.saveImagePpm(out_dir + "/policy_path.ppm");

    std::ofstream metrics(out_dir + "/policy_metrics.csv");
    metrics << "success,steps,collisions,revisits,observed_nodes,observed_edges,target_seen_step,fallback_count\n";
    metrics << m.success << ","
            << m.steps << ","
            << m.collisions << ","
            << m.revisits << ","
            << m.observed_nodes << ","
            << m.observed_edges << ","
            << m.target_seen_step << ","
            << m.fallback_count << "\n";

    std::cout << "Grid hybrid policy eval finished.\n";
    std::cout << "SSM policy + observed graph frontier fallback.\n";
    std::cout << "No A*, no goal coordinate, no full-map planning at runtime.\n";
    std::cout << "success=" << m.success
              << " steps=" << m.steps
              << " collisions=" << m.collisions
              << " revisits=" << m.revisits
              << " revisit_ratio=" << m.revisit_ratio
              << " unique_visited=" << m.unique_visited
              << " observed_nodes=" << m.observed_nodes
              << " observed_edges=" << m.observed_edges
              << " target_seen_step=" << m.target_seen_step
              << " fallback_count=" << m.fallback_count
              << " frontier_switches=" << m.frontier_switches
              << std::endl;

    return 0;
}
