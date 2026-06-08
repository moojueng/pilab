#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct Pos {
    int r = 0;
    int c = 0;

    bool operator<(const Pos& other) const {
        return std::tie(r, c) < std::tie(other.r, other.c);
    }

    bool operator==(const Pos& other) const {
        return r == other.r && c == other.c;
    }
};

struct GraphEdge {
    int from = -1;
    int to = -1;
};

struct SimResult {
    bool success = false;
    int steps = 0;
    int collisions = 0;
    int revisits = 0;
    int observed_nodes = 0;
    int observed_edges = 0;
};

class GridWorldSim {
public:
    bool loadMap(const std::string& path) {
        map_path_ = path;
        grid_.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open map: " << path << std::endl;
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

            if (!row.empty()) {
                grid_.push_back(row);
            }
        }

        rows_ = static_cast<int>(grid_.size());
        cols_ = rows_ > 0 ? static_cast<int>(grid_[0].size()) : 0;

        return rows_ > 0 && cols_ > 0;
    }

    SimResult run(const Pos& start, int max_steps, int vision_radius) {
        robot_ = start;
        heading_ = 1;  // 0 up, 1 right, 2 down, 3 left
        max_steps_ = max_steps;
        vision_radius_ = vision_radius;

        visited_.clear();
        observed_.clear();
        node_id_.clear();
        edges_.clear();
        trajectory_.clear();

        SimResult result;

        for (int step = 0; step < max_steps; ++step) {
            observe();
            rebuildGraph();
            trajectory_.push_back(robot_);

            if (targetVisible()) {
                result.success = true;
                result.steps = step;
                break;
            }

            int action = chooseFrontierAction();
            bool moved = applyAction(action);

            if (!moved) {
                result.collisions++;
            }

            if (visited_.count(robot_) > 0) {
                result.revisits++;
            }

            visited_.insert(robot_);
        }

        if (!result.success) {
            result.steps = max_steps;
        }

        result.observed_nodes = static_cast<int>(node_id_.size());
        result.observed_edges = static_cast<int>(edges_.size());

        return result;
    }

    void saveTrajectory(const std::string& path) const {
        std::ofstream file(path);
        file << "step,r,c\n";
        for (std::size_t i = 0; i < trajectory_.size(); ++i) {
            file << i << "," << trajectory_[i].r << "," << trajectory_[i].c << "\n";
        }
    }

    void saveObservedGraph(const std::string& path) const {
        std::ofstream file(path);
        file << "node_id,r,c\n";
        for (const auto& kv : node_id_) {
            file << kv.second << "," << kv.first.r << "," << kv.first.c << "\n";
        }

        file << "\nedges\n";
        file << "from,to\n";
        for (const auto& e : edges_) {
            file << e.from << "," << e.to << "\n";
        }
    }

    void savePpmImage(const std::string& path) const {
        const int scale = 40;
        const int width = cols_ * scale;
        const int height = rows_ * scale;

        std::vector<unsigned char> image(width * height * 3, 255);

        auto paintCell = [&](int r, int c, int red, int green, int blue) {
            for (int y = r * scale; y < (r + 1) * scale; ++y) {
                for (int x = c * scale; x < (c + 1) * scale; ++x) {
                    int idx = (y * width + x) * 3;
                    image[idx + 0] = static_cast<unsigned char>(red);
                    image[idx + 1] = static_cast<unsigned char>(green);
                    image[idx + 2] = static_cast<unsigned char>(blue);
                }
            }
        };

        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                Pos p{r, c};
                if (observed_.count(p) == 0) {
                    paintCell(r, c, 180, 180, 180);
                } else if (grid_[r][c] == 1) {
                    paintCell(r, c, 0, 0, 0);
                } else if (grid_[r][c] == 2) {
                    paintCell(r, c, 255, 0, 0);
                } else {
                    paintCell(r, c, 255, 255, 255);
                }
            }
        }

        for (const Pos& p : visited_) {
            paintCell(p.r, p.c, 120, 180, 255);
        }

        for (const Pos& p : trajectory_) {
            paintCell(p.r, p.c, 255, 220, 0);
        }

        paintCell(robot_.r, robot_.c, 0, 255, 0);

        std::ofstream file(path, std::ios::binary);
        file << "P6\n" << width << " " << height << "\n255\n";
        file.write(reinterpret_cast<const char*>(image.data()), image.size());
    }

private:
    bool inBounds(const Pos& p) const {
        return p.r >= 0 && p.r < rows_ && p.c >= 0 && p.c < cols_;
    }

    bool isFreeLike(const Pos& p) const {
        if (!inBounds(p)) {
            return false;
        }
        return grid_[p.r][p.c] == 0 || grid_[p.r][p.c] == 2;
    }

    void observe() {
        for (int dr = -vision_radius_; dr <= vision_radius_; ++dr) {
            for (int dc = -vision_radius_; dc <= vision_radius_; ++dc) {
                Pos p{robot_.r + dr, robot_.c + dc};
                if (inBounds(p)) {
                    observed_.insert(p);
                }
            }
        }
    }

    bool targetVisible() const {
        for (const Pos& p : observed_) {
            if (std::abs(p.r - robot_.r) <= vision_radius_ &&
                std::abs(p.c - robot_.c) <= vision_radius_ &&
                grid_[p.r][p.c] == 2) {
                return true;
            }
        }
        return false;
    }

    std::vector<Pos> neighbors4(const Pos& p) const {
        return {
            Pos{p.r - 1, p.c},
            Pos{p.r, p.c + 1},
            Pos{p.r + 1, p.c},
            Pos{p.r, p.c - 1}
        };
    }

    void rebuildGraph() {
        node_id_.clear();
        edges_.clear();

        int next_id = 0;
        for (const Pos& p : observed_) {
            if (isFreeLike(p)) {
                node_id_[p] = next_id++;
            }
        }

        for (const auto& kv : node_id_) {
            Pos p = kv.first;
            int from = kv.second;

            for (const Pos& n : neighbors4(p)) {
                auto it = node_id_.find(n);
                if (it != node_id_.end()) {
                    edges_.push_back(GraphEdge{from, it->second});
                }
            }
        }
    }

    int chooseFrontierAction() {
        std::vector<Pos> dirs = neighbors4(robot_);

        int best_action = 0;
        int best_score = -100000;

        for (int action = 0; action < 4; ++action) {
            Pos n = dirs[action];

            int score = 0;
            if (!isFreeLike(n)) {
                score -= 1000;
            }

            if (visited_.count(n) == 0) {
                score += 50;
            } else {
                score -= 20;
            }

            for (const Pos& nn : neighbors4(n)) {
                if (inBounds(nn) && observed_.count(nn) == 0) {
                    score += 10;
                }
            }

            if (score > best_score) {
                best_score = score;
                best_action = action;
            }
        }

        return best_action;
    }

    bool applyAction(int action) {
        std::vector<Pos> dirs = neighbors4(robot_);
        Pos next = dirs[action];

        heading_ = action;

        if (!isFreeLike(next)) {
            return false;
        }

        robot_ = next;
        return true;
    }

    std::string map_path_;
    std::vector<std::vector<int>> grid_;

    int rows_ = 0;
    int cols_ = 0;
    int heading_ = 1;
    int max_steps_ = 200;
    int vision_radius_ = 2;

    Pos robot_;
    std::set<Pos> visited_;
    std::set<Pos> observed_;
    std::vector<Pos> trajectory_;

    std::map<Pos, int> node_id_;
    std::vector<GraphEdge> edges_;
};

int main(int argc, char** argv) {
    std::string map_path = "maps/unseen/unseen_map_01.csv";
    std::string out_dir = "results/grid_sim";
    int start_r = 4;
    int start_c = 0;
    int max_steps = 200;
    int vision_radius = 2;

    if (argc > 1) {
        map_path = argv[1];
    }

    GridWorldSim sim;
    if (!sim.loadMap(map_path)) {
        return 1;
    }

    SimResult result = sim.run(Pos{start_r, start_c}, max_steps, vision_radius);

    sim.saveTrajectory(out_dir + "/trajectory_unseen.csv");
    sim.saveObservedGraph(out_dir + "/observed_graph_unseen.csv");
    sim.savePpmImage(out_dir + "/path_unseen.ppm");

    std::ofstream metrics(out_dir + "/metrics_unseen.csv");
    metrics << "success,steps,collisions,revisits,observed_nodes,observed_edges\n";
    metrics << result.success << ","
            << result.steps << ","
            << result.collisions << ","
            << result.revisits << ","
            << result.observed_nodes << ","
            << result.observed_edges << "\n";

    std::cout << "Grid sim finished.\n";
    std::cout << "success=" << result.success
              << " steps=" << result.steps
              << " collisions=" << result.collisions
              << " revisits=" << result.revisits
              << " observed_nodes=" << result.observed_nodes
              << " observed_edges=" << result.observed_edges
              << std::endl;

    return 0;
}
