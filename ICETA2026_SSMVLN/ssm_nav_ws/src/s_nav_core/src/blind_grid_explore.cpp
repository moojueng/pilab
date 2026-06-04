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
    int observed_free_nodes = 0;
    int observed_edges = 0;
    int target_seen_step = -1;
};

class BlindGridExplorer {
public:
    bool loadHiddenMap(const std::string& path) {
        hidden_map_.clear();

        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "failed to open map: " << path << "\n";
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

        return rows_ > 0 && cols_ > 0;
    }

    Metrics run(Pos start, int max_steps, int vision_radius) {
        robot_ = start;
        start_ = start;
        max_steps_ = max_steps;
        vision_radius_ = vision_radius;

        trajectory_.clear();
        visited_.clear();
        node_id_.clear();
        observed_edges_ = 0;
        collisions_ = 0;
        revisits_ = 0;

        Metrics m;

        if (!inBounds(robot_) || isObstacleHidden(robot_)) {
            std::cerr << "invalid start position\n";
            m.collisions = 1;
            return m;
        }

        for (int step = 0; step < max_steps_; ++step) {
            observeLocalPatch();
            buildObservedGraph();
            trajectory_.push_back(robot_);

            if (targetVisibleInLocalPatch()) {
                m.success = true;
                m.steps = step;
                m.target_seen_step = step;
                break;
            }

            Pos next;
            bool found = chooseNextFromObservedGraph(next);

            if (!found) {
                m.steps = step;
                break;
            }

            if (!move(next)) {
                collisions_++;
            }

            if (visited_.count(robot_) > 0) {
                revisits_++;
            }

            visited_.insert(robot_);
        }

        if (!m.success && m.steps == 0) {
            m.steps = max_steps_;
        }

        m.collisions = collisions_;
        m.revisits = revisits_;
        m.observed_free_nodes = static_cast<int>(node_id_.size());
        m.observed_edges = observed_edges_;

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
        const int scale = 60;
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

        paint(start_.r, start_.c, 180, 0, 255);
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
        if (!inBounds(p)) {
            return false;
        }
        return observed_map_[p.r][p.c] == 0 || observed_map_[p.r][p.c] == 2;
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

    bool chooseNextFromObservedGraph(Pos& next_out) {
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
                if (!isFreeObserved(n)) {
                    continue;
                }

                if (seen.count(n) > 0) {
                    continue;
                }

                seen.insert(n);
                parent[n] = cur;
                q.push(n);
            }
        }

        if (!found) {
            return chooseUnvisitedObservedNeighbor(next_out);
        }

        Pos cur = frontier;
        while (parent.count(cur) > 0 && parent[cur] != robot_) {
            cur = parent[cur];
        }

        next_out = cur;
        return true;
    }

    bool chooseUnvisitedObservedNeighbor(Pos& next_out) const {
        for (Pos n : neighbors4(robot_)) {
            if (isFreeObserved(n) && visited_.count(n) == 0) {
                next_out = n;
                return true;
            }
        }

        for (Pos n : neighbors4(robot_)) {
            if (isFreeObserved(n)) {
                next_out = n;
                return true;
            }
        }

        return false;
    }

    bool move(Pos next) {
        if (isObstacleHidden(next)) {
            return false;
        }

        robot_ = next;
        return true;
    }

    std::vector<std::vector<int>> hidden_map_;
    std::vector<std::vector<int>> observed_map_;

    int rows_ = 0;
    int cols_ = 0;
    int max_steps_ = 200;
    int vision_radius_ = 1;

    Pos robot_;
    Pos start_;
    std::set<Pos> visited_;
    std::vector<Pos> trajectory_;

    std::map<Pos, int> node_id_;
    int observed_edges_ = 0;
    int collisions_ = 0;
    int revisits_ = 0;
};

int main(int argc, char** argv) {
    std::string map_path = "maps/unseen/unseen_map_01.csv";
    if (argc > 1) {
        map_path = argv[1];
    }

    BlindGridExplorer sim;
    if (!sim.loadHiddenMap(map_path)) {
        return 1;
    }

    Pos start{4, 0};
    Metrics m = sim.run(start, 200, 1);

    std::string out_dir = "results/grid_sim";
    sim.saveTrajectory(out_dir + "/blind_trajectory.csv");
    sim.saveObservedMapCsv(out_dir + "/blind_observed_map.csv");
    sim.saveImagePpm(out_dir + "/blind_path.ppm");

    std::ofstream metrics(out_dir + "/blind_metrics.csv");
    metrics << "success,steps,collisions,revisits,observed_free_nodes,observed_edges,target_seen_step\n";
    metrics << m.success << ","
            << m.steps << ","
            << m.collisions << ","
            << m.revisits << ","
            << m.observed_free_nodes << ","
            << m.observed_edges << ","
            << m.target_seen_step << "\n";

    std::cout << "Blind 2D grid exploration finished.\n";
    std::cout << "No A*, no goal coordinate, no full-map planning at runtime.\n";
    std::cout << "success=" << m.success
              << " steps=" << m.steps
              << " collisions=" << m.collisions
              << " revisits=" << m.revisits
              << " observed_nodes=" << m.observed_free_nodes
              << " observed_edges=" << m.observed_edges
              << " target_seen_step=" << m.target_seen_step
              << std::endl;

    return 0;
}
