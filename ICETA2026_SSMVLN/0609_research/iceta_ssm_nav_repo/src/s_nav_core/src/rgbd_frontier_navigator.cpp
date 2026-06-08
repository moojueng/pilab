#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

class RgbdFrontierNavigator : public rclcpp::Node {
public:
    RgbdFrontierNavigator() : Node("rgbd_frontier_navigator") {
        this->declare_parameter("voxel_csv_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/observed_voxels.csv");
        this->declare_parameter("trajectory_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/trajectory.csv");
        this->declare_parameter("metrics_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/metrics.csv");
        this->declare_parameter("camera_debug_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/latest_camera.jpg");
        this->declare_parameter("graph_nodes_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/runtime_graph_nodes.csv");
        this->declare_parameter("graph_edges_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/runtime_graph_edges.csv");
        this->declare_parameter("frontier_features_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/frontier_features.csv");
        this->declare_parameter("target_events_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/target_events.csv");
        this->declare_parameter("policy_mode", "heuristic");
        this->declare_parameter("mission_mode", "coverage_patrol");
        this->declare_parameter("odom_topic", "/odom");
        this->declare_parameter("cmd_vel_topic", "/cmd_vel");
        this->declare_parameter("camera_topic", "/camera/image_raw");
        this->declare_parameter("depth_topic", "/camera/depth/image_raw");
        this->declare_parameter("target_goal", "chair");

        this->declare_parameter("voxel_size", 0.20);
        this->declare_parameter("min_x", -4.0);
        this->declare_parameter("max_x", 7.0);
        this->declare_parameter("min_y", -4.5);
        this->declare_parameter("max_y", 4.5);
        this->declare_parameter("min_z", 0.0);
        this->declare_parameter("max_z", 2.4);
        this->declare_parameter("obstacle_min_z", 0.45);
        this->declare_parameter("robot_clearance_z", 0.55);
        this->declare_parameter("target_red_ratio_threshold", 0.010);
        this->declare_parameter("target_candidate_ratio_threshold", 0.006);
        this->declare_parameter("target_confirm_ratio_threshold", 0.055);
        this->declare_parameter("target_confirm_max_distance", 0.55);
        this->declare_parameter("target_center_tolerance", 0.18);
        this->declare_parameter("min_frontier_distance", 0.45);
        this->declare_parameter("goal_tolerance", 0.12);
        this->declare_parameter("linear_speed", 0.11);
        this->declare_parameter("angular_speed", 0.55);
        this->declare_parameter("enable_local_avoidance", false);
        this->declare_parameter("stop_on_target", false);
        this->declare_parameter("approach_target_candidate", false);
        this->declare_parameter("enable_target_direction_prior", false);
        this->declare_parameter("event_log_period_sec", 2.0);
        this->declare_parameter("obstacle_stop_distance", 0.28);
        this->declare_parameter("entry_x", 1.5);
        this->declare_parameter("entry_y", -2.65);
        this->declare_parameter("enable_entry_behavior", true);
        this->declare_parameter("control_period_ms", 250);
        this->declare_parameter("max_runtime_sec", 240.0);

        voxel_csv_path_ = this->get_parameter("voxel_csv_path").as_string();
        trajectory_path_ = this->get_parameter("trajectory_path").as_string();
        metrics_path_ = this->get_parameter("metrics_path").as_string();
        camera_debug_path_ = this->get_parameter("camera_debug_path").as_string();
        graph_nodes_path_ = this->get_parameter("graph_nodes_path").as_string();
        graph_edges_path_ = this->get_parameter("graph_edges_path").as_string();
        frontier_features_path_ = this->get_parameter("frontier_features_path").as_string();
        target_events_path_ = this->get_parameter("target_events_path").as_string();
        policy_mode_ = this->get_parameter("policy_mode").as_string();
        mission_mode_ = this->get_parameter("mission_mode").as_string();
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
        camera_topic_ = this->get_parameter("camera_topic").as_string();
        depth_topic_ = this->get_parameter("depth_topic").as_string();
        target_goal_ = normalize_goal(this->get_parameter("target_goal").as_string());

        voxel_size_ = this->get_parameter("voxel_size").as_double();
        min_x_ = this->get_parameter("min_x").as_double();
        max_x_ = this->get_parameter("max_x").as_double();
        min_y_ = this->get_parameter("min_y").as_double();
        max_y_ = this->get_parameter("max_y").as_double();
        min_z_ = this->get_parameter("min_z").as_double();
        max_z_ = this->get_parameter("max_z").as_double();
        obstacle_min_z_ = this->get_parameter("obstacle_min_z").as_double();
        robot_clearance_z_ = this->get_parameter("robot_clearance_z").as_double();
        target_red_ratio_threshold_ = this->get_parameter("target_red_ratio_threshold").as_double();
        target_candidate_ratio_threshold_ = this->get_parameter("target_candidate_ratio_threshold").as_double();
        target_confirm_ratio_threshold_ = this->get_parameter("target_confirm_ratio_threshold").as_double();
        target_confirm_max_distance_ = this->get_parameter("target_confirm_max_distance").as_double();
        target_center_tolerance_ = this->get_parameter("target_center_tolerance").as_double();
        min_frontier_distance_ = this->get_parameter("min_frontier_distance").as_double();
        goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();
        linear_speed_ = this->get_parameter("linear_speed").as_double();
        angular_speed_ = this->get_parameter("angular_speed").as_double();
        enable_local_avoidance_ = this->get_parameter("enable_local_avoidance").as_bool();
        stop_on_target_ = this->get_parameter("stop_on_target").as_bool();
        approach_target_candidate_ = this->get_parameter("approach_target_candidate").as_bool();
        enable_target_direction_prior_ = this->get_parameter("enable_target_direction_prior").as_bool();
        event_log_period_sec_ = this->get_parameter("event_log_period_sec").as_double();
        obstacle_stop_distance_ = this->get_parameter("obstacle_stop_distance").as_double();
        entry_x_ = this->get_parameter("entry_x").as_double();
        entry_y_ = this->get_parameter("entry_y").as_double();
        enable_entry_behavior_ = this->get_parameter("enable_entry_behavior").as_bool();
        max_runtime_sec_ = this->get_parameter("max_runtime_sec").as_double();

        cols_ = std::max(1, static_cast<int>(std::ceil((max_x_ - min_x_) / voxel_size_)));
        rows_ = std::max(1, static_cast<int>(std::ceil((max_y_ - min_y_) / voxel_size_)));

        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10,
            std::bind(&RgbdFrontierNavigator::odom_callback, this, std::placeholders::_1));
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic_, 10,
            std::bind(&RgbdFrontierNavigator::image_callback, this, std::placeholders::_1));
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            depth_topic_, 10,
            std::bind(&RgbdFrontierNavigator::depth_callback, this, std::placeholders::_1));

        const int period_ms = std::max(50, static_cast<int>(this->get_parameter("control_period_ms").as_int()));
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&RgbdFrontierNavigator::control_loop, this));

        start_time_ = this->now();
        write_metrics(false);

        RCLCPP_INFO(
            this->get_logger(),
            "RGB-D frontier navigator started | voxel_csv=%s cmd=%s goal=%s mission=%s stop_on_target=%d",
            voxel_csv_path_.c_str(),
            cmd_vel_topic_.c_str(),
            target_goal_.c_str(),
            mission_mode_.c_str(),
            stop_on_target_);
    }

private:
    struct Cell {
        int ix = 0;
        int iy = 0;

        bool operator<(const Cell& o) const {
            return std::tie(ix, iy) < std::tie(o.ix, o.iy);
        }

        bool operator==(const Cell& o) const {
            return ix == o.ix && iy == o.iy;
        }
    };

    struct Frontier {
        Cell cell;
        double x = 0.0;
        double y = 0.0;
        double score = 0.0;
        int unknown_gain = 0;
    };

    struct Projection {
        std::set<Cell> free;
        std::set<Cell> occupied;
        std::set<Cell> observed;
        int voxel_rows = 0;
    };

    static double normalize_angle(double angle) {
        while (angle > M_PI) {
            angle -= 2.0 * M_PI;
        }
        while (angle < -M_PI) {
            angle += 2.0 * M_PI;
        }
        return angle;
    }

    static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q) {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    static std::string normalize_goal(std::string goal) {
        std::transform(goal.begin(), goal.end(), goal.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (goal == "bedroom" || goal == "blue_bed" || goal == "blue bed" ||
            goal == "침대" || goal == "파란침대" || goal == "파란 침대") {
            return "bed";
        }
        if (goal == "red_chair" || goal == "red chair" ||
            goal == "의자" || goal == "빨간의자" || goal == "빨간 의자") {
            return "chair";
        }
        return goal;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_x_ = msg->pose.pose.position.x;
        current_y_ = msg->pose.pose.position.y;
        current_yaw_ = quaternion_to_yaw(msg->pose.pose.orientation);
        has_odom_ = true;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            const double now_sec = this->now().seconds();
            if (now_sec - last_camera_save_sec_ > 0.5) {
                const std::filesystem::path out_path(camera_debug_path_);
                if (out_path.has_parent_path()) {
                    std::filesystem::create_directories(out_path.parent_path());
                }
                cv::imwrite(camera_debug_path_, cv_ptr->image);
                last_camera_save_sec_ = now_sec;
            }

            cv::Mat hsv;
            cv::cvtColor(cv_ptr->image, hsv, cv::COLOR_BGR2HSV);

            cv::Mat red_low, red_high, blue_mask;
            cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(12, 255, 255), red_low);
            cv::inRange(hsv, cv::Scalar(170, 80, 80), cv::Scalar(180, 255, 255), red_high);
            cv::inRange(hsv, cv::Scalar(95, 70, 60), cv::Scalar(130, 255, 255), blue_mask);

            const cv::Mat red_mask = red_low | red_high;
            const double denom = static_cast<double>(cv_ptr->image.rows * cv_ptr->image.cols);
            red_ratio_ = static_cast<double>(cv::countNonZero(red_mask)) / denom;
            blue_ratio_ = static_cast<double>(cv::countNonZero(blue_mask)) / denom;
            cyan_ratio_ = 0.0;

            cv::Mat target_mask;
            if (is_bed_goal()) {
                target_mask = blue_mask;
                target_score_ = blue_ratio_;
            } else {
                target_mask = red_mask;
                target_score_ = red_ratio_;
            }

            const cv::Moments m = cv::moments(target_mask, true);
            if (m.m00 > 1.0) {
                target_center_x_ = (m.m10 / m.m00) / static_cast<double>(target_mask.cols);
            } else {
                target_center_x_ = 0.5;
            }
            target_candidate_visible_ = target_score_ > target_candidate_ratio_threshold_;
            target_visible_ = target_candidate_visible_ && target_close_enough();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "image callback failed: %s",
                e.what());
        }
    }

    static float depth_at(const sensor_msgs::msg::Image& msg, int u, int v) {
        const bool is_u16 =
            msg.encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
            msg.encoding == sensor_msgs::image_encodings::MONO16;
        const std::size_t bytes = is_u16 ? 2U : 4U;
        const std::size_t offset =
            static_cast<std::size_t>(v) * static_cast<std::size_t>(msg.step) +
            static_cast<std::size_t>(u) * bytes;
        if (offset + bytes > msg.data.size()) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (msg.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            float value = std::numeric_limits<float>::quiet_NaN();
            std::memcpy(&value, msg.data.data() + offset, sizeof(float));
            return value;
        }
        if (is_u16) {
            std::uint16_t value = 0;
            std::memcpy(&value, msg.data.data() + offset, sizeof(std::uint16_t));
            return static_cast<float>(value) * 0.001f;
        }
        return std::numeric_limits<float>::quiet_NaN();
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        const int width = static_cast<int>(msg->width);
        const int height = static_cast<int>(msg->height);
        if (width <= 0 || height <= 0) {
            return;
        }

        auto min_range = [&](double u0, double u1) {
            float best = std::numeric_limits<float>::infinity();
            const int x0 = std::max(0, static_cast<int>(width * u0));
            const int x1 = std::min(width - 1, static_cast<int>(width * u1));
            const int y0 = static_cast<int>(height * 0.18);
            const int y1 = static_cast<int>(height * 0.48);
            for (int v = y0; v <= y1; v += 6) {
                for (int u = x0; u <= x1; u += 6) {
                    const float d = depth_at(*msg, u, v);
                    if (std::isfinite(d) && d > 0.08f) {
                        best = std::min(best, d);
                    }
                }
            }
            return best;
        };

        left_clearance_ = min_range(0.08, 0.36);
        center_clearance_ = min_range(0.38, 0.62);
        right_clearance_ = min_range(0.64, 0.92);
        has_depth_ = true;
    }

    Cell world_to_cell(double x, double y) const {
        return Cell{
            static_cast<int>((x - min_x_) / voxel_size_),
            static_cast<int>((y - min_y_) / voxel_size_),
        };
    }

    double cell_x(int ix) const {
        return min_x_ + (static_cast<double>(ix) + 0.5) * voxel_size_;
    }

    double cell_y(int iy) const {
        return min_y_ + (static_cast<double>(iy) + 0.5) * voxel_size_;
    }

    bool in_bounds(const Cell& c) const {
        return c.ix >= 0 && c.ix < cols_ && c.iy >= 0 && c.iy < rows_;
    }

    std::vector<Cell> neighbors4(const Cell& c) const {
        return {
            Cell{c.ix + 1, c.iy},
            Cell{c.ix - 1, c.iy},
            Cell{c.ix, c.iy + 1},
            Cell{c.ix, c.iy - 1},
        };
    }

    std::vector<Cell> neighbors8(const Cell& c) const {
        std::vector<Cell> out;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                out.push_back(Cell{c.ix + dx, c.iy + dy});
            }
        }
        return out;
    }

    Projection load_projection() const {
        Projection projection;
        std::ifstream file(voxel_csv_path_);
        if (!file.is_open()) {
            return projection;
        }

        std::string line;
        std::getline(file, line);

        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> cols;
            while (std::getline(ss, token, ',')) {
                cols.push_back(token);
            }
            if (cols.size() < 7) {
                continue;
            }

            const int ix = std::stoi(cols[0]);
            const int iy = std::stoi(cols[1]);
            const double z = std::stod(cols[5]);
            const int value = std::stoi(cols[6]);
            if (z > robot_clearance_z_) {
                continue;
            }

            const Cell cell{ix, iy};
            if (!in_bounds(cell)) {
                continue;
            }

            projection.voxel_rows++;
            projection.observed.insert(cell);
            if (value == 1 && z >= obstacle_min_z_) {
                projection.occupied.insert(cell);
            } else if (projection.occupied.count(cell) == 0) {
                projection.free.insert(cell);
            }
        }

        for (const Cell& occ : projection.occupied) {
            projection.free.erase(occ);
        }
        return projection;
    }

    bool is_frontier(const Projection& p, const Cell& cell) const {
        if (p.free.count(cell) == 0 || p.occupied.count(cell) > 0) {
            return false;
        }
        for (const Cell& n : neighbors8(cell)) {
            if (in_bounds(n) && p.observed.count(n) == 0) {
                return true;
            }
        }
        return false;
    }

    int unknown_gain(const Projection& p, const Cell& cell, int radius) const {
        int gain = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const Cell q{cell.ix + dx, cell.iy + dy};
                if (in_bounds(q) && p.observed.count(q) == 0) {
                    gain++;
                }
            }
        }
        return gain;
    }

    double revisit_penalty(const Cell& cell) const {
        double penalty = 0.0;
        for (const Cell& visited : visited_cells_) {
            const int dx = visited.ix - cell.ix;
            const int dy = visited.iy - cell.iy;
            if (std::abs(dx) <= 1 && std::abs(dy) <= 1) {
                penalty += 1.0;
            }
        }
        return penalty;
    }

    Cell nearest_free_cell(const Projection& p, const Cell& origin) const {
        if (p.free.count(origin) > 0) {
            return origin;
        }
        Cell best = origin;
        double best_dist = std::numeric_limits<double>::infinity();
        for (const Cell& cell : p.free) {
            const double dx = static_cast<double>(cell.ix - origin.ix);
            const double dy = static_cast<double>(cell.iy - origin.iy);
            const double dist = dx * dx + dy * dy;
            if (dist < best_dist) {
                best_dist = dist;
                best = cell;
            }
        }
        return best;
    }

    std::set<Cell> reachable_free_cells(const Projection& p) const {
        std::set<Cell> reachable;
        if (p.free.empty()) {
            return reachable;
        }

        const Cell robot_cell = nearest_free_cell(p, world_to_cell(current_x_, current_y_));
        if (p.free.count(robot_cell) == 0) {
            return reachable;
        }

        std::queue<Cell> q;
        q.push(robot_cell);
        reachable.insert(robot_cell);

        while (!q.empty()) {
            const Cell cell = q.front();
            q.pop();
            for (const Cell& next : neighbors4(cell)) {
                if (!in_bounds(next)) {
                    continue;
                }
                if (p.free.count(next) == 0 || reachable.count(next) > 0) {
                    continue;
                }
                reachable.insert(next);
                q.push(next);
            }
        }
        return reachable;
    }

    std::vector<Cell> graph_path_to(const Projection& p, const Cell& goal) const {
        std::vector<Cell> empty;
        if (p.free.empty() || p.free.count(goal) == 0) {
            return empty;
        }

        const Cell start = nearest_free_cell(p, world_to_cell(current_x_, current_y_));
        if (p.free.count(start) == 0) {
            return empty;
        }

        std::queue<Cell> q;
        std::set<Cell> seen;
        std::map<Cell, Cell> parent;
        q.push(start);
        seen.insert(start);

        while (!q.empty()) {
            const Cell cell = q.front();
            q.pop();
            if (cell == goal) {
                break;
            }
            for (const Cell& next : neighbors4(cell)) {
                if (!in_bounds(next) || p.free.count(next) == 0 || seen.count(next) > 0) {
                    continue;
                }
                seen.insert(next);
                parent[next] = cell;
                q.push(next);
            }
        }

        if (seen.count(goal) == 0) {
            return empty;
        }

        std::vector<Cell> path;
        Cell cur = goal;
        path.push_back(cur);
        while (!(cur == start)) {
            cur = parent[cur];
            path.push_back(cur);
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    Cell next_graph_waypoint(const Projection& p, const Cell& goal) const {
        const std::vector<Cell> path = graph_path_to(p, goal);
        if (path.empty()) {
            return goal;
        }
        const std::size_t lookahead = std::min<std::size_t>(path.size() - 1, 3);
        return path[lookahead];
    }

    double target_direction_prior(double x, double y) const {
        if (!enable_target_direction_prior_) {
            return 0.0;
        }
        if (is_bed_goal()) {
            return 1.05 * x - 0.75 * y;
        }
        if (is_chair_goal()) {
            return -0.95 * x + 0.75 * y;
        }
        return 0.0;
    }

    bool select_coverage_goal(const Projection& p, Frontier& out) const {
        const std::set<Cell> reachable = reachable_free_cells(p);
        double best_score = -std::numeric_limits<double>::infinity();
        bool found = false;

        for (const Cell& cell : reachable) {
            if (p.occupied.count(cell) > 0) {
                continue;
            }
            const double x = cell_x(cell.ix);
            const double y = cell_y(cell.iy);
            const double travel = std::hypot(x - current_x_, y - current_y_);
            if (travel < 0.55) {
                continue;
            }
            const int gain = unknown_gain(p, cell, 4);
            const double revisit = revisit_penalty(cell);
            const double score =
                1.10 * static_cast<double>(gain) +
                0.35 * travel -
                3.00 * revisit +
                2.20 * target_direction_prior(x, y);
            if (score > best_score) {
                best_score = score;
                out = Frontier{cell, x, y, score, gain};
                found = true;
            }
        }
        return found;
    }

    int node_id(const Cell& cell) const {
        return cell.iy * cols_ + cell.ix;
    }

    int observed_edge_count(const Projection& p) const {
        int edges = 0;
        for (const Cell& cell : p.free) {
            const Cell right{cell.ix + 1, cell.iy};
            const Cell up{cell.ix, cell.iy + 1};
            if (p.free.count(right) > 0) {
                edges++;
            }
            if (p.free.count(up) > 0) {
                edges++;
            }
        }
        return edges;
    }

    void write_runtime_graph(const Projection& p, const std::vector<Frontier>& frontiers) {
        std::set<Cell> frontier_cells;
        for (const Frontier& frontier : frontiers) {
            frontier_cells.insert(frontier.cell);
        }

        const std::filesystem::path node_path(graph_nodes_path_);
        if (node_path.has_parent_path()) {
            std::filesystem::create_directories(node_path.parent_path());
        }
        std::ofstream nodes(graph_nodes_path_);
        if (nodes.is_open()) {
            nodes << "timestamp,node_id,ix,iy,x,y,is_frontier,is_selected,is_visited,target_score,red_ratio,blue_ratio,cyan_ratio\n";
            for (const Cell& cell : p.free) {
                const bool is_selected = has_selected_ && cell == selected_cell_;
                nodes << this->now().seconds() << ","
                      << node_id(cell) << ","
                      << cell.ix << ","
                      << cell.iy << ","
                      << cell_x(cell.ix) << ","
                      << cell_y(cell.iy) << ","
                      << (frontier_cells.count(cell) > 0 ? 1 : 0) << ","
                      << (is_selected ? 1 : 0) << ","
                      << (visited_cells_.count(cell) > 0 ? 1 : 0) << ","
                      << target_score_ << ","
                      << red_ratio_ << ","
                      << blue_ratio_ << ","
                      << cyan_ratio_ << "\n";
            }
        }

        const std::filesystem::path edge_path(graph_edges_path_);
        if (edge_path.has_parent_path()) {
            std::filesystem::create_directories(edge_path.parent_path());
        }
        std::ofstream edges(graph_edges_path_);
        if (edges.is_open()) {
            edges << "timestamp,src_node_id,dst_node_id,src_ix,src_iy,dst_ix,dst_iy\n";
            for (const Cell& cell : p.free) {
                for (const Cell& n : neighbors4(cell)) {
                    if (p.free.count(n) == 0 || node_id(cell) >= node_id(n)) {
                        continue;
                    }
                    edges << this->now().seconds() << ","
                          << node_id(cell) << ","
                          << node_id(n) << ","
                          << cell.ix << ","
                          << cell.iy << ","
                          << n.ix << ","
                          << n.iy << "\n";
                }
            }
        }
    }

    void write_frontier_features(const std::vector<Frontier>& frontiers) {
        if (frontiers.empty()) {
            return;
        }
        const std::filesystem::path out_path(frontier_features_path_);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        std::ofstream file(frontier_features_path_);
        if (!file.is_open()) {
            return;
        }
        file << "timestamp,policy_mode,frontier_rank,node_id,ix,iy,x,y,distance,yaw_error,unknown_gain,revisit_penalty,heuristic_score,is_selected,target_goal,target_score,red_ratio,blue_ratio,cyan_ratio,observed_nodes,observed_edges,frontier_count\n";
        int rank = 0;
        for (const Frontier& frontier : frontiers) {
            const double dx = frontier.x - current_x_;
            const double dy = frontier.y - current_y_;
            const double distance = std::hypot(dx, dy);
            const double target_yaw = std::atan2(dy, dx);
            const double yaw_error = normalize_angle(target_yaw - current_yaw_);
            const double revisit = revisit_penalty(frontier.cell);
            const bool is_selected = has_selected_ && frontier.cell == selected_cell_;
            file << this->now().seconds() << ","
                 << policy_mode_ << ","
                 << rank << ","
                 << node_id(frontier.cell) << ","
                 << frontier.cell.ix << ","
                 << frontier.cell.iy << ","
                 << frontier.x << ","
                 << frontier.y << ","
                 << distance << ","
                 << yaw_error << ","
                 << frontier.unknown_gain << ","
                 << revisit << ","
                 << frontier.score << ","
                 << (is_selected ? 1 : 0) << ","
                 << target_goal_ << ","
                 << target_score_ << ","
                 << red_ratio_ << ","
                 << blue_ratio_ << ","
                 << cyan_ratio_ << ","
                 << observed_cells_ << ","
                 << observed_edges_ << ","
                 << frontier_count_ << "\n";
            rank++;
        }
    }

    std::vector<Frontier> extract_frontiers(const Projection& p) const {
        std::vector<Frontier> frontiers;
        const std::set<Cell> reachable = reachable_free_cells(p);
        for (const Cell& cell : p.free) {
            if (reachable.count(cell) == 0) {
                continue;
            }
            if (!is_frontier(p, cell)) {
                continue;
            }
            if (blocked_frontiers_.count(cell) > 0) {
                continue;
            }
            const double x = cell_x(cell.ix);
            const double y = cell_y(cell.iy);
            const double travel = std::hypot(x - current_x_, y - current_y_);
            if (travel < min_frontier_distance_) {
                continue;
            }
            const int gain = unknown_gain(p, cell, 2);
            const double revisit = revisit_penalty(cell);
            const double score =
                1.35 * static_cast<double>(gain) -
                1.1 * travel -
                3.5 * revisit +
                1.60 * target_direction_prior(x, y);
            frontiers.push_back(Frontier{cell, x, y, score, gain});
        }
        std::sort(frontiers.begin(), frontiers.end(), [](const Frontier& a, const Frontier& b) {
            return a.score > b.score;
        });
        return frontiers;
    }

    void append_trajectory() {
        const Cell cell = world_to_cell(current_x_, current_y_);
        visited_cells_.insert(cell);
        trajectory_.push_back(std::make_tuple(this->now().seconds(), current_x_, current_y_, current_yaw_));

        const std::filesystem::path out_path(trajectory_path_);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        const bool need_header = !std::filesystem::exists(out_path) || std::filesystem::file_size(out_path) == 0;
        std::ofstream file(trajectory_path_, std::ios::app);
        if (!file.is_open()) {
            return;
        }
        if (need_header) {
            file << "timestamp,x,y,yaw,target_x,target_y,frontier_count,target_goal,target_score,red_ratio,blue_ratio,cyan_ratio\n";
        }
        file << this->now().seconds() << ","
             << current_x_ << ","
             << current_y_ << ","
             << current_yaw_ << ","
             << selected_x_ << ","
             << selected_y_ << ","
             << frontier_count_ << ","
             << target_goal_ << ","
             << target_score_ << ","
             << red_ratio_ << ","
             << blue_ratio_ << ","
             << cyan_ratio_ << "\n";
    }

    void write_metrics(bool success) {
        const std::filesystem::path out_path(metrics_path_);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::ofstream file(metrics_path_);
        if (!file.is_open()) {
            return;
        }
        const double elapsed = (this->now() - start_time_).seconds();
        const double coverage_ratio =
            rows_ > 0 && cols_ > 0 ?
            static_cast<double>(observed_cells_) / static_cast<double>(rows_ * cols_) :
            0.0;
        file << "success,elapsed_sec,control_steps,observed_cells,observed_edges,frontier_count,frontier_switches,revisits,coverage_ratio,frontier_exhausted,target_event_count,confirmed_target_count,policy_mode,mission_mode,target_goal,target_candidate_visible,target_visible,target_score,target_center_x,target_range_m,red_ratio,blue_ratio,cyan_ratio\n";
        file << (success ? 1 : 0) << ","
             << elapsed << ","
             << control_steps_ << ","
             << observed_cells_ << ","
             << observed_edges_ << ","
             << frontier_count_ << ","
             << frontier_switches_ << ","
             << revisits_ << ","
             << coverage_ratio << ","
             << (frontier_exhausted_ ? 1 : 0) << ","
             << target_event_count_ << ","
             << confirmed_target_count_ << ","
             << policy_mode_ << ","
             << mission_mode_ << ","
             << target_goal_ << ","
             << (target_candidate_visible_ ? 1 : 0) << ","
             << (target_visible_ ? 1 : 0) << ","
             << target_score_ << ","
             << target_center_x_ << ","
             << (std::isfinite(center_clearance_) ? center_clearance_ : -1.0) << ","
             << red_ratio_ << ","
             << blue_ratio_ << ","
             << cyan_ratio_ << "\n";
    }

    void append_target_event(bool confirmed, const char* source) {
        const double now_sec = this->now().seconds();
        if (!confirmed && now_sec - last_candidate_event_sec_ < event_log_period_sec_) {
            return;
        }
        if (confirmed && now_sec - last_confirmed_event_sec_ < event_log_period_sec_) {
            return;
        }
        if (confirmed) {
            last_confirmed_event_sec_ = now_sec;
            confirmed_target_count_++;
        } else {
            last_candidate_event_sec_ = now_sec;
        }
        target_event_count_++;

        const std::filesystem::path out_path(target_events_path_);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        const bool need_header = !std::filesystem::exists(out_path) || std::filesystem::file_size(out_path) == 0;
        std::ofstream file(target_events_path_, std::ios::app);
        if (!file.is_open()) {
            return;
        }
        if (need_header) {
            file << "timestamp,event_index,target_goal,confirmed,source,x,y,yaw,target_score,target_center_x,target_range_m,red_ratio,blue_ratio,cyan_ratio,observed_cells,frontier_count,coverage_ratio\n";
        }
        const double coverage_ratio =
            rows_ > 0 && cols_ > 0 ?
            static_cast<double>(observed_cells_) / static_cast<double>(rows_ * cols_) :
            0.0;
        file << now_sec << ","
             << target_event_count_ << ","
             << target_goal_ << ","
             << (confirmed ? 1 : 0) << ","
             << source << ","
             << current_x_ << ","
             << current_y_ << ","
             << current_yaw_ << ","
             << target_score_ << ","
             << target_center_x_ << ","
             << (std::isfinite(center_clearance_) ? center_clearance_ : -1.0) << ","
             << red_ratio_ << ","
             << blue_ratio_ << ","
             << cyan_ratio_ << ","
             << observed_cells_ << ","
             << frontier_count_ << ","
             << coverage_ratio << "\n";

        RCLCPP_INFO(
            this->get_logger(),
            "Target event logged | goal=%s confirmed=%d score=%.4f range=%.2f pose=(%.2f %.2f)",
            target_goal_.c_str(),
            confirmed ? 1 : 0,
            target_score_,
            std::isfinite(center_clearance_) ? center_clearance_ : -1.0,
            current_x_,
            current_y_);
    }

    geometry_msgs::msg::Twist stop_cmd() const {
        return geometry_msgs::msg::Twist();
    }

    bool publish_avoidance_cmd() {
        if (!enable_local_avoidance_ || !has_depth_ || !std::isfinite(center_clearance_) || center_clearance_ > obstacle_stop_distance_) {
            return false;
        }
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = 0.0;
        cmd.angular.z = (left_clearance_ >= right_clearance_) ? angular_speed_ : -angular_speed_;
        cmd_pub_->publish(cmd);
        return true;
    }

    void publish_goal_cmd(double goal_x, double goal_y) {
        if (publish_avoidance_cmd()) {
            return;
        }
        geometry_msgs::msg::Twist cmd;
        const double dx = goal_x - current_x_;
        const double dy = goal_y - current_y_;
        const double full_dist = std::hypot(dx, dy);
        const double step = std::min(0.35, full_dist);
        const double scale = full_dist > 1e-6 ? step / full_dist : 0.0;
        const double local_x = current_x_ + dx * scale;
        const double local_y = current_y_ + dy * scale;
        const double ldx = local_x - current_x_;
        const double ldy = local_y - current_y_;
        const double dist = std::hypot(ldx, ldy);
        const double target_yaw = std::atan2(ldy, ldx);
        const double yaw_error = normalize_angle(target_yaw - current_yaw_);

        if (std::abs(yaw_error) > 1.75) {
            cmd.linear.x = 0.0;
            cmd.angular.z = yaw_error > 0.0 ? angular_speed_ : -angular_speed_;
        } else {
            const double turn_slowdown = std::max(0.18, 1.0 - std::abs(yaw_error) / 1.75);
            cmd.linear.x = std::max(0.045, std::min(linear_speed_, dist * 0.7)) * turn_slowdown;
            cmd.angular.z = std::max(-angular_speed_, std::min(angular_speed_, yaw_error * 1.10));
        }

        cmd_pub_->publish(cmd);
    }

    void publish_target_approach_cmd() {
        geometry_msgs::msg::Twist cmd;
        const double center_error = target_center_x_ - 0.5;
        if (std::abs(center_error) > 0.10) {
            cmd.linear.x = 0.0;
            cmd.angular.z = std::max(-angular_speed_, std::min(angular_speed_, -center_error * 1.6));
        } else {
            const bool near_object =
                has_depth_ &&
                std::isfinite(center_clearance_) &&
                center_clearance_ < (target_confirm_max_distance_ + 0.25);
            cmd.linear.x = near_object ? 0.035 : 0.060;
            cmd.angular.z = std::max(-0.28, std::min(0.28, -center_error * 1.2));
        }
        cmd_pub_->publish(cmd);
    }

    void publish_scan_cmd() {
        geometry_msgs::msg::Twist cmd;
        const bool forward_clear =
            !has_depth_ ||
            !std::isfinite(center_clearance_) ||
            center_clearance_ > 0.55;
        cmd.linear.x = forward_clear ? 0.045 : 0.0;
        cmd.angular.z = 0.35;
        cmd_pub_->publish(cmd);
    }

    void publish_local_explore_cmd() {
        geometry_msgs::msg::Twist cmd;
        const bool center_clear =
            !has_depth_ ||
            !std::isfinite(center_clearance_) ||
            center_clearance_ > 0.70;
        if (center_clear) {
            cmd.linear.x = 0.080;
            cmd.angular.z = 0.22 * explore_turn_sign_;
        } else {
            const double left = std::isfinite(left_clearance_) ? left_clearance_ : 0.0;
            const double right = std::isfinite(right_clearance_) ? right_clearance_ : 0.0;
            explore_turn_sign_ = (left >= right) ? 1.0 : -1.0;
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.55 * explore_turn_sign_;
        }
        cmd_pub_->publish(cmd);
    }

    void publish_escape_cmd() {
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = 0.055;
        cmd.angular.z = escape_turn_sign_ * 0.45;
        cmd_pub_->publish(cmd);
    }

    void block_frontier_neighborhood(const Cell& cell, int radius = 3) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const Cell blocked{cell.ix + dx, cell.iy + dy};
                if (in_bounds(blocked)) {
                    blocked_frontiers_.insert(blocked);
                }
            }
        }
    }

    bool entry_behavior_active() const {
        return enable_entry_behavior_ && current_y_ < (entry_y_ - goal_tolerance_);
    }

    bool target_close_enough() const {
        if (target_score_ < target_confirm_ratio_threshold_) {
            return false;
        }
        if (std::abs(target_center_x_ - 0.5) > target_center_tolerance_) {
            return false;
        }
        if (!has_depth_ || !std::isfinite(center_clearance_)) {
            return false;
        }
        return center_clearance_ <= target_confirm_max_distance_;
    }

    bool is_bed_goal() const {
        return target_goal_ == "bed" ||
               target_goal_ == "blue_bed" ||
               target_goal_ == "bedroom" ||
               target_goal_ == "침대" ||
               target_goal_ == "파란침대" ||
               target_goal_ == "파란 침대";
    }

    bool is_chair_goal() const {
        return target_goal_ == "chair" ||
               target_goal_ == "red_chair" ||
               target_goal_ == "의자" ||
               target_goal_ == "빨간의자" ||
               target_goal_ == "빨간 의자";
    }

    bool selected_frontier_stalled(double dist_to_goal) {
        if (!has_selected_) {
            return false;
        }
        if (last_selected_cell_valid_ && selected_cell_ == last_selected_cell_) {
            if (dist_to_goal < last_selected_distance_ - 0.015) {
                selected_stale_steps_ = 0;
            } else {
                selected_stale_steps_++;
            }
        } else {
            selected_stale_steps_ = 0;
            last_selected_cell_ = selected_cell_;
            last_selected_cell_valid_ = true;
        }
        last_selected_distance_ = dist_to_goal;
        return selected_stale_steps_ > 12;
    }

    bool robot_motion_stalled() {
        const double moved = std::hypot(current_x_ - last_motion_x_, current_y_ - last_motion_y_);
        if (!last_motion_valid_) {
            last_motion_x_ = current_x_;
            last_motion_y_ = current_y_;
            last_motion_valid_ = true;
            return false;
        }

        if (moved > 0.08) {
            motion_stale_steps_ = 0;
            last_motion_x_ = current_x_;
            last_motion_y_ = current_y_;
            return false;
        }

        motion_stale_steps_++;
        return motion_stale_steps_ > 14;
    }

    void control_loop() {
        if (!has_odom_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for odom");
            return;
        }

        if (escape_steps_remaining_ > 0) {
            publish_escape_cmd();
            append_trajectory();
            write_metrics(false);
            control_steps_++;
            escape_steps_remaining_--;
            return;
        }

        if (entry_behavior_active()) {
            publish_goal_cmd(entry_x_, entry_y_);
            append_trajectory();
            write_metrics(false);
            control_steps_++;
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Entry behavior active. Moving inside first. pose=(%.2f %.2f) entry=(%.2f %.2f)",
                current_x_,
                current_y_,
                entry_x_,
                entry_y_);
            return;
        }

        if (target_candidate_visible_) {
            append_target_event(target_visible_, target_visible_ ? "confirmed_vision" : "candidate_vision");
        }

        if (target_visible_ && stop_on_target_) {
            cmd_pub_->publish(stop_cmd());
            write_metrics(true);
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Target reached close enough. Stopping. goal=%s score=%.4f range=%.2f center_x=%.3f red=%.4f blue=%.4f cyan=%.4f",
                target_goal_.c_str(),
                target_score_,
                center_clearance_,
                target_center_x_,
                red_ratio_,
                blue_ratio_,
                cyan_ratio_);
            return;
        }

        if (target_candidate_visible_ && approach_target_candidate_) {
            publish_target_approach_cmd();
            append_trajectory();
            write_metrics(false);
            control_steps_++;
            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Target candidate visible. Approaching until close. goal=%s score=%.4f center_x=%.3f range=%.2f confirm_score=%.4f stop_range=%.2f",
                target_goal_.c_str(),
                target_score_,
                target_center_x_,
                center_clearance_,
                target_confirm_ratio_threshold_,
                target_confirm_max_distance_);
            return;
        }

        if ((this->now() - start_time_).seconds() > max_runtime_sec_) {
            cmd_pub_->publish(stop_cmd());
            write_metrics(false);
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Max runtime reached. Stopping.");
            return;
        }

        const Projection projection = load_projection();
        observed_cells_ = static_cast<int>(projection.observed.size());
        observed_edges_ = observed_edge_count(projection);
        frontier_exhausted_ = false;

        if (projection.free.empty()) {
            cmd_pub_->publish(stop_cmd());
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for observed free cells from voxel mapper");
            return;
        }

        std::vector<Frontier> frontiers = extract_frontiers(projection);
        if (frontiers.empty() && !blocked_frontiers_.empty()) {
            blocked_frontiers_.clear();
            has_selected_ = false;
            frontiers = extract_frontiers(projection);
        }
        frontier_count_ = static_cast<int>(frontiers.size());
        if (frontiers.empty()) {
            Frontier coverage;
            if (select_coverage_goal(projection, coverage)) {
                frontier_exhausted_ = true;
                selected_cell_ = coverage.cell;
                selected_x_ = coverage.x;
                selected_y_ = coverage.y;
                has_selected_ = true;
                frontier_switches_++;
                std::vector<Frontier> coverage_frontiers{coverage};
                write_runtime_graph(projection, coverage_frontiers);
                write_frontier_features(coverage_frontiers);
                if (robot_motion_stalled()) {
                    escape_turn_sign_ *= -1.0;
                    escape_steps_remaining_ = 12;
                    motion_stale_steps_ = 0;
                    last_motion_valid_ = false;
                    publish_escape_cmd();
                } else {
                    const Cell waypoint = next_graph_waypoint(projection, selected_cell_);
                    publish_goal_cmd(cell_x(waypoint.ix), cell_y(waypoint.iy));
                }
                append_trajectory();
                write_metrics(false);
                control_steps_++;
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    1000,
                    "No frontier candidates. Moving to coverage goal=(%.2f %.2f) gain=%d score=%.2f",
                    selected_x_,
                    selected_y_,
                    coverage.unknown_gain,
                    coverage.score);
                return;
            }
            frontier_exhausted_ = true;
            write_runtime_graph(projection, frontiers);
            write_frontier_features(frontiers);
            publish_local_explore_cmd();
            append_trajectory();
            write_metrics(false);
            control_steps_++;
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "No frontier candidates");
            return;
        }

        const Frontier best = frontiers.front();
        if (!has_selected_ || !(best.cell == selected_cell_)) {
            selected_cell_ = best.cell;
            selected_x_ = best.x;
            selected_y_ = best.y;
            has_selected_ = true;
            frontier_switches_++;
        }

        write_runtime_graph(projection, frontiers);
        write_frontier_features(frontiers);

        const double dist_to_goal = std::hypot(selected_x_ - current_x_, selected_y_ - current_y_);
        if (selected_frontier_stalled(dist_to_goal) || robot_motion_stalled()) {
            block_frontier_neighborhood(selected_cell_);
            has_selected_ = false;
            selected_stale_steps_ = 0;
            motion_stale_steps_ = 0;
            last_motion_valid_ = false;
            escape_turn_sign_ *= -1.0;
            escape_steps_remaining_ = 16;
            publish_escape_cmd();
            append_trajectory();
            write_metrics(false);
            control_steps_++;
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Selected frontier stalled. Blocking cell=(%d,%d) and scanning.",
                selected_cell_.ix,
                selected_cell_.iy);
            return;
        }
        if (dist_to_goal < goal_tolerance_) {
            block_frontier_neighborhood(selected_cell_, 2);
            has_selected_ = false;
        } else {
            const Cell waypoint = next_graph_waypoint(projection, selected_cell_);
            publish_goal_cmd(cell_x(waypoint.ix), cell_y(waypoint.iy));
        }

        const Cell current_cell = world_to_cell(current_x_, current_y_);
        if (visited_cells_.count(current_cell) > 0) {
            revisits_++;
        }

        append_trajectory();
        write_metrics(false);
        control_steps_++;

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Frontier nav | mode=%s goal=%s pose=(%.2f %.2f) selected=(%.2f %.2f) frontiers=%d observed_nodes=%d observed_edges=%d score=%.2f gain=%d target_score=%.4f",
            policy_mode_.c_str(),
            target_goal_.c_str(),
            current_x_,
            current_y_,
            selected_x_,
            selected_y_,
            frontier_count_,
            observed_cells_,
            observed_edges_,
            best.score,
            best.unknown_gain,
            target_score_);
    }

    std::string voxel_csv_path_;
    std::string trajectory_path_;
    std::string metrics_path_;
    std::string camera_debug_path_;
    std::string graph_nodes_path_;
    std::string graph_edges_path_;
    std::string frontier_features_path_;
    std::string target_events_path_;
    std::string policy_mode_ = "heuristic";
    std::string mission_mode_ = "coverage_patrol";
    std::string odom_topic_;
    std::string cmd_vel_topic_;
    std::string camera_topic_;
    std::string depth_topic_;
    std::string target_goal_ = "chair";

    double voxel_size_ = 0.2;
    double min_x_ = -3.0;
    double max_x_ = 6.0;
    double min_y_ = -4.0;
    double max_y_ = 4.0;
    double min_z_ = 0.0;
    double max_z_ = 2.4;
    double obstacle_min_z_ = 0.45;
    double robot_clearance_z_ = 0.55;
    double target_red_ratio_threshold_ = 0.010;
    double target_candidate_ratio_threshold_ = 0.006;
    double target_confirm_ratio_threshold_ = 0.055;
    double target_confirm_max_distance_ = 0.55;
    double target_center_tolerance_ = 0.18;
    double min_frontier_distance_ = 0.45;
    double goal_tolerance_ = 0.12;
    double linear_speed_ = 0.11;
    double angular_speed_ = 0.55;
    bool enable_local_avoidance_ = false;
    bool stop_on_target_ = false;
    bool approach_target_candidate_ = false;
    bool enable_target_direction_prior_ = false;
    double event_log_period_sec_ = 2.0;
    double obstacle_stop_distance_ = 0.28;
    double entry_x_ = 1.5;
    double entry_y_ = -2.65;
    bool enable_entry_behavior_ = true;
    double max_runtime_sec_ = 240.0;

    int cols_ = 0;
    int rows_ = 0;

    double current_x_ = 0.0;
    double current_y_ = 0.0;
    double current_yaw_ = 0.0;
    bool has_odom_ = false;

    bool target_visible_ = false;
    bool target_candidate_visible_ = false;
    double target_score_ = 0.0;
    double target_center_x_ = 0.5;
    double red_ratio_ = 0.0;
    double blue_ratio_ = 0.0;
    double cyan_ratio_ = 0.0;
    double last_camera_save_sec_ = 0.0;
    bool has_depth_ = false;
    double left_clearance_ = std::numeric_limits<double>::infinity();
    double center_clearance_ = std::numeric_limits<double>::infinity();
    double right_clearance_ = std::numeric_limits<double>::infinity();

    bool has_selected_ = false;
    Cell selected_cell_;
    double selected_x_ = 0.0;
    double selected_y_ = 0.0;
    bool last_selected_cell_valid_ = false;
    Cell last_selected_cell_;
    double last_selected_distance_ = std::numeric_limits<double>::infinity();
    int selected_stale_steps_ = 0;
    std::set<Cell> blocked_frontiers_;
    bool last_motion_valid_ = false;
    double last_motion_x_ = 0.0;
    double last_motion_y_ = 0.0;
    int motion_stale_steps_ = 0;
    int escape_steps_remaining_ = 0;
    double escape_turn_sign_ = 1.0;
    double explore_turn_sign_ = 1.0;

    int control_steps_ = 0;
    int observed_cells_ = 0;
    int observed_edges_ = 0;
    int frontier_count_ = 0;
    int frontier_switches_ = 0;
    int revisits_ = 0;
    bool frontier_exhausted_ = false;
    int target_event_count_ = 0;
    int confirmed_target_count_ = 0;
    double last_candidate_event_sec_ = -1e9;
    double last_confirmed_event_sec_ = -1e9;
    rclcpp::Time start_time_{0, 0, RCL_SYSTEM_TIME};

    std::set<Cell> visited_cells_;
    std::vector<std::tuple<double, double, double, double>> trajectory_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RgbdFrontierNavigator>());
    rclcpp::shutdown();
    return 0;
}
