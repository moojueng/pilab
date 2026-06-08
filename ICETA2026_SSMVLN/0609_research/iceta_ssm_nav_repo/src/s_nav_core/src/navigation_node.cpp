#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "s_nav_core/SsmInference.hpp"

class NavigationNode : public rclcpp::Node {
public:
    NavigationNode() : Node("ssm_navigation_node") {
        this->declare_parameter("model_path", "models/ssm_policy_robot_frame.onnx");
        this->declare_parameter("camera_topic", "/camera/image_raw");
        this->declare_parameter("scan_topic", "/scan");
        this->declare_parameter("odom_topic", "/odom");
        this->declare_parameter("cmd_vel_topic", "/cmd_vel_nav");
        this->declare_parameter("dataset_path", "/home/mj/my_research/ssm_nav_ws/datasets/ssm_nav/runtime_robot_frame.csv");

        this->declare_parameter("goal_x", 2.0);
        this->declare_parameter("goal_y", 2.0);
        this->declare_parameter("goal_tolerance", 0.35);

        this->declare_parameter("linear_speed", 0.10);
        this->declare_parameter("turn_linear_speed", 0.035);
        this->declare_parameter("reverse_speed", -0.06);
        this->declare_parameter("angular_speed", 0.50);

        this->declare_parameter("front_block_threshold", 0.22);
        this->declare_parameter("front_free_threshold", 0.34);
        this->declare_parameter("side_balance_margin", 0.08);
        this->declare_parameter("target_red_ratio_threshold", 0.010);

        std::string model_path = this->get_parameter("model_path").as_string();
        camera_topic_ = this->get_parameter("camera_topic").as_string();
        scan_topic_ = this->get_parameter("scan_topic").as_string();
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
        dataset_path_ = this->get_parameter("dataset_path").as_string();

        goal_x_ = this->get_parameter("goal_x").as_double();
        goal_y_ = this->get_parameter("goal_y").as_double();
        goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();

        linear_speed_ = this->get_parameter("linear_speed").as_double();
        turn_linear_speed_ = this->get_parameter("turn_linear_speed").as_double();
        reverse_speed_ = this->get_parameter("reverse_speed").as_double();
        angular_speed_ = this->get_parameter("angular_speed").as_double();

        front_block_threshold_ = this->get_parameter("front_block_threshold").as_double();
        front_free_threshold_ = this->get_parameter("front_free_threshold").as_double();
        side_balance_margin_ = this->get_parameter("side_balance_margin").as_double();
        target_red_ratio_threshold_ = this->get_parameter("target_red_ratio_threshold").as_double();

        inference_engine_ = std::make_shared<SsmInference>(model_path);

        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            camera_topic_, 10,
            std::bind(&NavigationNode::image_callback, this, std::placeholders::_1));

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            scan_topic_, 10,
            std::bind(&NavigationNode::scan_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10,
            std::bind(&NavigationNode::odom_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            std::bind(&NavigationNode::control_loop, this));

        RCLCPP_INFO(this->get_logger(), "SSM sensor-only exploration node started.");
        RCLCPP_INFO(this->get_logger(), "Red target stop enabled. No runtime map/A* is used.");
    }

private:
    struct VisionFeature {
        double mean_intensity = 0.0;
        double dark_ratio = 0.0;
        double edge_density = 0.0;
        double red_ratio = 0.0;
    };

    struct ScanFeature {
        double front_clear = 1.0;
        double left_clear = 1.0;
        double right_clear = 1.0;
        double min_range = 1.0;
        double front_min_m = 3.5;
        double left_min_m = 3.5;
        double right_min_m = 3.5;
    };

    static double clamp01(double value) {
        return std::max(0.0, std::min(1.0, value));
    }

    static double normalize_angle(double angle) {
        while (angle > M_PI) {
            angle -= 2.0 * M_PI;
        }
        while (angle < -M_PI) {
            angle += 2.0 * M_PI;
        }
        return angle;
    }

    static double normalize_distance(double distance, double max_distance) {
        if (max_distance <= 0.0) {
            return 0.0;
        }
        return clamp01(distance / max_distance);
    }

    static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q) {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    VisionFeature extract_vision_feature(const cv::Mat& bgr_image) {
        VisionFeature feature;

        if (bgr_image.empty()) {
            return feature;
        }

        cv::Mat gray;
        cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);

        cv::Scalar mean_value = cv::mean(gray);
        feature.mean_intensity = mean_value[0] / 255.0;

        cv::Mat dark_mask;
        cv::threshold(gray, dark_mask, 60, 255, cv::THRESH_BINARY_INV);
        feature.dark_ratio =
            static_cast<double>(cv::countNonZero(dark_mask)) /
            static_cast<double>(gray.rows * gray.cols);

        cv::Mat edges;
        cv::Canny(gray, edges, 80, 160);
        feature.edge_density =
            static_cast<double>(cv::countNonZero(edges)) /
            static_cast<double>(gray.rows * gray.cols);

        cv::Mat hsv;
        cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);

        cv::Mat red_low;
        cv::Mat red_high;
        cv::inRange(hsv, cv::Scalar(0, 80, 80), cv::Scalar(12, 255, 255), red_low);
        cv::inRange(hsv, cv::Scalar(170, 80, 80), cv::Scalar(180, 255, 255), red_high);

        cv::Mat red_mask = red_low | red_high;
        feature.red_ratio =
            static_cast<double>(cv::countNonZero(red_mask)) /
            static_cast<double>(bgr_image.rows * bgr_image.cols);

        return feature;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");

            latest_vision_feature_ = extract_vision_feature(cv_ptr->image);
            target_visible_ = latest_vision_feature_.red_ratio > target_red_ratio_threshold_;
            has_vision_feature_ = true;

            cv::imwrite(
                "/home/mj/my_research/ssm_nav_ws/results/images/latest_camera.png",
                cv_ptr->image);

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Camera %ux%u | mean=%.3f dark=%.3f edge=%.3f red=%.4f target=%d",
                msg->width,
                msg->height,
                latest_vision_feature_.mean_intensity,
                latest_vision_feature_.dark_ratio,
                latest_vision_feature_.edge_density,
                latest_vision_feature_.red_ratio,
                target_visible_);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
        }
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        latest_scan_feature_ = extract_scan_feature(*msg);
        has_scan_feature_ = true;
    }

    ScanFeature extract_scan_feature(const sensor_msgs::msg::LaserScan& scan) {
        ScanFeature feature;

        const double usable_max = std::min(
            std::isfinite(scan.range_max) ? static_cast<double>(scan.range_max) : 3.5,
            3.5);

        double front_min = usable_max;
        double left_min = usable_max;
        double right_min = usable_max;
        double global_min = usable_max;

        for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
            const float range = scan.ranges[i];
            if (!std::isfinite(range) || range < scan.range_min) {
                continue;
            }

            const double angle_raw =
                static_cast<double>(scan.angle_min) +
                static_cast<double>(i) * static_cast<double>(scan.angle_increment);

            const double angle = normalize_angle(angle_raw);
            const double r = std::min(static_cast<double>(range), usable_max);

            global_min = std::min(global_min, r);

            if (std::abs(angle) <= 0.35) {
                front_min = std::min(front_min, r);
            } else if (angle > 0.35 && angle <= 1.55) {
                left_min = std::min(left_min, r);
            } else if (angle < -0.35 && angle >= -1.55) {
                right_min = std::min(right_min, r);
            }
        }

        feature.front_min_m = front_min;
        feature.left_min_m = left_min;
        feature.right_min_m = right_min;

        feature.front_clear = normalize_distance(front_min, usable_max);
        feature.left_clear = normalize_distance(left_min, usable_max);
        feature.right_clear = normalize_distance(right_min, usable_max);
        feature.min_range = normalize_distance(global_min, usable_max);

        return feature;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_x_ = msg->pose.pose.position.x;
        current_y_ = msg->pose.pose.position.y;
        current_yaw_ = quaternion_to_yaw(msg->pose.pose.orientation);
        has_odom_ = true;

        const double moved = std::hypot(current_x_ - last_progress_x_, current_y_ - last_progress_y_);
        if (moved > 0.08) {
            last_progress_x_ = current_x_;
            last_progress_y_ = current_y_;
            last_progress_time_ = this->now();
            stuck_recovery_until_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
        }
    }

    std::vector<float> build_policy_features() const {
        double dx = goal_x_ - current_x_;
        double dy = goal_y_ - current_y_;
        double distance_to_goal = std::sqrt(dx * dx + dy * dy);

        double world_dir_x = 0.0;
        double world_dir_y = 0.0;
        if (distance_to_goal > 1e-6) {
            world_dir_x = dx / distance_to_goal;
            world_dir_y = dy / distance_to_goal;
        }

        double cos_yaw = std::cos(current_yaw_);
        double sin_yaw = std::sin(current_yaw_);

        double direction_x = cos_yaw * world_dir_x + sin_yaw * world_dir_y;
        double direction_y = -sin_yaw * world_dir_x + cos_yaw * world_dir_y;

        std::vector<float> features;
        features.reserve(14);

        features.push_back(static_cast<float>(current_x_));
        features.push_back(static_cast<float>(current_y_));
        features.push_back(static_cast<float>(goal_x_));
        features.push_back(static_cast<float>(goal_y_));

        features.push_back(static_cast<float>(direction_x));
        features.push_back(static_cast<float>(direction_y));
        features.push_back(static_cast<float>(std::min(distance_to_goal, 10.0)));

        features.push_back(static_cast<float>(latest_vision_feature_.mean_intensity));
        features.push_back(static_cast<float>(latest_vision_feature_.dark_ratio));
        features.push_back(static_cast<float>(latest_vision_feature_.edge_density));

        features.push_back(static_cast<float>(latest_scan_feature_.front_clear));
        features.push_back(static_cast<float>(latest_scan_feature_.left_clear));
        features.push_back(static_cast<float>(latest_scan_feature_.right_clear));
        features.push_back(static_cast<float>(latest_scan_feature_.min_range));

        return features;
    }

    const char* action_name(int action) const {
        switch (action) {
            case 0:
                return "forward";
            case 1:
                return "reverse";
            case 2:
                return "left";
            case 3:
                return "right";
            default:
                return "unknown";
        }
    }

    bool recovery_active() const {
        return this->now() < stuck_recovery_until_;
    }

    void update_stuck_recovery() {
        if (!has_odom_) {
            return;
        }

        if (last_progress_time_.nanoseconds() == 0) {
            last_progress_x_ = current_x_;
            last_progress_y_ = current_y_;
            last_progress_time_ = this->now();
            return;
        }

        const double idle_sec = (this->now() - last_progress_time_).seconds();
        if (idle_sec > 3.0 && !recovery_active()) {
            stuck_recovery_until_ = this->now() + rclcpp::Duration::from_seconds(1.6);
            recovery_turn_direction_ =
                latest_scan_feature_.left_clear > latest_scan_feature_.right_clear ? 1.0 : -1.0;

            RCLCPP_WARN(
                this->get_logger(),
                "Stuck detected. Recovery turn direction=%.1f",
                recovery_turn_direction_);
        }
    }

    geometry_msgs::msg::Twist decide_cmd_vel(int policy_action) {
        geometry_msgs::msg::Twist cmd;

        update_stuck_recovery();

        if (recovery_active()) {
            cmd.linear.x = 0.0;
            cmd.angular.z = recovery_turn_direction_ * angular_speed_;
            return cmd;
        }

        const bool front_blocked = latest_scan_feature_.front_clear < front_block_threshold_;
        const bool front_narrow = latest_scan_feature_.front_clear < front_free_threshold_;

        if (front_blocked) {
            cmd.linear.x = 0.0;
            cmd.angular.z =
                latest_scan_feature_.left_clear > latest_scan_feature_.right_clear ?
                angular_speed_ : -angular_speed_;
            return cmd;
        }

        if (front_narrow) {
            cmd.linear.x = turn_linear_speed_;
            cmd.angular.z =
                latest_scan_feature_.left_clear > latest_scan_feature_.right_clear ?
                angular_speed_ * 0.65 : -angular_speed_ * 0.65;
            return cmd;
        }

        if (policy_action == 2) {
            cmd.linear.x = turn_linear_speed_;
            cmd.angular.z = angular_speed_ * 0.75;
            return cmd;
        }

        if (policy_action == 3) {
            cmd.linear.x = turn_linear_speed_;
            cmd.angular.z = -angular_speed_ * 0.75;
            return cmd;
        }

        if (policy_action == 1) {
            cmd.linear.x = reverse_speed_;
            cmd.angular.z = 0.0;
            return cmd;
        }

        cmd.linear.x = linear_speed_;

        const double side_error = latest_scan_feature_.left_clear - latest_scan_feature_.right_clear;
        if (std::abs(side_error) > side_balance_margin_) {
            cmd.angular.z = side_error > 0.0 ? 0.10 : -0.10;
        } else {
            cmd.angular.z = 0.0;
        }

        return cmd;
    }

    void append_runtime_log(int action, const std::vector<float>& features) {
        bool need_header = false;
        {
            std::ifstream check_file(dataset_path_);
            need_header = !check_file.good() || check_file.peek() == std::ifstream::traits_type::eof();
        }

        std::ofstream file(dataset_path_, std::ios::app);
        if (!file.is_open()) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Failed to open dataset file: %s",
                dataset_path_.c_str());
            return;
        }

        if (need_header) {
            file << "timestamp,current_x,current_y,goal_x,goal_y,"
                 << "robot_dir_x,robot_dir_y,distance_to_goal,"
                 << "vision_mean,vision_dark,vision_edge,"
                 << "front_clear,left_clear,right_clear,min_range,"
                 << "predicted_action,target_visible\n";
        }

        file << std::fixed << std::setprecision(6)
             << this->now().seconds();

        for (float value : features) {
            file << "," << value;
        }

        file << "," << action << "," << target_visible_ << "\n";
    }

    void control_loop() {
        if (!has_odom_ || !has_scan_feature_ || !has_vision_feature_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for sensors | odom=%d scan=%d vision=%d",
                has_odom_,
                has_scan_feature_,
                has_vision_feature_);
            return;
        }

        if (target_visible_) {
            geometry_msgs::msg::Twist stop_cmd;
            cmd_vel_pub_->publish(stop_cmd);

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Target cylinder detected. Stopping robot. red_ratio=%.4f",
                latest_vision_feature_.red_ratio);
            return;
        }

        std::vector<float> policy_features = build_policy_features();
        int action = inference_engine_->predictAction(policy_features);

        geometry_msgs::msg::Twist cmd = decide_cmd_vel(action);
        cmd_vel_pub_->publish(cmd);

        append_runtime_log(action, policy_features);

        const double dist = std::hypot(goal_x_ - current_x_, goal_y_ - current_y_);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            500,
            "Explore policy action=%d (%s) | pos=(%.2f, %.2f) yaw=%.2f dist=%.2f scan(front=%.2f left=%.2f right=%.2f) cmd(x=%.2f z=%.2f) red=%.4f",
            action,
            action_name(action),
            current_x_,
            current_y_,
            current_yaw_,
            dist,
            latest_scan_feature_.front_clear,
            latest_scan_feature_.left_clear,
            latest_scan_feature_.right_clear,
            cmd.linear.x,
            cmd.angular.z,
            latest_vision_feature_.red_ratio);
    }

    std::string camera_topic_;
    std::string scan_topic_;
    std::string odom_topic_;
    std::string cmd_vel_topic_;
    std::string dataset_path_;

    double goal_x_ = 2.0;
    double goal_y_ = 2.0;
    double goal_tolerance_ = 0.35;

    double linear_speed_ = 0.10;
    double turn_linear_speed_ = 0.035;
    double reverse_speed_ = -0.06;
    double angular_speed_ = 0.50;

    double front_block_threshold_ = 0.22;
    double front_free_threshold_ = 0.34;
    double side_balance_margin_ = 0.08;
    double target_red_ratio_threshold_ = 0.010;

    double current_x_ = 0.0;
    double current_y_ = 0.0;
    double current_yaw_ = 0.0;

    double last_progress_x_ = 0.0;
    double last_progress_y_ = 0.0;
    double recovery_turn_direction_ = 1.0;

    rclcpp::Time last_progress_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time stuck_recovery_until_{0, 0, RCL_ROS_TIME};

    bool has_odom_ = false;
    bool has_scan_feature_ = false;
    bool has_vision_feature_ = false;
    bool target_visible_ = false;

    VisionFeature latest_vision_feature_;
    ScanFeature latest_scan_feature_;

    std::shared_ptr<SsmInference> inference_engine_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NavigationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
