#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/quaternion.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

class DepthVoxelMapper : public rclcpp::Node {
public:
    DepthVoxelMapper() : Node("depth_voxel_mapper") {
        this->declare_parameter("depth_topic", "/camera/depth/image_raw");
        this->declare_parameter("camera_info_topic", "/camera/depth/camera_info");
        this->declare_parameter("odom_topic", "/odom");
        this->declare_parameter("output_path", "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/observed_voxels.csv");

        this->declare_parameter("voxel_size", 0.20);
        this->declare_parameter("min_x", -4.0);
        this->declare_parameter("max_x", 7.0);
        this->declare_parameter("min_y", -4.5);
        this->declare_parameter("max_y", 4.5);
        this->declare_parameter("min_z", 0.0);
        this->declare_parameter("max_z", 2.4);
        this->declare_parameter("camera_height", 0.31);
        this->declare_parameter("camera_forward_offset", 0.26);
        this->declare_parameter("max_depth", 2.6);
        this->declare_parameter("pixel_stride", 8);
        this->declare_parameter("save_period_sec", 1.0);

        depth_topic_ = this->get_parameter("depth_topic").as_string();
        camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
        odom_topic_ = this->get_parameter("odom_topic").as_string();
        output_path_ = this->get_parameter("output_path").as_string();

        voxel_size_ = this->get_parameter("voxel_size").as_double();
        min_x_ = this->get_parameter("min_x").as_double();
        max_x_ = this->get_parameter("max_x").as_double();
        min_y_ = this->get_parameter("min_y").as_double();
        max_y_ = this->get_parameter("max_y").as_double();
        min_z_ = this->get_parameter("min_z").as_double();
        max_z_ = this->get_parameter("max_z").as_double();
        camera_height_ = this->get_parameter("camera_height").as_double();
        camera_forward_offset_ = this->get_parameter("camera_forward_offset").as_double();
        max_depth_ = this->get_parameter("max_depth").as_double();
        pixel_stride_ = std::max(1, static_cast<int>(this->get_parameter("pixel_stride").as_int()));
        save_period_sec_ = this->get_parameter("save_period_sec").as_double();

        cols_ = std::max(1, static_cast<int>(std::ceil((max_x_ - min_x_) / voxel_size_)));
        rows_ = std::max(1, static_cast<int>(std::ceil((max_y_ - min_y_) / voxel_size_)));
        depth_layers_ = std::max(1, static_cast<int>(std::ceil((max_z_ - min_z_) / voxel_size_)));
        voxels_.assign(static_cast<std::size_t>(cols_ * rows_ * depth_layers_), -1);

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            depth_topic_, 10,
            std::bind(&DepthVoxelMapper::depth_callback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, 10,
            std::bind(&DepthVoxelMapper::camera_info_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic_, 10,
            std::bind(&DepthVoxelMapper::odom_callback, this, std::placeholders::_1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&DepthVoxelMapper::timer_callback, this));

        RCLCPP_INFO(
            this->get_logger(),
            "Depth voxel mapper started | grid=%dx%dx%d voxel=%.2fm depth_topic=%s",
            cols_, rows_, depth_layers_, voxel_size_, depth_topic_.c_str());
    }

private:
    struct Point3 {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q) {
        const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
        has_camera_info_ = fx_ > 0.0 && fy_ > 0.0;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        robot_x_ = msg->pose.pose.position.x;
        robot_y_ = msg->pose.pose.position.y;
        robot_yaw_ = quaternion_to_yaw(msg->pose.pose.orientation);
        has_odom_ = true;
    }

    int index(int ix, int iy, int iz) const {
        return iz * rows_ * cols_ + iy * cols_ + ix;
    }

    bool world_to_voxel(const Point3& p, int& ix, int& iy, int& iz) const {
        if (p.x < min_x_ || p.x >= max_x_ ||
            p.y < min_y_ || p.y >= max_y_ ||
            p.z < min_z_ || p.z >= max_z_) {
            return false;
        }
        ix = static_cast<int>((p.x - min_x_) / voxel_size_);
        iy = static_cast<int>((p.y - min_y_) / voxel_size_);
        iz = static_cast<int>((p.z - min_z_) / voxel_size_);
        return ix >= 0 && ix < cols_ && iy >= 0 && iy < rows_ && iz >= 0 && iz < depth_layers_;
    }

    void mark_voxel(const Point3& p, int value) {
        int ix = 0;
        int iy = 0;
        int iz = 0;
        if (!world_to_voxel(p, ix, iy, iz)) {
            return;
        }
        const int idx = index(ix, iy, iz);
        if (value == 1) {
            voxels_[idx] = 1;
        } else if (voxels_[idx] != 1) {
            voxels_[idx] = 0;
        }
    }

    void mark_ray_free_then_endpoint(const Point3& origin, const Point3& hit) {
        const double dx = hit.x - origin.x;
        const double dy = hit.y - origin.y;
        const double dz = hit.z - origin.z;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 1e-6) {
            return;
        }

        const int samples = std::max(1, static_cast<int>(std::ceil(dist / (voxel_size_ * 0.5))));
        for (int i = 0; i < samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(samples);
            Point3 p{
                origin.x + dx * t,
                origin.y + dy * t,
                origin.z + dz * t,
            };
            mark_voxel(p, 0);
        }
        mark_voxel(hit, 1);
    }

    void mark_free_range(const Point3& origin, const Point3& direction, double range) {
        const double dist = std::max(0.0, std::min(range, max_depth_));
        const int samples = std::max(1, static_cast<int>(std::ceil(dist / (voxel_size_ * 0.5))));
        for (int i = 0; i <= samples; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(samples) * dist;
            Point3 p{
                origin.x + direction.x * t,
                origin.y + direction.y * t,
                origin.z + direction.z * t,
            };
            mark_voxel(p, 0);
        }
    }

    float depth_at(const sensor_msgs::msg::Image& msg, int u, int v) const {
        const std::size_t offset =
            static_cast<std::size_t>(v) * static_cast<std::size_t>(msg.step) +
            static_cast<std::size_t>(u) *
            (msg.encoding == sensor_msgs::image_encodings::TYPE_16UC1 || msg.encoding == sensor_msgs::image_encodings::MONO16 ? 2U : 4U);

        if (offset >= msg.data.size()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        if (msg.encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            float value = std::numeric_limits<float>::quiet_NaN();
            std::memcpy(&value, msg.data.data() + offset, sizeof(float));
            return value;
        }

        if (msg.encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
            msg.encoding == sensor_msgs::image_encodings::MONO16) {
            std::uint16_t value = 0;
            std::memcpy(&value, msg.data.data() + offset, sizeof(std::uint16_t));
            return static_cast<float>(value) * 0.001f;
        }

        return std::numeric_limits<float>::quiet_NaN();
    }

    Point3 camera_to_world(double x_right, double y_down, double z_forward) const {
        const double forward = z_forward + camera_forward_offset_;
        const double left = -x_right;
        const double up = -y_down;

        const double cos_yaw = std::cos(robot_yaw_);
        const double sin_yaw = std::sin(robot_yaw_);

        return Point3{
            robot_x_ + cos_yaw * forward - sin_yaw * left,
            robot_y_ + sin_yaw * forward + cos_yaw * left,
            camera_height_ + up,
        };
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!has_camera_info_ || !has_odom_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for camera_info/odom | camera_info=%d odom=%d",
                has_camera_info_,
                has_odom_);
            return;
        }

        Point3 origin = camera_to_world(0.0, 0.0, 0.0);
        int rays = 0;
        int valid_hits = 0;

        for (int v = 0; v < static_cast<int>(msg->height); v += pixel_stride_) {
            for (int u = 0; u < static_cast<int>(msg->width); u += pixel_stride_) {
                const float depth_m = depth_at(*msg, u, v);
                ++rays;

                const double z_forward =
                    (std::isfinite(depth_m) && depth_m > 0.05f) ?
                    std::min(static_cast<double>(depth_m), max_depth_) :
                    max_depth_;
                const double x_right = (static_cast<double>(u) - cx_) * z_forward / fx_;
                const double y_down = (static_cast<double>(v) - cy_) * z_forward / fy_;
                const Point3 hit = camera_to_world(x_right, y_down, z_forward);

                Point3 dir{
                    hit.x - origin.x,
                    hit.y - origin.y,
                    hit.z - origin.z,
                };
                const double len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
                if (len < 1e-6) {
                    continue;
                }
                dir.x /= len;
                dir.y /= len;
                dir.z /= len;

                if (!std::isfinite(depth_m) || depth_m <= 0.05f || depth_m > max_depth_) {
                    mark_free_range(origin, dir, max_depth_);
                    continue;
                }

                if (hit.z < min_z_ || hit.z >= max_z_) {
                    mark_free_range(origin, dir, static_cast<double>(depth_m));
                    continue;
                }

                mark_ray_free_then_endpoint(origin, hit);
                ++valid_hits;
            }
        }

        total_depth_frames_++;
        total_rays_ += rays;
        total_valid_hits_ += valid_hits;

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Depth frame mapped | valid_hits=%d rays=%d observed=%d occupied=%d",
            valid_hits,
            rays,
            count_value(0) + count_value(1),
            count_value(1));
    }

    int count_value(int value) const {
        return static_cast<int>(std::count(voxels_.begin(), voxels_.end(), value));
    }

    void timer_callback() {
        const double now = this->now().seconds();
        if (now - last_save_sec_ < save_period_sec_) {
            return;
        }
        last_save_sec_ = now;
        save_csv();
    }

    void save_csv() {
        const std::filesystem::path out_path(output_path_);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::ofstream file(output_path_);
        if (!file.is_open()) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Failed to write voxel map: %s",
                output_path_.c_str());
            return;
        }

        file << "ix,iy,iz,x,y,z,value\n";
        for (int iz = 0; iz < depth_layers_; ++iz) {
            for (int iy = 0; iy < rows_; ++iy) {
                for (int ix = 0; ix < cols_; ++ix) {
                    const int value = voxels_[index(ix, iy, iz)];
                    if (value < 0) {
                        continue;
                    }
                    const double x = min_x_ + (static_cast<double>(ix) + 0.5) * voxel_size_;
                    const double y = min_y_ + (static_cast<double>(iy) + 0.5) * voxel_size_;
                    const double z = min_z_ + (static_cast<double>(iz) + 0.5) * voxel_size_;
                    file << ix << "," << iy << "," << iz << ","
                         << x << "," << y << "," << z << "," << value << "\n";
                }
            }
        }
    }

    std::string depth_topic_;
    std::string camera_info_topic_;
    std::string odom_topic_;
    std::string output_path_;

    double voxel_size_ = 0.2;
    double min_x_ = -3.0;
    double max_x_ = 6.0;
    double min_y_ = -4.0;
    double max_y_ = 4.0;
    double min_z_ = 0.0;
    double max_z_ = 2.4;
    double camera_height_ = 0.31;
    double camera_forward_offset_ = 0.26;
    double max_depth_ = 2.6;
    int pixel_stride_ = 8;
    double save_period_sec_ = 1.0;

    int cols_ = 0;
    int rows_ = 0;
    int depth_layers_ = 0;
    std::vector<int> voxels_;

    double fx_ = 0.0;
    double fy_ = 0.0;
    double cx_ = 0.0;
    double cy_ = 0.0;
    bool has_camera_info_ = false;

    double robot_x_ = 0.0;
    double robot_y_ = 0.0;
    double robot_yaw_ = 0.0;
    bool has_odom_ = false;

    int total_depth_frames_ = 0;
    int total_rays_ = 0;
    int total_valid_hits_ = 0;
    double last_save_sec_ = 0.0;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DepthVoxelMapper>());
    rclcpp::shutdown();
    return 0;
}
