#include "laserscan_generator/laserscan_generator_node.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

ScanGeneratorNode::ScanGeneratorNode() : Node("scan_generator_node")
{
    this->set_parameter(rclcpp::Parameter("use_sim_time", true));
    RCLCPP_INFO(this->get_logger(), "2D Scan Generator Node starting...");
    this->declare_and_get_params();
    this->load_walls_from_csv();

    scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
    scan_with_obstacles_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan_with_obstacles", 10);
    
    if (debug_) {
        const auto transient_local_qos = rclcpp::QoS(1).transient_local();
        wall_marker_publisher_ = this->create_publisher<Marker>("~/debug/walls", transient_local_qos);
        hit_points_marker_publisher_ = this->create_publisher<Marker>("~/debug/scan_hit_points", 10);
        obstacle_marker_publisher_ = this->create_publisher<Marker>("~/debug/v2x_obstacles", 10);
        RCLCPP_INFO(this->get_logger(), "Debug mode is ON. Publishing markers.");
        publish_wall_markers();
    }
    
    const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    pose_subscriber_ = this->create_subscription<PoseWithCovarianceStamped>(
        "/localization/pose_with_covariance", bv_qos, [this](const PoseWithCovarianceStamped::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            current_pose_ = msg;
        });

    if (enable_v2x_obstacles_) {
        const auto v2x_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
        v2x_subscriber_ = this->create_subscription<V2XVehiclePositionArray>(
            v2x_topic_, v2x_qos, [this](const V2XVehiclePositionArray::SharedPtr msg) {
                this->v2x_callback(msg);
            });
        RCLCPP_INFO(
            this->get_logger(),
            "V2X vehicle obstacles enabled. Subscribing to %s, self_vehicle_id=%s",
            v2x_topic_.c_str(), self_vehicle_id_.c_str());
    }
    
    if (timer_hz_ <= 0.0) {
        RCLCPP_ERROR(this->get_logger(), "timer_hz must be positive. Shutting down.");
        rclcpp::shutdown();
        return;
    }
    const auto timer_period = std::chrono::duration<double>(1.0 / timer_hz_);
    timer_ = this->create_wall_timer(timer_period, std::bind(&ScanGeneratorNode::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "Timer started with a frequency of %.2f Hz.", timer_hz_);
}

void ScanGeneratorNode::declare_and_get_params()
{
    this->declare_parameter<std::string>("csv_path", "/path/to/your/lane_boundaries.csv");
    this->declare_parameter<std::string>("lidar_frame_id", "virtual_lidar");
    this->declare_parameter<double>("fov_deg", 270.0);
    this->declare_parameter<double>("max_range", 100.0);
    this->declare_parameter<int>("num_rays", 1080);
    this->declare_parameter<double>("range_min", 0.1);
    this->declare_parameter<double>("timer_hz", 10.0);
    this->declare_parameter<bool>("debug", false);
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<bool>("enable_v2x_obstacles", true);
    this->declare_parameter<std::string>("v2x_topic", "/v2x/vehicle_positions");
    this->declare_parameter<std::string>("self_vehicle_id", "d1");
    this->declare_parameter<double>("obstacle_vehicle_length", 2.0);
    this->declare_parameter<double>("obstacle_vehicle_width", 1.45);
    this->declare_parameter<double>("obstacle_timeout_sec", 1.0);
    this->declare_parameter<double>("obstacle_heading_min_motion", 0.05);

    this->get_parameter("csv_path", csv_path_);
    this->get_parameter("lidar_frame_id", lidar_frame_id_);
    this->get_parameter("fov_deg", fov_deg_);
    this->get_parameter("max_range", max_range_);
    this->get_parameter("num_rays", num_rays_);
    this->get_parameter("range_min", range_min_);
    this->get_parameter("timer_hz", timer_hz_);
    this->get_parameter("debug", debug_);
    this->get_parameter("map_frame_id", map_frame_id_);
    this->get_parameter("enable_v2x_obstacles", enable_v2x_obstacles_);
    this->get_parameter("v2x_topic", v2x_topic_);
    this->get_parameter("self_vehicle_id", self_vehicle_id_);
    this->get_parameter("obstacle_vehicle_length", obstacle_vehicle_length_);
    this->get_parameter("obstacle_vehicle_width", obstacle_vehicle_width_);
    this->get_parameter("obstacle_timeout_sec", obstacle_timeout_sec_);
    this->get_parameter("obstacle_heading_min_motion", obstacle_heading_min_motion_);
}

void ScanGeneratorNode::load_walls_from_csv()
{
    std::ifstream file(csv_path_);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_path_.c_str());
        rclcpp::shutdown();
        return;
    }

    std::map<int, std::vector<std::pair<int, Point2D>>> way_points;
    std::string line;
    std::getline(file, line); 
    int way_id_idx = 1, seq_idx = 4, x_idx = 5, y_idx = 6;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) { row.push_back(cell); }
        try {
            int way_id = std::stoi(row[way_id_idx]);
            int seq_order = std::stoi(row[seq_idx]);
            Point2D p = {std::stod(row[x_idx]), std::stod(row[y_idx])};

            if (!is_offset_initialized_) {
                map_offset_ = p;
                is_offset_initialized_ = true;
                RCLCPP_INFO(this->get_logger(), "Map offset initialized to (x: %.2f, y: %.2f)", map_offset_.x, map_offset_.y);
            }
            p.x -= map_offset_.x;
            p.y -= map_offset_.y;
            
            way_points[way_id].push_back({seq_order, p});
        } catch (const std::exception &e) {
             RCLCPP_WARN(this->get_logger(), "Could not parse line in CSV: %s", e.what());
        }
    }

    for (auto const& [way_id, points_vec] : way_points) {
        auto sorted_points = points_vec;
        std::sort(sorted_points.begin(), sorted_points.end(), 
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
        for (size_t i = 0; i < sorted_points.size() - 1; ++i) {
            walls_.push_back({sorted_points[i].second, sorted_points[i+1].second});
        }
    }
    RCLCPP_INFO(this->get_logger(), "Loaded %zu wall segments with coordinate offset.", walls_.size());
}

void ScanGeneratorNode::timer_callback()
{
    PoseWithCovarianceStamped::ConstSharedPtr current_pose_msg;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        if (!current_pose_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Waiting for pose message on topic '/localization/pose_with_covariance'...");
            return;
        }
        current_pose_msg = current_pose_;
    }

    run_simulation(current_pose_msg);
}

void ScanGeneratorNode::run_simulation(const PoseWithCovarianceStamped::ConstSharedPtr& pose_msg)
{
    Point2D robot_pos = {
        pose_msg->pose.pose.position.x - map_offset_.x, 
        pose_msg->pose.pose.position.y - map_offset_.y
    };
    
    tf2::Quaternion q;
    tf2::fromMsg(pose_msg->pose.pose.orientation, q);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    
    auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
    scan_msg->header.stamp = this->get_clock()->now();
    scan_msg->header.frame_id = lidar_frame_id_;
    
    if (num_rays_ < 2) {
        RCLCPP_WARN(this->get_logger(), "num_rays must be at least 2. Skipping simulation.");
        return;
    }

    const double fov_rad = fov_deg_ * M_PI / 180.0;
    const double angle_min = -fov_rad / 2.0;
    const double angle_max = fov_rad / 2.0;
    const double angle_increment = fov_rad / (num_rays_ - 1);

    scan_msg->angle_min = angle_min;
    scan_msg->angle_max = angle_max;
    scan_msg->angle_increment = angle_increment;
    scan_msg->time_increment = 0.0;
    scan_msg->scan_time = 1.0 / timer_hz_;
    scan_msg->range_min = range_min_;
    scan_msg->range_max = max_range_;
    scan_msg->ranges.resize(num_rays_, std::numeric_limits<float>::infinity());
    
    std::vector<geometry_msgs::msg::Point> hit_points;
    add_segments_to_scan(
        walls_, robot_pos, yaw, angle_min, angle_increment, pose_msg->pose.pose.position.z,
        *scan_msg, debug_ ? &hit_points : nullptr);

    auto scan_with_obstacles_msg = std::make_unique<sensor_msgs::msg::LaserScan>(*scan_msg);
    std::vector<VehicleObstacle> obstacles;
    std::vector<std::pair<Point2D, Point2D>> obstacle_segments;

    if (enable_v2x_obstacles_) {
        obstacles = get_recent_obstacles(rclcpp::Time(scan_msg->header.stamp));
        for (const auto& obstacle : obstacles) {
            const auto segments = build_vehicle_segments(obstacle);
            obstacle_segments.insert(obstacle_segments.end(), segments.begin(), segments.end());
        }

        if (!obstacle_segments.empty()) {
            add_segments_to_scan(
                obstacle_segments, robot_pos, yaw, angle_min, angle_increment,
                pose_msg->pose.pose.position.z, *scan_with_obstacles_msg, nullptr);
        }
    }

    scan_publisher_->publish(std::move(scan_msg));
    scan_with_obstacles_publisher_->publish(std::move(scan_with_obstacles_msg));

    if (debug_ && !hit_points.empty()) {
        Marker points_marker;
        points_marker.header.frame_id = pose_msg->header.frame_id;
        points_marker.header.stamp = this->get_clock()->now();
        points_marker.ns = "hit_points";
        points_marker.id = 1;
        points_marker.type = Marker::POINTS;
        points_marker.action = Marker::ADD;
        points_marker.pose.orientation.w = 1.0;
        points_marker.scale.x = 0.1;
        points_marker.scale.y = 0.1;
        points_marker.color.r = 1.0f;
        points_marker.color.a = 1.0;
        points_marker.points = hit_points;
        hit_points_marker_publisher_->publish(points_marker);
    }

    if (debug_ && enable_v2x_obstacles_) {
        publish_obstacle_markers(obstacles);
    }
}

void ScanGeneratorNode::v2x_callback(const V2XVehiclePositionArray::SharedPtr msg)
{
    const auto now = this->get_clock()->now();
    rclcpp::Time array_stamp(msg->header.stamp);
    if (array_stamp.nanoseconds() == 0) {
        array_stamp = now;
    }

    std::vector<VehicleObstacle> next_obstacles;
    std::lock_guard<std::mutex> lock(v2x_mutex_);

    for (const auto& vehicle : msg->vehicles) {
        if (vehicle.vehicle_id == self_vehicle_id_) {
            continue;
        }

        if (!std::isfinite(vehicle.position.x) || !std::isfinite(vehicle.position.y)) {
            continue;
        }

        const std::string frame_id =
            vehicle.header.frame_id.empty() ? msg->header.frame_id : vehicle.header.frame_id;
        if (!frame_id.empty() && frame_id != map_frame_id_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "V2X vehicle frame_id is '%s', expected '%s'. Using coordinates as-is.",
                frame_id.c_str(), map_frame_id_.c_str());
        }

        VehicleObstacle obstacle;
        obstacle.vehicle_id = vehicle.vehicle_id;
        obstacle.center = {
            vehicle.position.x - map_offset_.x,
            vehicle.position.y - map_offset_.y
        };

        rclcpp::Time vehicle_stamp(vehicle.header.stamp);
        obstacle.stamp = vehicle_stamp.nanoseconds() == 0 ? array_stamp : vehicle_stamp;

        obstacle.yaw = 0.0;
        const auto previous_it = previous_obstacles_by_id_.find(obstacle.vehicle_id);
        if (previous_it != previous_obstacles_by_id_.end()) {
            const auto& previous = previous_it->second;
            const double dx = obstacle.center.x - previous.center.x;
            const double dy = obstacle.center.y - previous.center.y;
            const double motion = std::hypot(dx, dy);
            obstacle.yaw = motion >= obstacle_heading_min_motion_
                ? std::atan2(dy, dx)
                : previous.yaw;
        }

        next_obstacles.push_back(obstacle);
    }

    previous_obstacles_by_id_.clear();
    for (const auto& obstacle : next_obstacles) {
        previous_obstacles_by_id_[obstacle.vehicle_id] = obstacle;
    }
    latest_obstacles_ = std::move(next_obstacles);
}

void ScanGeneratorNode::add_segments_to_scan(
    const std::vector<std::pair<Point2D, Point2D>>& segments,
    const Point2D& robot_pos,
    double robot_yaw,
    double angle_min,
    double angle_increment,
    double hit_points_z,
    sensor_msgs::msg::LaserScan& scan_msg,
    std::vector<geometry_msgs::msg::Point>* hit_points)
{
    for (int i = 0; i < num_rays_; ++i) {
        const double ray_angle = robot_yaw + angle_min + i * angle_increment;
        const Point2D ray_end = {
            robot_pos.x + max_range_ * std::cos(ray_angle),
            robot_pos.y + max_range_ * std::sin(ray_angle)
        };

        double best_distance = std::isfinite(scan_msg.ranges[i])
            ? static_cast<double>(scan_msg.ranges[i])
            : max_range_;
        std::optional<Point2D> closest_intersection = std::nullopt;

        for (const auto& segment : segments) {
            const auto intersection =
                get_line_segment_intersection(robot_pos, ray_end, segment.first, segment.second);
            if (!intersection) {
                continue;
            }

            const double distance =
                std::hypot(intersection->x - robot_pos.x, intersection->y - robot_pos.y);
            if (distance >= range_min_ && distance < best_distance) {
                best_distance = distance;
                closest_intersection = intersection;
            }
        }

        if (closest_intersection) {
            scan_msg.ranges[i] = static_cast<float>(best_distance);
            if (hit_points != nullptr) {
                geometry_msgs::msg::Point p;
                p.x = closest_intersection->x;
                p.y = closest_intersection->y;
                p.z = hit_points_z;
                hit_points->push_back(p);
            }
        }
    }
}

std::vector<VehicleObstacle> ScanGeneratorNode::get_recent_obstacles(const rclcpp::Time& now)
{
    std::vector<VehicleObstacle> obstacles;
    std::lock_guard<std::mutex> lock(v2x_mutex_);

    for (const auto& obstacle : latest_obstacles_) {
        if (obstacle_timeout_sec_ > 0.0) {
            const double age_sec = (now - obstacle.stamp).seconds();
            if (age_sec > obstacle_timeout_sec_) {
                continue;
            }
        }
        obstacles.push_back(obstacle);
    }

    return obstacles;
}

std::vector<std::pair<Point2D, Point2D>> ScanGeneratorNode::build_vehicle_segments(
    const VehicleObstacle& obstacle) const
{
    const double half_length = obstacle_vehicle_length_ * 0.5;
    const double half_width = obstacle_vehicle_width_ * 0.5;
    const double cos_yaw = std::cos(obstacle.yaw);
    const double sin_yaw = std::sin(obstacle.yaw);

    const Point2D forward = {cos_yaw * half_length, sin_yaw * half_length};
    const Point2D left = {-sin_yaw * half_width, cos_yaw * half_width};

    const std::array<Point2D, 4> corners = {{
        {obstacle.center.x + forward.x + left.x, obstacle.center.y + forward.y + left.y},
        {obstacle.center.x + forward.x - left.x, obstacle.center.y + forward.y - left.y},
        {obstacle.center.x - forward.x - left.x, obstacle.center.y - forward.y - left.y},
        {obstacle.center.x - forward.x + left.x, obstacle.center.y - forward.y + left.y}
    }};

    return {
        {corners[0], corners[1]},
        {corners[1], corners[2]},
        {corners[2], corners[3]},
        {corners[3], corners[0]}
    };
}

std::optional<Point2D> ScanGeneratorNode::get_line_segment_intersection(
    Point2D p1, Point2D p2, Point2D p3, Point2D p4) const
{
    double x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;
    double x3 = p3.x, y3 = p3.y, x4 = p4.x, y4 = p4.y;
    double den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (den == 0) return std::nullopt;
    double t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4);
    double u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3));
    double t = t_num / den;
    double u = u_num / den;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        return Point2D{x1 + t * (x2 - x1), y1 + t * (y2 - y1)};
    }
    return std::nullopt;
}

void ScanGeneratorNode::publish_wall_markers()
{
    Marker wall_marker;
    wall_marker.header.frame_id = map_frame_id_;
    wall_marker.header.stamp = this->get_clock()->now();
    wall_marker.ns = "walls";
    wall_marker.id = 0;
    wall_marker.type = Marker::LINE_LIST;
    wall_marker.action = Marker::ADD;
    wall_marker.pose.orientation.w = 1.0;
    wall_marker.scale.x = 0.05;
    wall_marker.color.g = 1.0f;
    wall_marker.color.a = 0.8;

    for (const auto& wall : walls_) {
        geometry_msgs::msg::Point p1, p2;
        p1.x = wall.first.x;
        p1.y = wall.first.y;
        p1.z = 0.0;
        p2.x = wall.second.x;
        p2.y = wall.second.y;
        p2.z = 0.0;
        wall_marker.points.push_back(p1);
        wall_marker.points.push_back(p2);
    }
    wall_marker_publisher_->publish(wall_marker);
    RCLCPP_INFO(this->get_logger(), "Published %zu wall segments to marker topic.", walls_.size());
}

void ScanGeneratorNode::publish_obstacle_markers(const std::vector<VehicleObstacle>& obstacles)
{
    if (!obstacle_marker_publisher_) {
        return;
    }

    Marker obstacle_marker;
    obstacle_marker.header.frame_id = map_frame_id_;
    obstacle_marker.header.stamp = this->get_clock()->now();
    obstacle_marker.ns = "v2x_vehicle_obstacles";
    obstacle_marker.id = 0;
    obstacle_marker.type = Marker::LINE_LIST;
    obstacle_marker.action = obstacles.empty() ? Marker::DELETE : Marker::ADD;
    obstacle_marker.pose.orientation.w = 1.0;
    obstacle_marker.scale.x = 0.08;
    obstacle_marker.color.r = 1.0f;
    obstacle_marker.color.g = 0.55f;
    obstacle_marker.color.a = 0.95f;

    for (const auto& obstacle : obstacles) {
        const auto segments = build_vehicle_segments(obstacle);
        for (const auto& segment : segments) {
            geometry_msgs::msg::Point p1;
            geometry_msgs::msg::Point p2;
            p1.x = segment.first.x;
            p1.y = segment.first.y;
            p1.z = 0.2;
            p2.x = segment.second.x;
            p2.y = segment.second.y;
            p2.z = 0.2;
            obstacle_marker.points.push_back(p1);
            obstacle_marker.points.push_back(p2);
        }
    }

    obstacle_marker_publisher_->publish(obstacle_marker);
}


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ScanGeneratorNode>());
    rclcpp::shutdown();
    return 0;
}
