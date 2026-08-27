#include "laserscan_generator/laserscan_generator_node.hpp"

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>

ScanGeneratorNode::ScanGeneratorNode() : Node("scan_generator_node")
{
    this->set_parameter(rclcpp::Parameter("use_sim_time", true));
    RCLCPP_INFO(this->get_logger(), "2D Scan Generator Node starting...");
    this->declare_and_get_params();
    this->load_walls_from_csv();

    scan_publisher_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
    scan_without_obstacles_publisher_ =
        this->create_publisher<sensor_msgs::msg::LaserScan>("scan_without_obstacles", 10);
    
    if (debug_) {
        const auto transient_local_qos = rclcpp::QoS(1).transient_local();
        wall_marker_publisher_ = this->create_publisher<Marker>("~/debug/walls", transient_local_qos);
        hit_points_marker_publisher_ = this->create_publisher<Marker>("~/debug/scan_hit_points", 10);
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
        v2x_subscription_ =
            this->create_subscription<v2x_msgs::msg::V2XVehiclePositionArray>(
                v2x_topic_, rclcpp::QoS(1).best_effort(),
                std::bind(&ScanGeneratorNode::on_v2x_positions, this, std::placeholders::_1));
        RCLCPP_INFO(
            this->get_logger(),
            "V2X obstacle scan enabled: topic=%s ego=%s box=%.2f x %.2f m timeout=%.2f s",
            v2x_topic_.c_str(), ego_vehicle_id_.empty() ? "unknown" : ego_vehicle_id_.c_str(),
            v2x_vehicle_length_, v2x_vehicle_width_, v2x_timeout_sec_);
        if (ego_vehicle_id_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "ego_vehicle_id is unknown; set it explicitly to prevent the ego vehicle appearing in its own scan");
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "V2X obstacle scan disabled; publishing static boundaries only.");
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
    this->declare_parameter<std::string>("lidar_frame_id", "laser_link");
    this->declare_parameter<double>("fov_deg", 270.0);
    this->declare_parameter<double>("max_range", 100.0);
    this->declare_parameter<int>("num_rays", 1080);
    this->declare_parameter<double>("range_min", 0.1);
    this->declare_parameter<double>("timer_hz", 10.0);
    this->declare_parameter<bool>("debug", false);
    this->declare_parameter<std::string>("map_frame_id", "map");
    this->declare_parameter<bool>("enable_v2x_obstacles", false);
    this->declare_parameter<std::string>("v2x_topic", "/v2x/vehicle_positions");
    this->declare_parameter<std::string>("ego_vehicle_id", "");
    this->declare_parameter<double>("v2x_vehicle_length", 2.0);
    this->declare_parameter<double>("v2x_vehicle_width", 1.45);
    this->declare_parameter<double>("v2x_heading_min_displacement", 0.05);
    this->declare_parameter<double>("v2x_timeout_sec", 1.0);

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
    this->get_parameter("ego_vehicle_id", ego_vehicle_id_);
    this->get_parameter("v2x_vehicle_length", v2x_vehicle_length_);
    this->get_parameter("v2x_vehicle_width", v2x_vehicle_width_);
    this->get_parameter("v2x_heading_min_displacement", v2x_heading_min_displacement_);
    this->get_parameter("v2x_timeout_sec", v2x_timeout_sec_);

    if (ego_vehicle_id_.empty()) {
        ego_vehicle_id_ = ego_vehicle_id_from_domain();
    }
    v2x_vehicle_length_ = std::max(0.0, v2x_vehicle_length_);
    v2x_vehicle_width_ = std::max(0.0, v2x_vehicle_width_);
    v2x_heading_min_displacement_ = std::max(0.0, v2x_heading_min_displacement_);
    v2x_timeout_sec_ = std::max(0.0, v2x_timeout_sec_);
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

    run_simulation(current_pose_msg, get_active_vehicles());
}

std::string ScanGeneratorNode::ego_vehicle_id_from_domain()
{
    const char* raw_domain_id = std::getenv("ROS_DOMAIN_ID");
    if (raw_domain_id == nullptr) {
        return "";
    }
    try {
        const int domain_id = std::stoi(raw_domain_id);
        if (domain_id > 0) {
            return "d" + std::to_string(domain_id);
        }
    } catch (const std::exception&) {
    }
    return "";
}

void ScanGeneratorNode::on_v2x_positions(
    const v2x_msgs::msg::V2XVehiclePositionArray::ConstSharedPtr msg)
{
    std::vector<DynamicVehicle> vehicles;
    vehicles.reserve(msg->vehicles.size());

    for (const auto& vehicle : msg->vehicles) {
        if (!ego_vehicle_id_.empty() && vehicle.vehicle_id == ego_vehicle_id_) {
            continue;
        }
        const auto& vehicle_frame =
            vehicle.header.frame_id.empty() ? msg->header.frame_id : vehicle.header.frame_id;
        if (!vehicle_frame.empty() && vehicle_frame != map_frame_id_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Ignoring V2X vehicle '%s': frame '%s' does not match map frame '%s'",
                vehicle.vehicle_id.c_str(), vehicle_frame.c_str(), map_frame_id_.c_str());
            continue;
        }
        if (!std::isfinite(vehicle.position.x) || !std::isfinite(vehicle.position.y)) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "Ignoring V2X vehicle '%s': position is not finite", vehicle.vehicle_id.c_str());
            continue;
        }

        DynamicVehicle dynamic_vehicle;
        dynamic_vehicle.id = vehicle.vehicle_id;
        dynamic_vehicle.position = {
            vehicle.position.x - map_offset_.x,
            vehicle.position.y - map_offset_.y};
        vehicles.push_back(std::move(dynamic_vehicle));
    }

    std::lock_guard<std::mutex> lock(vehicles_mutex_);
    for (auto& vehicle : vehicles) {
        const auto previous = std::find_if(
            vehicles_.begin(), vehicles_.end(), [&vehicle](const auto& candidate) {
                return candidate.id == vehicle.id;
            });
        if (previous == vehicles_.end()) {
            continue;
        }
        const double dx = vehicle.position.x - previous->position.x;
        const double dy = vehicle.position.y - previous->position.y;
        if (std::hypot(dx, dy) >= v2x_heading_min_displacement_) {
            vehicle.yaw_rad = std::atan2(dy, dx);
        } else {
            vehicle.yaw_rad = previous->yaw_rad;
        }
    }
    vehicles_ = std::move(vehicles);
    last_v2x_update_ = std::chrono::steady_clock::now();
    has_v2x_update_ = true;
}

std::vector<DynamicVehicle> ScanGeneratorNode::get_active_vehicles()
{
    if (!enable_v2x_obstacles_) {
        return {};
    }
    std::lock_guard<std::mutex> lock(vehicles_mutex_);
    if (!has_v2x_update_) {
        return {};
    }
    if (v2x_timeout_sec_ > 0.0) {
        const double age = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_v2x_update_).count();
        if (age > v2x_timeout_sec_) {
            return {};
        }
    }
    return vehicles_;
}

void ScanGeneratorNode::run_simulation(
    const PoseWithCovarianceStamped::ConstSharedPtr& pose_msg,
    const std::vector<DynamicVehicle>& vehicles)
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

    for (int i = 0; i < num_rays_; ++i) {
        double ray_angle = yaw + angle_min + i * angle_increment;
        Point2D ray_end = {
            robot_pos.x + max_range_ * std::cos(ray_angle),
            robot_pos.y + max_range_ * std::sin(ray_angle)
        };
        double min_dist_sq = max_range_ * max_range_;
        std::optional<Point2D> closest_intersection = std::nullopt;

        for (const auto& wall : walls_) {
            auto intersection = get_line_segment_intersection(robot_pos, ray_end, wall.first, wall.second);
            if (intersection) {
                double dist_sq = std::pow(intersection->x - robot_pos.x, 2) + std::pow(intersection->y - robot_pos.y, 2);
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    closest_intersection = intersection;
                }
            }
        }

        if (min_dist_sq < max_range_ * max_range_) {
            double distance = std::sqrt(min_dist_sq);
            if (distance >= range_min_) {
                scan_msg->ranges[i] = static_cast<float>(distance);
                if (debug_ && closest_intersection) {
                    geometry_msgs::msg::Point p;
                    p.x = closest_intersection->x;
                    p.y = closest_intersection->y;
                    p.z = pose_msg->pose.pose.position.z; 
                    hit_points.push_back(p);
                }
            }
        }
    }
    // Always publish a static-only scan. TinyLiDARNet NPCs subscribe to this
    // topic so nearby vehicles cannot destabilize their learned lane following.
    scan_without_obstacles_publisher_->publish(
        std::make_unique<sensor_msgs::msg::LaserScan>(*scan_msg));

    // The primary scan is selectable per player. Joycon data collection enables
    // this branch and receives other vehicles as V2X-derived oriented boxes.
    if (enable_v2x_obstacles_) {
        for (int i = 0; i < num_rays_; ++i) {
            const double ray_angle = yaw + angle_min + i * angle_increment;
            const Point2D ray_end = {
                robot_pos.x + max_range_ * std::cos(ray_angle),
                robot_pos.y + max_range_ * std::sin(ray_angle)};
            double selected_distance = static_cast<double>(scan_msg->ranges[i]);

            for (const auto& vehicle : vehicles) {
                const auto vehicle_distance =
                    get_ray_box_intersection_distance(robot_pos, ray_end, vehicle);
                if (vehicle_distance && *vehicle_distance >= range_min_ &&
                    *vehicle_distance <= max_range_) {
                    selected_distance = std::min(selected_distance, *vehicle_distance);
                }
            }
            scan_msg->ranges[i] = static_cast<float>(selected_distance);
        }
    }

    scan_publisher_->publish(std::move(scan_msg));

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
}

std::optional<double> ScanGeneratorNode::get_ray_box_intersection_distance(
    Point2D ray_start, Point2D ray_end, const DynamicVehicle& vehicle) const
{
    if (v2x_vehicle_length_ <= 0.0 || v2x_vehicle_width_ <= 0.0) {
        return std::nullopt;
    }

    const double cos_yaw = std::cos(vehicle.yaw_rad);
    const double sin_yaw = std::sin(vehicle.yaw_rad);
    const auto to_vehicle_frame = [vehicle, cos_yaw, sin_yaw](const Point2D point) {
        const double dx = point.x - vehicle.position.x;
        const double dy = point.y - vehicle.position.y;
        return Point2D{
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy};
    };

    const Point2D local_start = to_vehicle_frame(ray_start);
    const Point2D local_end = to_vehicle_frame(ray_end);
    const double direction_x = local_end.x - local_start.x;
    const double direction_y = local_end.y - local_start.y;
    const double ray_length = std::hypot(direction_x, direction_y);
    if (ray_length <= 1e-12) {
        return std::nullopt;
    }

    double t_min = 0.0;
    double t_max = 1.0;
    const auto intersect_slab =
        [&t_min, &t_max](const double origin, const double direction,
                         const double min_value, const double max_value) {
            if (std::abs(direction) <= 1e-12) {
                return origin >= min_value && origin <= max_value;
            }
            double t1 = (min_value - origin) / direction;
            double t2 = (max_value - origin) / direction;
            if (t1 > t2) {
                std::swap(t1, t2);
            }
            t_min = std::max(t_min, t1);
            t_max = std::min(t_max, t2);
            return t_min <= t_max;
        };

    const double half_length = 0.5 * v2x_vehicle_length_;
    const double half_width = 0.5 * v2x_vehicle_width_;
    if (!intersect_slab(local_start.x, direction_x, -half_length, half_length) ||
        !intersect_slab(local_start.y, direction_y, -half_width, half_width)) {
        return std::nullopt;
    }
    if (t_min < 0.0 || t_min > 1.0) {
        return std::nullopt;
    }
    return t_min * ray_length;
}

std::optional<Point2D> ScanGeneratorNode::get_line_segment_intersection(Point2D p1, Point2D p2, Point2D p3, Point2D p4)
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


int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ScanGeneratorNode>());
    rclcpp::shutdown();
    return 0;
}
