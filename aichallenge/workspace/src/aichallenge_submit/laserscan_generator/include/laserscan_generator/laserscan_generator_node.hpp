#ifndef SCAN_GENERATOR_NODE_HPP_
#define SCAN_GENERATOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <v2x_msgs/msg/v2_x_vehicle_position_array.hpp>

#include <chrono>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <mutex>

struct Point2D {
    double x, y;
};

struct DynamicVehicle {
    std::string id;
    Point2D position;
    double yaw_rad{0.0};
};

class ScanGeneratorNode : public rclcpp::Node
{
public:
    ScanGeneratorNode();

private:
    using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
    using Marker = visualization_msgs::msg::Marker;

    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_without_obstacles_publisher_;
    rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr pose_subscriber_;
    rclcpp::Subscription<v2x_msgs::msg::V2XVehiclePositionArray>::SharedPtr v2x_subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    rclcpp::Publisher<Marker>::SharedPtr wall_marker_publisher_;
    rclcpp::Publisher<Marker>::SharedPtr hit_points_marker_publisher_;

    std::string csv_path_;
    std::string lidar_frame_id_;
    double fov_deg_;
    double max_range_;
    int num_rays_;
    double range_min_;
    double timer_hz_;
    bool debug_;
    std::string map_frame_id_;
    bool enable_v2x_obstacles_;
    std::string v2x_topic_;
    std::string ego_vehicle_id_;
    double v2x_vehicle_length_;
    double v2x_vehicle_width_;
    double v2x_heading_min_displacement_;
    double v2x_timeout_sec_;

    std::vector<std::pair<Point2D, Point2D>> walls_;
    PoseWithCovarianceStamped::ConstSharedPtr current_pose_;
    std::mutex pose_mutex_;
    std::vector<DynamicVehicle> vehicles_;
    std::mutex vehicles_mutex_;
    std::chrono::steady_clock::time_point last_v2x_update_;
    bool has_v2x_update_{false};

    Point2D map_offset_ = {0.0, 0.0};
    bool is_offset_initialized_ = false;

    void declare_and_get_params();
    void load_walls_from_csv();
    void timer_callback();
    void run_simulation(
        const PoseWithCovarianceStamped::ConstSharedPtr& pose_msg,
        const std::vector<DynamicVehicle>& vehicles);
    void on_v2x_positions(
        const v2x_msgs::msg::V2XVehiclePositionArray::ConstSharedPtr msg);
    std::vector<DynamicVehicle> get_active_vehicles();
    std::optional<double> get_ray_box_intersection_distance(
        Point2D ray_start, Point2D ray_end, const DynamicVehicle& vehicle) const;
    static std::string ego_vehicle_id_from_domain();
    std::optional<Point2D> get_line_segment_intersection(Point2D p1, Point2D p2, Point2D p3, Point2D p4);
    void publish_wall_markers();
};

#endif // SCAN_GENERATOR_NODE_HPP_
