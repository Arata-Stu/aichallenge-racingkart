#ifndef SCAN_GENERATOR_NODE_HPP_
#define SCAN_GENERATOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <v2x_msgs/msg/v2_x_vehicle_position_array.hpp>

#include <array>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <mutex>

struct Point2D {
    double x, y;
};

struct VehicleObstacle {
    std::string vehicle_id;
    Point2D center;
    double yaw;
    rclcpp::Time stamp;
};

class ScanGeneratorNode : public rclcpp::Node
{
public:
    ScanGeneratorNode();

private:
    using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
    using Marker = visualization_msgs::msg::Marker;
    using V2XVehiclePositionArray = v2x_msgs::msg::V2XVehiclePositionArray;

    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_with_obstacles_publisher_;
    rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr pose_subscriber_;
    rclcpp::Subscription<V2XVehiclePositionArray>::SharedPtr v2x_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    rclcpp::Publisher<Marker>::SharedPtr wall_marker_publisher_;
    rclcpp::Publisher<Marker>::SharedPtr hit_points_marker_publisher_;
    rclcpp::Publisher<Marker>::SharedPtr obstacle_marker_publisher_;

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
    std::string self_vehicle_id_;
    double obstacle_vehicle_length_;
    double obstacle_vehicle_width_;
    double obstacle_timeout_sec_;
    double obstacle_heading_min_motion_;

    std::vector<std::pair<Point2D, Point2D>> walls_;
    std::vector<VehicleObstacle> latest_obstacles_;
    std::map<std::string, VehicleObstacle> previous_obstacles_by_id_;
    PoseWithCovarianceStamped::ConstSharedPtr current_pose_;
    std::mutex pose_mutex_;
    std::mutex v2x_mutex_;

    Point2D map_offset_ = {0.0, 0.0};
    bool is_offset_initialized_ = false;

    void declare_and_get_params();
    void load_walls_from_csv();
    void timer_callback();
    void run_simulation(const PoseWithCovarianceStamped::ConstSharedPtr& pose_msg);
    void v2x_callback(const V2XVehiclePositionArray::SharedPtr msg);
    void add_segments_to_scan(
        const std::vector<std::pair<Point2D, Point2D>>& segments,
        const Point2D& robot_pos,
        double robot_yaw,
        double angle_min,
        double angle_increment,
        double hit_points_z,
        sensor_msgs::msg::LaserScan& scan_msg,
        std::vector<geometry_msgs::msg::Point>* hit_points);
    std::vector<VehicleObstacle> get_recent_obstacles(const rclcpp::Time& now);
    std::vector<std::pair<Point2D, Point2D>> build_vehicle_segments(const VehicleObstacle& obstacle) const;
    std::optional<Point2D> get_line_segment_intersection(Point2D p1, Point2D p2, Point2D p3, Point2D p4) const;
    void publish_wall_markers();
    void publish_obstacle_markers(const std::vector<VehicleObstacle>& obstacles);
};

#endif // SCAN_GENERATOR_NODE_HPP_
