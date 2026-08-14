#ifndef RSU_LASERSCAN_GENERATOR_NODE_HPP_
#define RSU_LASERSCAN_GENERATOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <v2x_msgs/msg/v2_x_vehicle_position_array.hpp>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace rsu_laserscan_generator
{

struct Point2D
{
  double x{0.0};
  double y{0.0};
};

struct WallSegment
{
  int64_t lanelet_id{0};
  int64_t way_id{0};
  std::string boundary_type;
  Point2D start;
  Point2D end;
};

struct DynamicVehicle
{
  std::string id;
  Point2D position;
};

struct RsuConfig
{
  std::string id;
  std::string frame_id;
  std::string topic;
  Point2D position;
  double z{0.0};
  double yaw_rad{0.0};
  double fov_rad{0.0};
  double max_range{0.0};
  double range_min{0.0};
  double timer_hz{0.0};
  int num_rays{0};
  int hit_rank{1};
  std::string target_boundary{"any"};
  std::set<int64_t> target_lanelet_ids;
  std::set<int64_t> target_way_ids;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher;
};

class RsuLaserScanGeneratorNode : public rclcpp::Node
{
public:
  RsuLaserScanGeneratorNode();

private:
  using Marker = visualization_msgs::msg::Marker;
  using MarkerArray = visualization_msgs::msg::MarkerArray;

  std::string csv_path_;
  std::string map_frame_id_;
  std::string v2x_topic_;
  double default_fov_deg_{90.0};
  double default_max_range_{35.0};
  double default_range_min_{0.05};
  double default_timer_hz_{20.0};
  int default_num_rays_{361};
  int default_hit_rank_{2};
  double v2x_vehicle_radius_{1.0};
  double v2x_timeout_sec_{1.0};
  bool enable_v2x_vehicles_{true};
  bool publish_static_tf_{true};
  bool debug_{true};

  std::vector<WallSegment> walls_;
  std::vector<RsuConfig> rsus_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_publisher_;
  rclcpp::Subscription<v2x_msgs::msg::V2XVehiclePositionArray>::SharedPtr v2x_subscription_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  std::mutex vehicles_mutex_;
  std::vector<DynamicVehicle> vehicles_;
  std::chrono::steady_clock::time_point last_v2x_update_;
  bool has_v2x_update_{false};

  Point2D map_offset_{0.0, 0.0};
  bool is_offset_initialized_{false};

  void declare_and_get_params();
  void load_rsus_from_params();
  void load_walls_from_csv();
  void on_v2x_positions(const v2x_msgs::msg::V2XVehiclePositionArray::ConstSharedPtr msg);
  void timer_callback();
  void publish_scan(const RsuConfig & rsu, const std::vector<DynamicVehicle> & vehicles);
  void publish_debug_markers();
  void publish_static_transforms();
  std::vector<DynamicVehicle> get_active_vehicles();

  bool wall_matches_rsu(const WallSegment & wall, const RsuConfig & rsu) const;
  std::optional<Point2D> get_line_segment_intersection(
    Point2D p1, Point2D p2, Point2D p3, Point2D p4) const;
  std::optional<double> get_ray_circle_intersection_distance(
    Point2D ray_start, Point2D ray_end, Point2D center, double radius) const;
  Point2D to_internal(Point2D point) const;
  Point2D to_map(Point2D point) const;
};

}  // namespace rsu_laserscan_generator

#endif  // RSU_LASERSCAN_GENERATOR_NODE_HPP_
